use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Barrier, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use vllm_cpp::{
    compose_video_mux_argv, Device, EmbeddingEngine, Engine, Error, FinishReason, Request,
    RequestOutcome, SamplingParams, StreamControl, StructuredOutput, TranscriptionEngine,
    TranscriptionInput, VideoEngine, VideoMuxParams,
};

fn model_path() -> Option<PathBuf> {
    let path = std::env::var_os("VLLM_CPP_TEST_MODEL").map(PathBuf::from)?;
    assert!(
        path.is_dir(),
        "VLLM_CPP_TEST_MODEL is not a directory: {}",
        path.display()
    );
    Some(path)
}

fn with_engine(test: impl FnOnce(&Engine, &Path)) {
    let Some(path) = model_path() else {
        eprintln!("skipping model test; set VLLM_CPP_TEST_MODEL with `just setup-test-model`");
        return;
    };
    if std::env::var_os("VLLM_CPP_TEST_ISOLATED_ENGINE").is_some() {
        let engine = load_engine(&path);
        test(&engine, &path);
        return;
    }
    static ENGINE: OnceLock<Mutex<Engine>> = OnceLock::new();
    let engine = ENGINE.get_or_init(|| Mutex::new(load_engine(&path)));
    let engine = engine.lock().expect("model test engine lock");
    test(&engine, &path);
}

fn load_engine(path: &Path) -> Engine {
    Engine::builder(path)
        .num_blocks(64)
        .max_model_len(256)
        .max_num_seqs(2)
        .max_num_batched_tokens(256)
        .load()
        .expect("load Qwen3-0.6B")
}

fn wait_until_done(request: &Request) {
    let deadline = Instant::now() + Duration::from_secs(180);
    while !request.is_done() {
        assert!(
            Instant::now() < deadline,
            "request did not finish before timeout"
        );
        thread::sleep(Duration::from_millis(1));
    }
}

fn native_fixture(relative: &str) -> Option<PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../vllm-cpp-sys/vllm.cpp/tests/vllm/models/fixtures")
        .join(relative);
    if path.exists() {
        Some(path)
    } else {
        eprintln!(
            "skipping native fixture test; fixture is absent: {}",
            path.display()
        );
        None
    }
}

#[derive(Debug, PartialEq)]
struct FixtureWav {
    samples: Vec<f32>,
    sample_rate: u32,
}

fn decode_pcm16_mono_wav(bytes: &[u8]) -> Result<FixtureWav, String> {
    if bytes.len() < 12 {
        return Err("truncated RIFF/WAVE header".to_owned());
    }
    if &bytes[..4] != b"RIFF" {
        return Err("missing RIFF identifier".to_owned());
    }
    if &bytes[8..12] != b"WAVE" {
        return Err("missing WAVE identifier".to_owned());
    }

    let riff_size = usize::try_from(u32::from_le_bytes(
        bytes[4..8]
            .try_into()
            .map_err(|_| "truncated RIFF size".to_owned())?,
    ))
    .map_err(|_| "RIFF size exceeds address space".to_owned())?;
    let riff_end = 8usize
        .checked_add(riff_size)
        .ok_or_else(|| "RIFF size overflows address space".to_owned())?;
    if riff_end < 12 {
        return Err("RIFF size does not include the WAVE identifier".to_owned());
    }
    if riff_end > bytes.len() {
        return Err("declared RIFF end exceeds input length".to_owned());
    }
    if riff_end != bytes.len() {
        return Err("bytes remain after the declared RIFF end".to_owned());
    }

    let mut fmt = None;
    let mut data = None;
    let mut offset = 12usize;
    while offset < riff_end {
        let header_end = offset
            .checked_add(8)
            .ok_or_else(|| "chunk header offset overflows address space".to_owned())?;
        if header_end > riff_end {
            return Err(format!("truncated chunk header at byte {offset}"));
        }
        let name: [u8; 4] = bytes[offset..offset + 4]
            .try_into()
            .map_err(|_| format!("truncated chunk identifier at byte {offset}"))?;
        let size = usize::try_from(u32::from_le_bytes(
            bytes[offset + 4..header_end]
                .try_into()
                .map_err(|_| format!("truncated chunk size at byte {offset}"))?,
        ))
        .map_err(|_| format!("chunk length exceeds address space at byte {offset}"))?;
        let body_end = header_end
            .checked_add(size)
            .ok_or_else(|| format!("chunk length overflows address space at byte {offset}"))?;
        if body_end > riff_end {
            return Err(format!("chunk length exceeds RIFF bounds at byte {offset}"));
        }
        let padded_end = body_end
            .checked_add(size & 1)
            .ok_or_else(|| format!("chunk padding overflows address space at byte {offset}"))?;
        if padded_end > riff_end {
            return Err(format!("missing padding byte for chunk at byte {offset}"));
        }

        match &name {
            b"fmt " => {
                if fmt.is_some() {
                    return Err("duplicate fmt chunk".to_owned());
                }
                if size < 16 {
                    return Err(format!("fmt chunk is too short: {size} bytes"));
                }
                let body = &bytes[header_end..body_end];
                let format = u16::from_le_bytes([body[0], body[1]]);
                let channels = u16::from_le_bytes([body[2], body[3]]);
                let sample_rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                let byte_rate = u32::from_le_bytes(body[8..12].try_into().unwrap());
                let block_align = u16::from_le_bytes([body[12], body[13]]);
                let bits_per_sample = u16::from_le_bytes([body[14], body[15]]);
                if format != 1 {
                    return Err(format!("WAV format code must be PCM 1, found {format}"));
                }
                if channels != 1 {
                    return Err(format!("WAV must be mono, found {channels} channels"));
                }
                if sample_rate != 16_000 {
                    return Err(format!(
                        "WAV sample rate must be 16000, found {sample_rate}"
                    ));
                }
                if bits_per_sample != 16 {
                    return Err(format!(
                        "WAV sample width must be 16 bits, found {bits_per_sample}"
                    ));
                }
                if block_align != 2 {
                    return Err(format!(
                        "WAV block alignment must be 2, found {block_align}"
                    ));
                }
                if byte_rate != 32_000 {
                    return Err(format!("WAV byte rate must be 32000, found {byte_rate}"));
                }
                fmt = Some(sample_rate);
            }
            b"data" => {
                if data.is_some() {
                    return Err("duplicate data chunk".to_owned());
                }
                if size == 0 {
                    return Err("WAV data chunk is empty".to_owned());
                }
                if size % 2 != 0 {
                    return Err(format!("WAV data chunk has odd size {size}"));
                }
                data = Some((header_end, body_end));
            }
            _ => {}
        }
        offset = padded_end;
    }

    let sample_rate = fmt.ok_or_else(|| "missing fmt chunk".to_owned())?;
    let (data_start, data_end) = data.ok_or_else(|| "missing data chunk".to_owned())?;
    let samples = bytes[data_start..data_end]
        .chunks_exact(2)
        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]) as f32 / 32768.0)
        .collect();
    Ok(FixtureWav {
        samples,
        sample_rate,
    })
}

fn read_pcm16_mono_wav(path: &Path) -> Result<FixtureWav, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("failed to read fixture WAV {}: {error}", path.display()))?;
    decode_pcm16_mono_wav(&bytes)
}

#[cfg(test)]
fn fixture_fmt_chunk() -> Vec<u8> {
    let mut fmt = Vec::new();
    fmt.extend_from_slice(&1u16.to_le_bytes());
    fmt.extend_from_slice(&1u16.to_le_bytes());
    fmt.extend_from_slice(&16_000u32.to_le_bytes());
    fmt.extend_from_slice(&32_000u32.to_le_bytes());
    fmt.extend_from_slice(&2u16.to_le_bytes());
    fmt.extend_from_slice(&16u16.to_le_bytes());
    fmt
}

#[cfg(test)]
fn fixture_wav_bytes(chunks: &[([u8; 4], Vec<u8>)]) -> Vec<u8> {
    let mut body = b"WAVE".to_vec();
    for (name, chunk) in chunks {
        body.extend_from_slice(name);
        body.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        body.extend_from_slice(chunk);
        if chunk.len() % 2 != 0 {
            body.push(0);
        }
    }
    let mut bytes = b"RIFF".to_vec();
    bytes.extend_from_slice(&(body.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&body);
    bytes
}

#[cfg(test)]
fn assert_wav_error(bytes: &[u8], expected: &str) {
    let error = decode_pcm16_mono_wav(bytes).expect_err("malformed WAV must fail");
    assert!(
        error.contains(expected),
        "expected {expected:?} in WAV error {error:?}"
    );
}

#[test]
fn wav_parser_accepts_unknown_padding_and_data_before_fmt() {
    let data = [i16::MIN.to_le_bytes(), 0i16.to_le_bytes()].concat();
    let bytes = fixture_wav_bytes(&[
        (*b"JUNK", vec![7]),
        (*b"data", data),
        (*b"fmt ", fixture_fmt_chunk()),
    ]);
    let wav = decode_pcm16_mono_wav(&bytes).expect("valid PCM fixture WAV");
    assert_eq!(wav.sample_rate, 16_000);
    assert_eq!(wav.samples, [-1.0, 0.0]);
}

#[test]
fn wav_parser_rejects_identifiers_and_riff_bounds() {
    assert_wav_error(&[], "truncated RIFF/WAVE header");

    let valid = fixture_wav_bytes(&[(*b"fmt ", fixture_fmt_chunk()), (*b"data", vec![0, 0])]);
    let mut wrong_riff = valid.clone();
    wrong_riff[..4].copy_from_slice(b"RIFX");
    assert_wav_error(&wrong_riff, "missing RIFF identifier");
    let mut wrong_wave = valid.clone();
    wrong_wave[8..12].copy_from_slice(b"WVAE");
    assert_wav_error(&wrong_wave, "missing WAVE identifier");

    let mut short_riff = valid.clone();
    short_riff[4..8].copy_from_slice(&3u32.to_le_bytes());
    assert_wav_error(
        &short_riff,
        "RIFF size does not include the WAVE identifier",
    );

    let mut truncated = valid.clone();
    let declared = u32::from_le_bytes(truncated[4..8].try_into().unwrap());
    truncated[4..8].copy_from_slice(&(declared + 1).to_le_bytes());
    assert_wav_error(&truncated, "declared RIFF end exceeds input length");

    let mut trailing = valid;
    trailing.push(0);
    assert_wav_error(&trailing, "bytes remain after the declared RIFF end");
}

#[test]
fn wav_parser_rejects_truncated_chunks_and_padding() {
    let mut header = b"RIFF".to_vec();
    header.extend_from_slice(&7u32.to_le_bytes());
    header.extend_from_slice(b"WAVEabc");
    assert_wav_error(&header, "truncated chunk header");

    let mut body = b"WAVEdata".to_vec();
    body.extend_from_slice(&4u32.to_le_bytes());
    body.extend_from_slice(&[0, 0]);
    let mut truncated_body = b"RIFF".to_vec();
    truncated_body.extend_from_slice(&(body.len() as u32).to_le_bytes());
    truncated_body.extend_from_slice(&body);
    assert_wav_error(&truncated_body, "chunk length exceeds RIFF bounds");

    let mut odd_body = b"WAVEJUNK".to_vec();
    odd_body.extend_from_slice(&1u32.to_le_bytes());
    odd_body.push(7);
    let mut missing_padding = b"RIFF".to_vec();
    missing_padding.extend_from_slice(&(odd_body.len() as u32).to_le_bytes());
    missing_padding.extend_from_slice(&odd_body);
    assert_wav_error(&missing_padding, "missing padding byte");

    let mut huge_body = b"WAVEdata".to_vec();
    huge_body.extend_from_slice(&u32::MAX.to_le_bytes());
    let mut huge = b"RIFF".to_vec();
    huge.extend_from_slice(&(huge_body.len() as u32).to_le_bytes());
    huge.extend_from_slice(&huge_body);
    assert_wav_error(&huge, "chunk length exceeds RIFF bounds");
}

#[test]
fn wav_parser_rejects_invalid_format_metadata() {
    let cases = [
        (0usize, 3u16.to_le_bytes().to_vec(), "format code"),
        (2, 2u16.to_le_bytes().to_vec(), "mono"),
        (4, 8_000u32.to_le_bytes().to_vec(), "sample rate"),
        (8, 16_000u32.to_le_bytes().to_vec(), "byte rate"),
        (12, 4u16.to_le_bytes().to_vec(), "block alignment"),
        (14, 8u16.to_le_bytes().to_vec(), "sample width"),
    ];
    for (offset, replacement, expected) in cases {
        let mut fmt = fixture_fmt_chunk();
        fmt[offset..offset + replacement.len()].copy_from_slice(&replacement);
        let bytes = fixture_wav_bytes(&[(*b"fmt ", fmt), (*b"data", vec![0, 0])]);
        assert_wav_error(&bytes, expected);
    }

    let short = fixture_wav_bytes(&[(*b"fmt ", vec![0; 15]), (*b"data", vec![0, 0])]);
    assert_wav_error(&short, "fmt chunk is too short");
}

#[test]
fn wav_parser_rejects_missing_duplicate_and_invalid_data_chunks() {
    let fmt = fixture_fmt_chunk();
    assert_wav_error(
        &fixture_wav_bytes(&[(*b"data", vec![0, 0])]),
        "missing fmt chunk",
    );
    assert_wav_error(
        &fixture_wav_bytes(&[(*b"fmt ", fmt.clone())]),
        "missing data chunk",
    );
    assert_wav_error(
        &fixture_wav_bytes(&[
            (*b"fmt ", fmt.clone()),
            (*b"fmt ", fmt.clone()),
            (*b"data", vec![0, 0]),
        ]),
        "duplicate fmt chunk",
    );
    assert_wav_error(
        &fixture_wav_bytes(&[
            (*b"fmt ", fmt.clone()),
            (*b"data", vec![0, 0]),
            (*b"data", vec![0, 0]),
        ]),
        "duplicate data chunk",
    );
    assert_wav_error(
        &fixture_wav_bytes(&[(*b"fmt ", fmt.clone()), (*b"data", Vec::new())]),
        "data chunk is empty",
    );
    assert_wav_error(
        &fixture_wav_bytes(&[(*b"fmt ", fmt), (*b"data", vec![0])]),
        "data chunk has odd size",
    );
}

#[test]
fn pretokenized_completion_matches_qwen_hello_and_reports_truncation() {
    with_engine(|engine, _| {
        let params = SamplingParams::greedy().max_tokens(4);
        let full = engine
            .complete_tokens(&[9707], &params, 8, true)
            .expect("full pre-tokenized completion");
        assert_eq!(full.token_ids.len(), 4);
        assert!(!full.truncated);
        let details = full.completion.as_ref().expect("completion details");
        assert_eq!(details.prompt_tokens, 1);
        assert_eq!(details.completion_tokens, 4);
        assert_eq!(
            details,
            &engine
                .complete("Hello", &params)
                .expect("string-prompt parity completion")
        );

        let small = engine
            .complete_tokens(&[9707], &params, 2, false)
            .expect("truncated pre-tokenized completion");
        assert_eq!(small.token_ids, full.token_ids[..2]);
        assert!(small.completion.is_none());
        assert!(small.truncated);

        let zero = engine
            .complete_tokens(&[9707], &params, 0, false)
            .expect("zero-capacity pre-tokenized completion");
        assert!(zero.token_ids.is_empty());
        assert!(zero.completion.is_none());
        assert!(zero.truncated);
    });
}

#[test]
fn pretokenized_logits_processor_controls_tokens_and_panic_leaves_engine_reusable() {
    with_engine(|engine, _| {
        let params = SamplingParams::greedy()
            .max_tokens(3)
            .logits_processor(|_, logits| {
                logits.fill(f32::NEG_INFINITY);
                logits[10] = f32::INFINITY;
            });
        let forced = engine
            .complete_tokens(&[9707], &params, 3, false)
            .expect("forced token completion");
        assert_eq!(forced.token_ids, [10, 10, 10]);

        let panicking = SamplingParams::greedy()
            .max_tokens(1)
            .logits_processor(|_, _| panic!("intentional token processor panic"));
        assert_eq!(
            engine
                .complete_tokens(&[9707], &panicking, 1, true)
                .expect_err("processor panic"),
            Error::LogitsProcessorPanicked
        );
        engine
            .complete_tokens(&[9707], &SamplingParams::greedy().max_tokens(1), 1, false)
            .expect("engine remains reusable");
    });
}

#[test]
fn committed_transcription_fixture_supports_path_pcm_and_wrong_task() {
    let Some(root) = native_fixture("parakeet_e2e") else {
        return;
    };
    let model = root.join("ctc");
    let wav = root.join("audio.wav");
    let mut engine = TranscriptionEngine::builder(&model)
        .device(Device::Cpu)
        .load()
        .expect("load CTC fixture on CPU");
    let cuda_error = TranscriptionEngine::builder(&model)
        .device(Device::Cuda)
        .load()
        .err()
        .expect("transcription fixture must refuse explicit CUDA");
    assert!(
        matches!(cuda_error, Error::InvalidArgument { .. }),
        "{cuda_error:?}"
    );
    let from_path = engine
        .transcribe(TranscriptionInput::WavFile(&wav))
        .expect("transcribe fixture path");
    assert_eq!(from_path.token_ids, [3, 4, 3]);
    assert_eq!(from_path.text.as_deref(), Some("atheat"));

    let fixture = read_pcm16_mono_wav(&wav).expect("parse fixture WAV");
    let from_pcm = engine
        .transcribe(TranscriptionInput::Pcm {
            samples: &fixture.samples,
            sample_rate: fixture.sample_rate,
        })
        .expect("transcribe fixture PCM");
    assert_eq!(from_pcm, from_path);

    let text = Engine::load(&model).expect("task-neutral load of CTC fixture");
    assert!(matches!(
        text.complete_tokens(&[0], &SamplingParams::greedy().max_tokens(1), 1, false),
        Err(Error::InvalidArgument { .. })
    ));

    if let Some(embedding_model) = native_fixture("llama_embed_e2e") {
        let mut wrong =
            TranscriptionEngine::load(&embedding_model).expect("task-neutral embedding load");
        assert!(matches!(
            wrong.transcribe(TranscriptionInput::WavFile(&wav)),
            Err(Error::InvalidArgument { .. })
        ));
    }
}

#[test]
fn committed_embedding_fixture_preserves_shape_order_ownership_and_wrong_task() {
    let Some(model) = native_fixture("llama_embed_e2e") else {
        return;
    };
    let mut engine = EmbeddingEngine::builder(&model)
        .block_size(16)
        .num_blocks(32)
        .max_model_len(128)
        .max_num_seqs(2)
        .max_num_batched_tokens(128)
        .prefix_caching(vllm_cpp::Toggle::Off)
        .device(Device::Cpu)
        .gpu_memory_utilization(1.25)
        .kv_cache_memory_bytes(4096)
        .load()
        .expect("load configured embedding fixture");
    let result = engine
        .embed(["the quick brown fox", "the lazy dog"])
        .expect("embed fixture inputs");
    assert_eq!(result.n_embeddings(), 2);
    assert_eq!(result.dimension(), 64);
    assert!(result.prompt_tokens() > 0);
    assert_ne!(result.row(0), result.row(1));
    for row in result.rows() {
        let l2 = row
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((l2 - 1.0).abs() < 1e-5);
    }
    drop(engine);
    assert_eq!(result.values().len(), 128);

    let text = Engine::load(&model).expect("task-neutral embedding load");
    assert!(matches!(
        text.complete_tokens(&[0], &SamplingParams::greedy().max_tokens(1), 1, false),
        Err(Error::InvalidArgument { .. })
    ));

    if let Some(transcription_root) = native_fixture("parakeet_e2e") {
        let mut wrong = EmbeddingEngine::load(transcription_root.join("ctc"))
            .expect("task-neutral transcription load");
        assert!(matches!(
            wrong.embed(["hello"]),
            Err(Error::InvalidArgument { .. })
        ));
    }

    let mut engine = EmbeddingEngine::load(&model).expect("reload embedding fixture");
    engine
        .embed(["the fox"])
        .expect("embedding owner remains usable");
}

#[test]
fn committed_video_mux_goldens_preserve_exact_argument_boundaries() {
    let Some(root) = native_fixture("minimax_h3_video_fold") else {
        return;
    };

    let audio = compose_video_mux_argv(
        &VideoMuxParams::new(root.join("frame_%06d.ppm"), root.join("video.mp4"))
            .audio_path(root.join("audio.wav")),
    )
    .expect("compose committed audio mux golden");
    let golden =
        std::fs::read_to_string(root.join("golden_mux_argv.txt")).expect("read audio mux golden");
    let expected = golden
        .split_ascii_whitespace()
        .map(|argument| {
            argument.strip_prefix("W/").map_or_else(
                || OsString::from(argument),
                |relative| root.join(relative).into_os_string(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(audio.args(), expected);

    let silent = compose_video_mux_argv(&VideoMuxParams::new("frames_%06d.ppm", "silent.mp4"))
        .expect("compose committed silent mux golden");
    let golden = std::fs::read_to_string(root.join("golden_mux_argv_silent.txt"))
        .expect("read silent mux golden");
    let expected = golden
        .split_ascii_whitespace()
        .map(OsString::from)
        .collect::<Vec<_>>();
    assert_eq!(silent.args(), expected);
}

#[test]
fn committed_parakeet_directory_is_rejected_as_video_dit() {
    let Some(root) = native_fixture("parakeet_e2e") else {
        return;
    };
    let model = root.join("ctc");
    let error = VideoEngine::builder(&model)
        .video_vae_path(&model)
        .audio_vae_path(&model)
        .load()
        .err()
        .expect("Parakeet must not load as a video DiT");
    match error {
        Error::ModelLoad { message } => {
            assert!(message.contains("vllm_engine_load"), "{message}");
        }
        other => panic!("unexpected wrong-direction error: {other:?}"),
    }
}

#[test]
fn greedy_completion_and_streaming_match() {
    with_engine(|engine, _| {
        let params = SamplingParams::greedy().max_tokens(8);
        let completion = engine
            .complete("The capital of France is", &params)
            .expect("blocking completion");
        assert!(!completion.text.is_empty());
        assert_eq!(completion.completion_tokens, 8);
        assert_eq!(completion.finish_reason, Some(FinishReason::Length));

        let mut streamed = String::new();
        let outcome = engine
            .complete_stream("The capital of France is", &params, |event| {
                streamed.push_str(&event.delta);
                StreamControl::Continue
            })
            .expect("streaming completion");
        assert!(!outcome.stopped_by_callback);
        assert_eq!(streamed, completion.text);
    });
}

#[test]
fn seeded_sampling_is_repeatable() {
    with_engine(|engine, _| {
        let params = SamplingParams::default()
            .temperature(0.8)
            .seed(42)
            .max_tokens(8);
        let first = engine
            .complete("A surprising fact about rust is", &params)
            .expect("first seeded completion");
        let second = engine
            .complete("A surprising fact about rust is", &params)
            .expect("second seeded completion");
        assert_eq!(first, second);
    });
}

#[test]
fn early_stop_and_callback_panic_leave_engine_reusable() {
    with_engine(|engine, _| {
        let params = SamplingParams::greedy().max_tokens(8);
        let outcome = engine
            .complete_stream("Count from one to ten:", &params, |_| StreamControl::Stop)
            .expect("early stop");
        assert!(outcome.stopped_by_callback);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = engine.complete_stream("Say hello", &params, |_| {
                panic!("intentional callback panic")
            });
        }));
        assert!(panic.is_err());

        let completion = engine
            .complete("Say hello", &SamplingParams::greedy().max_tokens(2))
            .expect("engine remains reusable");
        assert!(!completion.text.is_empty());
    });
}

#[test]
fn concurrent_requests_batch_with_correct_output() {
    with_engine(|engine, _| {
        let params = SamplingParams::greedy().max_tokens(12);
        let expected_first = engine
            .complete("The capital of France is", &params)
            .expect("first reference completion")
            .text;
        let expected_second = engine
            .complete("The capital of Germany is", &params)
            .expect("second reference completion")
            .text;

        let first_text = Arc::new(Mutex::new(String::new()));
        let second_text = Arc::new(Mutex::new(String::new()));
        let mut first = engine
            .submit("The capital of France is", &params, {
                let text = Arc::clone(&first_text);
                move |event| {
                    text.lock()
                        .expect("first callback text")
                        .push_str(&event.delta);
                    StreamControl::Continue
                }
            })
            .expect("submit first request");
        let mut second = engine
            .submit("The capital of Germany is", &params, {
                let text = Arc::clone(&second_text);
                move |event| {
                    text.lock()
                        .expect("second callback text")
                        .push_str(&event.delta);
                    StreamControl::Continue
                }
            })
            .expect("submit second request");

        assert_eq!(first.wait().expect("wait first"), RequestOutcome::Completed);
        assert_eq!(
            second.wait().expect("wait second"),
            RequestOutcome::Completed
        );
        assert_eq!(*first_text.lock().expect("first result"), expected_first);
        assert_eq!(*second_text.lock().expect("second result"), expected_second);
        assert_eq!(first.native_error().expect("first native error"), None);
        assert_eq!(second.native_error().expect("second native error"), None);
    });
}

#[test]
fn engine_clones_submit_and_wait_from_multiple_rust_threads() {
    with_engine(|engine, _| {
        let start = Arc::new(Barrier::new(3));
        let workers = ["The capital of France is", "The capital of Germany is"]
            .into_iter()
            .map(|prompt| {
                let engine = engine.clone();
                let start = Arc::clone(&start);
                thread::spawn(move || {
                    let output = Arc::new(Mutex::new(String::new()));
                    start.wait();
                    let mut request = engine
                        .submit(prompt, &SamplingParams::greedy().max_tokens(8), {
                            let output = Arc::clone(&output);
                            move |event| {
                                output
                                    .lock()
                                    .expect("cross-thread callback output")
                                    .push_str(&event.delta);
                                StreamControl::Continue
                            }
                        })
                        .expect("submit from Rust worker thread");
                    let outcome = request.wait().expect("wait on Rust worker thread");
                    let native_error = request
                        .native_error()
                        .expect("native error on Rust worker thread");
                    let output = output.lock().expect("cross-thread output").clone();
                    (outcome, native_error, output)
                })
            })
            .collect::<Vec<_>>();

        start.wait();
        for worker in workers {
            let (outcome, native_error, output) = worker.join().expect("Rust request worker");
            assert_eq!(outcome, RequestOutcome::Completed);
            assert_eq!(native_error, None);
            assert!(!output.is_empty());
        }
    });
}

#[test]
fn live_request_moves_to_rust_thread_for_cancel_wait_and_drop() {
    with_engine(|engine, _| {
        let callback_release = Arc::new(Barrier::new(2));
        let (callback_started_sender, callback_started_receiver) = mpsc::channel();
        let (callback_drop_sender, callback_drop_receiver) = mpsc::channel();
        struct CallbackDropProbe(mpsc::Sender<thread::ThreadId>);
        impl Drop for CallbackDropProbe {
            fn drop(&mut self) {
                let _ = self.0.send(thread::current().id());
            }
        }

        let mut callback_started_sender = Some(callback_started_sender);
        let request = engine
            .submit(
                "Write a long numbered list:",
                &SamplingParams::greedy().max_tokens(64),
                {
                    let callback_release = Arc::clone(&callback_release);
                    let drop_probe = CallbackDropProbe(callback_drop_sender);
                    move |_| {
                        let _ = &drop_probe;
                        if let Some(sender) = callback_started_sender.take() {
                            sender.send(()).expect("report live callback");
                            callback_release.wait();
                        }
                        StreamControl::Continue
                    }
                },
            )
            .expect("submit request before moving it");

        let worker = thread::spawn(move || {
            let mut request = request;
            callback_started_receiver
                .recv_timeout(Duration::from_secs(180))
                .expect("callback starts while request is live");
            let cancel_result = request.cancel();
            callback_release.wait();
            cancel_result.expect("cancel moved request");
            let outcome = request.wait().expect("wait for moved request");
            let native_error = request.native_error().expect("moved request error");
            let worker_thread = thread::current().id();
            drop(request);
            (worker_thread, outcome, native_error)
        });

        let (worker_thread, outcome, native_error) = worker.join().expect("moved request worker");
        assert!(matches!(
            outcome,
            RequestOutcome::Cancelled | RequestOutcome::Completed
        ));
        assert_eq!(native_error, None);
        assert_eq!(
            callback_drop_receiver
                .recv_timeout(Duration::from_secs(30))
                .expect("callback state drops with moved request"),
            worker_thread
        );
    });
}

#[test]
fn request_outcomes_and_probes_are_precise() {
    with_engine(|engine, _| {
        let params = SamplingParams::greedy().max_tokens(64);
        let mut stopped = engine
            .submit("Count upward forever:", &params, |_| StreamControl::Stop)
            .expect("submit callback-stop request");
        assert_eq!(
            stopped.wait().expect("wait callback-stop request"),
            RequestOutcome::StoppedByCallback
        );
        assert!(stopped.is_done());
        assert!(stopped.is_done());
        assert_eq!(
            stopped.wait().expect("repeat callback-stop wait"),
            RequestOutcome::StoppedByCallback
        );

        let mut terminal_stopped = engine
            .submit(
                "Say hello",
                &SamplingParams::greedy().max_tokens(1),
                |event| {
                    if event.finished {
                        StreamControl::Stop
                    } else {
                        StreamControl::Continue
                    }
                },
            )
            .expect("submit terminal callback-stop request");
        assert_eq!(
            terminal_stopped
                .wait()
                .expect("wait terminal callback-stop request"),
            RequestOutcome::StoppedByCallback
        );

        let barrier = Arc::new(Barrier::new(2));
        let mut cancelled = engine
            .submit("Write a long numbered list:", &params, {
                let barrier = Arc::clone(&barrier);
                let mut first = true;
                move |_| {
                    if first {
                        first = false;
                        barrier.wait();
                        thread::sleep(Duration::from_millis(20));
                    }
                    StreamControl::Continue
                }
            })
            .expect("submit cancellable request");
        barrier.wait();
        cancelled.cancel().expect("first cancel");
        cancelled.cancel().expect("idempotent cancel");
        assert!(matches!(
            cancelled.wait().expect("wait cancelled request"),
            RequestOutcome::Cancelled | RequestOutcome::Completed
        ));
        assert!(cancelled.is_done());
        assert!(cancelled.is_done());
        assert_eq!(cancelled.native_error().expect("cancel native error"), None);
    });
}

#[test]
fn callback_panic_is_reported_and_engine_is_reusable() {
    with_engine(|engine, _| {
        let params = SamplingParams::greedy().max_tokens(16);
        let mut request = engine
            .submit("Say hello", &params, |_| {
                panic!("intentional async callback panic")
            })
            .expect("submit panic request");
        assert_eq!(request.wait().unwrap_err(), Error::CallbackPanicked);
        assert_eq!(request.wait().unwrap_err(), Error::CallbackPanicked);

        let completion = engine
            .complete("Say hello", &SamplingParams::greedy().max_tokens(2))
            .expect("engine remains reusable after async panic");
        assert!(!completion.text.is_empty());
    });
}

#[test]
fn request_retains_engine_and_live_drop_is_safe() {
    let Some(path) = model_path() else {
        eprintln!("skipping model test; set VLLM_CPP_TEST_MODEL using `just setup-test-model`");
        return;
    };
    let engine = Engine::builder(path)
        .num_blocks(64)
        .max_model_len(256)
        .max_num_seqs(2)
        .max_num_batched_tokens(256)
        .load()
        .expect("load drop-order engine");
    let params = SamplingParams::greedy().max_tokens(64);
    let final_clone = engine.clone();
    let live = engine
        .submit("Write a long numbered list:", &params, |_| {
            thread::sleep(Duration::from_millis(1));
            StreamControl::Continue
        })
        .expect("submit live request");
    drop(engine);
    drop(live);
    let completion = final_clone
        .complete("Say hello", &SamplingParams::greedy().max_tokens(2))
        .expect("final public engine clone remains usable");
    assert!(!completion.text.is_empty());

    let retained = final_clone
        .submit(
            "Count from one:",
            &SamplingParams::greedy().max_tokens(4),
            |_| StreamControl::Continue,
        )
        .expect("submit engine-retaining request");
    drop(final_clone);
    let mut retained = retained;
    assert_eq!(
        retained.wait().expect("request outlives public engines"),
        RequestOutcome::Completed
    );
}

#[test]
fn callback_thread_self_wait_is_rejected_and_self_drop_is_deferred() {
    with_engine(|engine, _| {
        let params = SamplingParams::greedy().max_tokens(16);
        let slot = Arc::new(Mutex::new(None::<Request>));
        let wait_error = Arc::new(Mutex::new(None));
        let callback_finished = Arc::new(Barrier::new(2));
        let callback_started = Arc::new(Barrier::new(2));
        let callback_thread = Arc::new(Mutex::new(None));
        let (drop_sender, drop_receiver) = mpsc::channel();
        struct CallbackDropProbe(mpsc::Sender<thread::ThreadId>);
        impl Drop for CallbackDropProbe {
            fn drop(&mut self) {
                let _ = self.0.send(thread::current().id());
            }
        }
        let request = engine
            .submit("Count from one:", &params, {
                let slot = Arc::clone(&slot);
                let wait_error = Arc::clone(&wait_error);
                let callback_started = Arc::clone(&callback_started);
                let callback_finished = Arc::clone(&callback_finished);
                let callback_thread = Arc::clone(&callback_thread);
                let drop_probe = CallbackDropProbe(drop_sender);
                move |_| {
                    let _ = &drop_probe;
                    *callback_thread.lock().expect("callback thread") =
                        Some(thread::current().id());
                    callback_started.wait();
                    if let Some(mut request) = slot.lock().expect("request slot").take() {
                        *wait_error.lock().expect("wait result") =
                            Some(request.wait().unwrap_err());
                        drop(request);
                        callback_finished.wait();
                    }
                    StreamControl::Stop
                }
            })
            .expect("submit self-lifecycle request");
        *slot.lock().expect("request slot") = Some(request);
        callback_started.wait();
        callback_finished.wait();
        assert_eq!(
            wait_error.lock().expect("wait result").take(),
            Some(Error::RequestCallbackThread { operation: "wait" })
        );
        let callback_thread = callback_thread
            .lock()
            .expect("callback thread")
            .expect("callback thread recorded");
        let cleanup_thread = drop_receiver
            .recv_timeout(Duration::from_secs(30))
            .expect("deferred callback state drop");
        assert_ne!(cleanup_thread, callback_thread);
    });
}

#[test]
fn concurrent_request_lifecycle_stress() {
    with_engine(|engine, _| {
        let params = SamplingParams::greedy().max_tokens(16);
        for round in 0..16 {
            let mut requests = Vec::new();
            for index in 0..4 {
                let request = engine
                    .submit(
                        &format!("Round {round}, item {index}:"),
                        &params,
                        move |event| {
                            if (round + index) % 4 != 0
                                && (round + index) % 5 == 0
                                && !event.finished
                            {
                                StreamControl::Stop
                            } else {
                                StreamControl::Continue
                            }
                        },
                    )
                    .expect("submit stress request");
                requests.push(request);
            }

            for (index, mut request) in requests.drain(..).enumerate() {
                match (round + index) % 4 {
                    0 => {
                        request.cancel().expect("stress cancel");
                        request.cancel().expect("stress repeated cancel");
                        assert!(matches!(
                            request.wait().expect("wait stress cancel"),
                            RequestOutcome::Cancelled | RequestOutcome::Completed
                        ));
                    }
                    1 => {
                        wait_until_done(&request);
                        assert!(request.is_done());
                        request.wait().expect("wait probed stress request");
                    }
                    2 => {
                        request.wait().expect("wait stress request");
                    }
                    _ => drop(request),
                }
            }
        }

        let completion = engine
            .complete("Say hello", &SamplingParams::greedy().max_tokens(2))
            .expect("engine remains reusable after lifecycle stress");
        assert!(!completion.text.is_empty());
    });
}

#[test]
fn custom_logits_processor_controls_tokens_and_receives_history() {
    with_engine(|engine, _| {
        let histories = Arc::new(Mutex::new(Vec::<Vec<i32>>::new()));
        let params = SamplingParams::greedy().max_tokens(3).logits_processor({
            let histories = Arc::clone(&histories);
            move |tokens, logits| {
                histories
                    .lock()
                    .expect("processor histories")
                    .push(tokens.to_vec());
                let forced = if tokens.is_empty() { 10 } else { 11 };
                logits.fill(f32::NEG_INFINITY);
                logits[forced] = f32::INFINITY;
            }
        });
        let completion = engine
            .complete("Say anything", &params)
            .expect("custom processor completion");
        assert_eq!(completion.completion_tokens, 3);
        assert_eq!(
            *histories.lock().expect("processor histories"),
            vec![vec![], vec![10], vec![10, 11]]
        );
    });
}

#[test]
fn logits_processor_panic_is_contained_for_blocking_and_async_requests() {
    with_engine(|engine, _| {
        let params = SamplingParams::greedy()
            .max_tokens(2)
            .logits_processor(|_, _| panic!("intentional logits processor panic"));
        let error = engine
            .complete("Say hello", &params)
            .expect_err("blocking processor panic");
        assert_eq!(error, Error::LogitsProcessorPanicked);

        let mut request = engine
            .submit("Say hello", &params, |_| StreamControl::Continue)
            .expect("submit processor panic request");
        assert_eq!(
            request.wait().expect_err("async processor panic"),
            Error::LogitsProcessorPanicked
        );

        let completion = engine
            .complete("Say hello", &SamplingParams::greedy().max_tokens(1))
            .expect("engine remains reusable");
        assert!(!completion.text.is_empty());
    });
}

#[test]
fn logits_processor_self_wait_is_rejected_and_state_is_released_after_cleanup() {
    with_engine(|engine, _| {
        let slot = Arc::new(Mutex::new(None::<Request>));
        let processor_ready = Arc::new(Barrier::new(2));
        let (result_sender, result_receiver) = mpsc::channel();
        let (drop_sender, drop_receiver) = mpsc::channel();

        struct ProcessorDropProbe(mpsc::Sender<thread::ThreadId>);
        impl Drop for ProcessorDropProbe {
            fn drop(&mut self) {
                let _ = self.0.send(thread::current().id());
            }
        }

        let params = SamplingParams::greedy().max_tokens(8).logits_processor({
            let slot = Arc::clone(&slot);
            let processor_ready = Arc::clone(&processor_ready);
            let result_sender = result_sender.clone();
            let drop_probe = ProcessorDropProbe(drop_sender);
            move |_, _| {
                let _ = &drop_probe;
                processor_ready.wait();
                if let Some(mut request) = slot.lock().expect("request slot").take() {
                    let thread_id = thread::current().id();
                    let error = request.wait().expect_err("self wait rejection");
                    result_sender
                        .send((thread_id, error))
                        .expect("processor result");
                    drop(request);
                }
            }
        });
        let request = engine
            .submit("Count from one:", &params, |_| StreamControl::Continue)
            .expect("submit processor self-lifecycle request");
        drop(params);
        *slot.lock().expect("request slot") = Some(request);
        processor_ready.wait();

        let (_processor_thread, error) = result_receiver
            .recv_timeout(Duration::from_secs(30))
            .expect("processor self-wait result");
        assert_eq!(error, Error::RequestCallbackThread { operation: "wait" });
        assert!(matches!(
            drop_receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
        drop(slot);
        drop_receiver
            .recv_timeout(Duration::from_secs(30))
            .expect("processor state released after request cleanup");
    });
}

#[test]
fn structured_choice_is_enforced() {
    with_engine(|engine, _| {
        let params =
            SamplingParams::greedy()
                .max_tokens(8)
                .structured_output(StructuredOutput::Choice(vec![
                    "red".to_owned(),
                    "blue".to_owned(),
                ]));
        let completion = engine
            .complete("Choose exactly one color: red or blue. Answer:", &params)
            .expect("structured completion");
        assert!(
            completion.text.trim() == "red" || completion.text.trim() == "blue",
            "unexpected choice: {:?}",
            completion.text
        );
    });
}

#[test]
fn structured_json_schema_is_enforced() {
    with_engine(|engine, _| {
        let schema = r#"{
            "type": "object",
            "properties": {
                "location": { "type": "string" },
                "temperature_celsius": { "type": "number" },
                "condition": { "type": "string" }
            },
            "required": ["location", "temperature_celsius", "condition"],
            "additionalProperties": false
        }"#;
        let params = SamplingParams::greedy()
            .max_tokens(64)
            .structured_output(StructuredOutput::JsonSchema(schema.to_owned()));
        let completion = engine
            .complete(
                "Extract the weather report as JSON: Paris is sunny and 22 degrees Celsius.",
                &params,
            )
            .expect("JSON Schema completion");
        let value: serde_json::Value =
            serde_json::from_str(completion.text.trim()).expect("valid structured JSON");
        let object = value.as_object().expect("JSON object");
        assert_eq!(object.len(), 3, "unexpected properties: {object:?}");
        assert!(object
            .get("location")
            .is_some_and(|value| value.is_string()));
        assert!(object
            .get("temperature_celsius")
            .is_some_and(|value| value.is_number()));
        assert!(object
            .get("condition")
            .is_some_and(|value| value.is_string()));
    });
}

#[test]
fn terminal_stop_is_natural_finish_for_completion_and_chat() {
    with_engine(|engine, _| {
        let mut completion_finished = false;
        let completion_outcome = engine
            .complete_stream(
                "Reply with hello.",
                &SamplingParams::greedy().max_tokens(4),
                |event| {
                    completion_finished |= event.finished;
                    if event.finished {
                        StreamControl::Stop
                    } else {
                        StreamControl::Continue
                    }
                },
            )
            .expect("terminal completion stop");
        assert!(completion_finished);
        assert!(!completion_outcome.stopped_by_callback);

        let request = r#"{
            "messages":[{"role":"user","content":"Reply with hello."}],
            "temperature":0,
            "max_tokens":4
        }"#;
        let mut chat_finished = false;
        let chat_outcome = engine
            .chat_stream_json(request, |event| {
                chat_finished |= event.finished;
                if event.finished {
                    StreamControl::Stop
                } else {
                    StreamControl::Continue
                }
            })
            .expect("terminal chat stop");
        assert!(chat_finished);
        assert!(!chat_outcome.stopped_by_callback);
    });
}

#[test]
fn blocking_and_streaming_chat_return_json() {
    with_engine(|engine, _| {
        let request = r#"{
            "messages":[{"role":"user","content":"Reply with hello."}],
            "temperature":0,
            "max_tokens":4
        }"#;
        let response = engine.chat_json(request).expect("blocking chat");
        let json: serde_json::Value = serde_json::from_str(&response).expect("valid response JSON");
        assert_eq!(json["object"], "chat.completion");
        assert!(json["choices"]
            .as_array()
            .is_some_and(|choices| !choices.is_empty()));

        let mut chunks = Vec::new();
        let outcome = engine
            .chat_stream_json(request, |event| {
                if !event.finished {
                    let chunk: serde_json::Value =
                        serde_json::from_str(&event.delta).expect("valid chunk JSON");
                    chunks.push(chunk);
                }
                StreamControl::Continue
            })
            .expect("streaming chat");
        assert!(!outcome.stopped_by_callback);
        assert!(!chunks.is_empty());
    });
}
