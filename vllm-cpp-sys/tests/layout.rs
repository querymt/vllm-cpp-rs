use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::mem::{align_of, offset_of, size_of};
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use vllm_cpp_sys::{
    vllm_completion, vllm_embedding_result, vllm_engine, vllm_logits_processor, vllm_model_params,
    vllm_request, vllm_sampling_params, vllm_status, vllm_status_VLLM_ERR_INVALID_ARGUMENT,
    vllm_status_VLLM_ERR_MODEL_LOAD, vllm_status_VLLM_ERR_RUNTIME, vllm_status_VLLM_ERR_UNKNOWN,
    vllm_status_VLLM_OK, vllm_token_callback, vllm_transcription, vllm_transcription_params,
    vllm_video_engine, vllm_video_model_params, vllm_video_mux_params, vllm_video_params,
    vllm_video_result,
};

#[test]
fn generated_bindings_match_c_layout() {
    let probe = LayoutProbe::compile();
    let output = Command::new(&probe.executable)
        .output()
        .expect("failed to execute the C layout probe");
    assert!(
        output.status.success(),
        "C layout probe failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let actual = parse_layouts(&String::from_utf8(output.stdout).expect("probe emits UTF-8"));

    macro_rules! expect_layout {
        ($type:ty) => {
            assert_eq!(
                actual[stringify!($type)],
                (size_of::<$type>(), align_of::<$type>()),
                "layout mismatch for {}",
                stringify!($type)
            );
        };
    }
    macro_rules! expect_offset {
        ($type:ty, $field:ident) => {
            assert_eq!(
                actual[concat!(stringify!($type), ".", stringify!($field))].0,
                offset_of!($type, $field),
                "offset mismatch for {}.{}",
                stringify!($type),
                stringify!($field)
            );
        };
    }

    macro_rules! expect_named_layout {
        ($name:literal, $type:ty) => {
            assert_eq!(
                actual[$name],
                (size_of::<$type>(), align_of::<$type>()),
                "layout mismatch for {}",
                $name
            );
        };
    }
    macro_rules! expect_value {
        ($name:literal, $value:expr) => {
            assert_eq!(
                actual[$name].0, $value as usize,
                "value mismatch for {}",
                $name
            );
        };
    }

    expect_layout!(vllm_status);
    expect_value!("VLLM_OK", vllm_status_VLLM_OK);
    expect_value!(
        "VLLM_ERR_INVALID_ARGUMENT",
        vllm_status_VLLM_ERR_INVALID_ARGUMENT
    );
    expect_value!("VLLM_ERR_MODEL_LOAD", vllm_status_VLLM_ERR_MODEL_LOAD);
    expect_value!("VLLM_ERR_RUNTIME", vllm_status_VLLM_ERR_RUNTIME);
    expect_value!("VLLM_ERR_UNKNOWN", vllm_status_VLLM_ERR_UNKNOWN);

    expect_named_layout!("vllm_engine_ptr", *mut vllm_engine);
    expect_named_layout!("vllm_request_ptr", *mut vllm_request);
    expect_named_layout!("vllm_video_engine_ptr", *mut vllm_video_engine);

    expect_layout!(vllm_model_params);
    expect_offset!(vllm_model_params, model_path);
    expect_offset!(vllm_model_params, tokenizer_config_path);
    expect_offset!(vllm_model_params, block_size);
    expect_offset!(vllm_model_params, num_blocks);
    expect_offset!(vllm_model_params, max_model_len);
    expect_offset!(vllm_model_params, max_num_seqs);
    expect_offset!(vllm_model_params, tool_parser);
    expect_offset!(vllm_model_params, reasoning_parser);
    expect_offset!(vllm_model_params, speculative_config);
    expect_offset!(vllm_model_params, enable_prefix_caching);
    expect_offset!(vllm_model_params, max_num_batched_tokens);
    expect_offset!(vllm_model_params, scheduling_policy);
    expect_offset!(vllm_model_params, kv_transfer_config);
    expect_offset!(vllm_model_params, enable_jump_forward);
    expect_offset!(vllm_model_params, device);
    expect_offset!(vllm_model_params, gpu_memory_utilization);
    expect_offset!(vllm_model_params, kv_cache_memory_bytes);

    expect_layout!(vllm_sampling_params);
    expect_offset!(vllm_sampling_params, temperature);
    expect_offset!(vllm_sampling_params, top_p);
    expect_offset!(vllm_sampling_params, top_k);
    expect_offset!(vllm_sampling_params, min_p);
    expect_offset!(vllm_sampling_params, max_tokens);
    expect_offset!(vllm_sampling_params, seed);
    expect_offset!(vllm_sampling_params, has_seed);
    expect_offset!(vllm_sampling_params, presence_penalty);
    expect_offset!(vllm_sampling_params, frequency_penalty);
    expect_offset!(vllm_sampling_params, repetition_penalty);
    expect_offset!(vllm_sampling_params, min_tokens);
    expect_offset!(vllm_sampling_params, ignore_eos);
    expect_offset!(vllm_sampling_params, stop);
    expect_offset!(vllm_sampling_params, n_stop);
    expect_offset!(vllm_sampling_params, structured_json);
    expect_offset!(vllm_sampling_params, structured_regex);
    expect_offset!(vllm_sampling_params, structured_choice);
    expect_offset!(vllm_sampling_params, n_structured_choice);
    expect_offset!(vllm_sampling_params, structured_grammar);
    expect_offset!(vllm_sampling_params, structured_json_object);
    expect_offset!(vllm_sampling_params, logits_processor);
    expect_offset!(vllm_sampling_params, logits_processor_user_data);

    expect_layout!(vllm_completion);
    expect_offset!(vllm_completion, text);
    expect_offset!(vllm_completion, finish_reason);
    expect_offset!(vllm_completion, prompt_tokens);
    expect_offset!(vllm_completion, completion_tokens);

    expect_layout!(vllm_transcription_params);
    expect_offset!(vllm_transcription_params, audio_path);
    expect_offset!(vllm_transcription_params, pcm);
    expect_offset!(vllm_transcription_params, n_samples);
    expect_offset!(vllm_transcription_params, sample_rate);

    expect_layout!(vllm_transcription);
    expect_offset!(vllm_transcription, text);
    expect_offset!(vllm_transcription, token_ids);
    expect_offset!(vllm_transcription, n_token_ids);
    expect_offset!(vllm_transcription, has_text);

    expect_layout!(vllm_embedding_result);
    expect_offset!(vllm_embedding_result, values);
    expect_offset!(vllm_embedding_result, n_embeddings);
    expect_offset!(vllm_embedding_result, dim);
    expect_offset!(vllm_embedding_result, prompt_tokens);

    expect_layout!(vllm_video_model_params);
    expect_offset!(vllm_video_model_params, dit_path);
    expect_offset!(vllm_video_model_params, encoder_path);
    expect_offset!(vllm_video_model_params, tokenizer_path);
    expect_offset!(vllm_video_model_params, video_vae_path);
    expect_offset!(vllm_video_model_params, video_vae_config_path);
    expect_offset!(vllm_video_model_params, audio_vae_path);
    expect_offset!(vllm_video_model_params, audio_vae_config_path);
    expect_offset!(vllm_video_model_params, prompt_embeds_path);
    expect_offset!(vllm_video_model_params, partition);
    expect_offset!(vllm_video_model_params, device);
    expect_offset!(vllm_video_model_params, dequant_bf16);
    expect_offset!(vllm_video_model_params, fp4_resident);

    expect_layout!(vllm_video_params);
    expect_offset!(vllm_video_params, prompt);
    expect_offset!(vllm_video_params, width);
    expect_offset!(vllm_video_params, height);
    expect_offset!(vllm_video_params, num_frames);
    expect_offset!(vllm_video_params, steps);
    expect_offset!(vllm_video_params, seed);
    expect_offset!(vllm_video_params, has_seed);
    expect_offset!(vllm_video_params, first_frame);
    expect_offset!(vllm_video_params, last_frame);
    expect_offset!(vllm_video_params, ref_image);
    expect_offset!(vllm_video_params, ref_video);
    expect_offset!(vllm_video_params, ref_audio);
    expect_offset!(vllm_video_params, noise_aug);
    expect_offset!(vllm_video_params, output_dir);

    expect_layout!(vllm_video_result);
    expect_offset!(vllm_video_result, frame_dir);
    expect_offset!(vllm_video_result, audio_path);
    expect_offset!(vllm_video_result, frame_count);
    expect_offset!(vllm_video_result, width);
    expect_offset!(vllm_video_result, height);
    expect_offset!(vllm_video_result, fps);
    expect_offset!(vllm_video_result, sample_rate);
    expect_offset!(vllm_video_result, mux_argv);
    expect_offset!(vllm_video_result, mux_argc);

    expect_layout!(vllm_video_mux_params);
    expect_offset!(vllm_video_mux_params, frames);
    expect_offset!(vllm_video_mux_params, audio_path);
    expect_offset!(vllm_video_mux_params, output_path);
    expect_offset!(vllm_video_mux_params, fps);
    expect_offset!(vllm_video_mux_params, crf);

    expect_layout!(vllm_token_callback);
    expect_layout!(vllm_logits_processor);
}

struct LayoutProbe {
    temp_dir: PathBuf,
    executable: PathBuf,
}

impl LayoutProbe {
    fn compile() -> Self {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let include_dir = if cfg!(feature = "system") {
            PathBuf::from(
                env::var_os("VLLM_CPP_ROOT")
                    .expect("VLLM_CPP_ROOT is required by the system feature"),
            )
            .join("include")
        } else {
            manifest_dir.join("vllm.cpp/include")
        };
        let temp_dir = unique_temp_dir();
        fs::create_dir(&temp_dir).expect("failed to create a layout probe directory");
        let probe = Self {
            executable: temp_dir.join("vllm-layout"),
            temp_dir,
        };
        let compiler = env::var_os("CC").unwrap_or_else(|| "cc".into());
        let output = Command::new(compiler)
            .arg("-std=c11")
            .arg("-Wall")
            .arg("-Wextra")
            .arg("-I")
            .arg(&include_dir)
            .arg(manifest_dir.join("tests/layout.c"))
            .arg("-o")
            .arg(&probe.executable)
            .output()
            .expect("failed to execute the C compiler for the layout probe");
        assert!(
            output.status.success(),
            "failed to compile the C layout probe:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        probe
    }
}

impl Drop for LayoutProbe {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.temp_dir);
    }
}

fn unique_temp_dir() -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before the Unix epoch")
        .as_nanos();
    env::temp_dir().join(format!(
        "vllm-cpp-sys-layout-{}-{timestamp}",
        std::process::id()
    ))
}

fn parse_layouts(output: &str) -> BTreeMap<String, (usize, usize)> {
    output
        .lines()
        .map(|line| {
            let mut fields = line.split_whitespace();
            let name = fields.next().expect("probe record has a name").to_owned();
            let first = fields
                .next()
                .expect("probe record has a value")
                .parse()
                .expect("probe value is an integer");
            let second = fields.next().map_or(0, |value| {
                value.parse().expect("probe alignment is an integer")
            });
            assert!(fields.next().is_none(), "unexpected probe record: {line}");
            (name, (first, second))
        })
        .collect()
}
