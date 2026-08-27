use std::ffi::OsString;
use std::path::Path;

use static_assertions::{assert_impl_all, assert_not_impl_any};
use vllm_cpp::{
    compose_video_mux_argv, Device, EmbeddingEngine, EmbeddingEngineBuilder, EmbeddingResult,
    Engine, EngineBuilder, Error, HuggingFaceError, HuggingFaceModel, Request, SchedulerPolicy,
    Toggle, TokenCompletion, Transcription, TranscriptionEngine, TranscriptionEngineBuilder,
    TranscriptionInput, VideoDevice, VideoEngine, VideoEngineBuilder, VideoGenerationParams,
    VideoMuxArgv, VideoMuxParams, VideoPartition, VideoResult,
};

assert_impl_all!(Device: Clone, Copy, std::fmt::Debug, Default, Eq, PartialEq, Send, Sync);
assert_impl_all!(Engine: Send, Sync, Clone);
assert_impl_all!(EngineBuilder: Clone, std::fmt::Debug, Send, Sync);
assert_impl_all!(TranscriptionEngineBuilder: Clone, std::fmt::Debug, Send, Sync);
assert_impl_all!(EmbeddingEngineBuilder: Clone, std::fmt::Debug, Send, Sync);
assert_impl_all!(HuggingFaceError: Clone, std::fmt::Debug, Eq, PartialEq);
assert_impl_all!(HuggingFaceModel: Clone, std::fmt::Debug);
assert_impl_all!(Request: Send);
assert_impl_all!(vllm_cpp::SamplingParams: Clone, Send, Sync);
assert_impl_all!(TokenCompletion: Clone, std::fmt::Debug, Eq, PartialEq, Send, Sync);
assert_impl_all!(Transcription: Clone, std::fmt::Debug, Eq, PartialEq, Send, Sync);
assert_impl_all!(EmbeddingResult: Clone, std::fmt::Debug, PartialEq, Send, Sync);
assert_impl_all!(TranscriptionInput<'static>: Clone, Copy, std::fmt::Debug, Send, Sync);
assert_impl_all!(VideoDevice: Clone, Copy, std::fmt::Debug, Default, Eq, PartialEq, Send, Sync);
assert_impl_all!(VideoPartition: Clone, Copy, std::fmt::Debug, Eq, PartialEq, Send, Sync);
assert_impl_all!(VideoEngineBuilder: Clone, std::fmt::Debug, Send, Sync);
assert_impl_all!(VideoGenerationParams: Clone, std::fmt::Debug, Send, Sync);
assert_impl_all!(VideoMuxParams: Clone, std::fmt::Debug, Send, Sync);
assert_impl_all!(VideoResult: Clone, std::fmt::Debug, Eq, PartialEq, Send, Sync);
assert_impl_all!(VideoMuxArgv: Clone, std::fmt::Debug, Eq, PartialEq, Send, Sync);
assert_not_impl_any!(Request: Sync);
assert_not_impl_any!(TranscriptionEngine: Send, Sync, Clone);
assert_not_impl_any!(EmbeddingEngine: Send, Sync, Clone);
assert_not_impl_any!(VideoEngine: Send, Sync, Clone);

fn missing_model() -> &'static str {
    "/nonexistent/vllm-cpp-rs-safe-api-model"
}

#[test]
fn constructs_both_borrowed_transcription_inputs() {
    let path = std::path::Path::new("audio.wav");
    let samples = [0.0_f32, 0.25];
    let inputs = [
        TranscriptionInput::WavFile(path),
        TranscriptionInput::Pcm {
            samples: &samples,
            sample_rate: 16_000,
        },
    ];
    assert!(matches!(inputs[0], TranscriptionInput::WavFile(_)));
    assert!(matches!(inputs[1], TranscriptionInput::Pcm { .. }));
}

#[test]
fn invalid_native_output_display_is_stable() {
    let error = Error::InvalidNativeOutput {
        field: "embedding dimension",
        message: "dimension is zero",
    };
    assert_eq!(
        error.to_string(),
        "invalid native output for embedding dimension: dimension is zero"
    );
}

#[test]
fn hugging_face_constructors_accept_default_and_explicit_revisions() {
    let gguf = HuggingFaceModel::gguf("owner/model", "model.gguf");
    let safetensors = HuggingFaceModel::safetensors("owner/model").revision("release");
    assert!(format!("{gguf:?}").contains("revision: \"main\""));
    assert!(format!("{safetensors:?}").contains("revision: \"release\""));
}

#[test]
fn reports_expected_abi() {
    assert_eq!(vllm_cpp::expected_abi_version(), 17);
    assert_eq!(vllm_cpp::abi_version(), 17);
    assert!(vllm_cpp::version()
        .expect("native version")
        .starts_with("0.0.2"));
}

#[test]
fn missing_model_is_typed_for_every_task_owner_and_builder() {
    let errors = [
        Engine::load(missing_model()).unwrap_err(),
        TranscriptionEngine::load(missing_model())
            .err()
            .expect("missing transcription model error"),
        TranscriptionEngine::builder(missing_model())
            .device(Device::Cpu)
            .load()
            .err()
            .expect("configured missing transcription model error"),
        EmbeddingEngine::load(missing_model())
            .err()
            .expect("missing embedding model error"),
        EmbeddingEngine::builder(missing_model())
            .block_size(16)
            .num_blocks(32)
            .max_model_len(128)
            .max_num_seqs(2)
            .max_num_batched_tokens(64)
            .prefix_caching(Toggle::Off)
            .device(Device::Cpu)
            .gpu_memory_utilization(1.25)
            .kv_cache_memory_bytes(4096)
            .load()
            .err()
            .expect("configured missing embedding model error"),
    ];
    for error in errors {
        assert!(matches!(error, Error::ModelLoad { .. }), "{error:?}");
        assert!(!error.to_string().is_empty());
    }
}

#[test]
fn malformed_engine_json_is_invalid_argument_before_loading() {
    let error = Engine::builder(missing_model())
        .speculative_config("{")
        .load()
        .unwrap_err();
    assert!(matches!(error, Error::InvalidArgument { .. }), "{error:?}");
}

#[test]
fn interior_nul_fails_before_ffi() {
    let errors = [
        Engine::builder("bad\0model").load().unwrap_err(),
        TranscriptionEngineBuilder::new("bad\0model")
            .load()
            .err()
            .expect("transcription interior NUL"),
        EmbeddingEngineBuilder::new("bad\0model")
            .load()
            .err()
            .expect("embedding interior NUL"),
    ];
    for error in errors {
        assert_eq!(
            error,
            Error::InteriorNul {
                field: "model path"
            }
        );
    }
}

#[test]
fn device_defaults_to_native_auto_selection() {
    assert_eq!(Device::default(), Device::Auto);
}

#[test]
fn engine_builder_accepts_all_safe_options() {
    let error = Engine::builder(missing_model())
        .tokenizer_config_path("/nonexistent/tokenizer_config.json")
        .block_size(16)
        .num_blocks(32)
        .max_model_len(128)
        .max_num_seqs(2)
        .tool_parser("hermes")
        .reasoning_parser("none")
        .prefix_caching(Toggle::Off)
        .max_num_batched_tokens(128)
        .scheduler(SchedulerPolicy::LongestPrefixMatch)
        .kv_transfer_config("")
        .jump_forward(Toggle::Off)
        .device(Device::Cpu)
        .gpu_memory_utilization(1.25)
        .kv_cache_memory_bytes(4096)
        .load()
        .unwrap_err();
    assert!(matches!(error, Error::ModelLoad { .. }), "{error:?}");
}

#[test]
fn rejects_invalid_gpu_memory_utilization() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0, -1.0] {
        for error in [
            Engine::builder(missing_model())
                .gpu_memory_utilization(value)
                .load()
                .unwrap_err(),
            EmbeddingEngine::builder(missing_model())
                .gpu_memory_utilization(value)
                .load()
                .err()
                .expect("invalid embedding utilization"),
        ] {
            assert!(
                matches!(error, Error::InvalidConfiguration { .. }),
                "{value:?}: {error:?}"
            );
        }
    }
}

#[test]
fn rejects_invalid_kv_cache_memory_bytes() {
    for value in [0, i64::MAX as u64 + 1, u64::MAX] {
        for error in [
            Engine::builder(missing_model())
                .kv_cache_memory_bytes(value)
                .load()
                .unwrap_err(),
            EmbeddingEngine::builder(missing_model())
                .kv_cache_memory_bytes(value)
                .load()
                .err()
                .expect("invalid embedding KV bytes"),
        ] {
            assert!(
                matches!(error, Error::InvalidConfiguration { .. }),
                "{value}: {error:?}"
            );
        }
    }
}

#[test]
fn valid_memory_settings_and_native_precedence_reach_model_loading() {
    let errors = [
        Engine::builder(missing_model())
            .gpu_memory_utilization(f64::MIN_POSITIVE)
            .load()
            .unwrap_err(),
        Engine::builder(missing_model())
            .gpu_memory_utilization(2.0)
            .load()
            .unwrap_err(),
        Engine::builder(missing_model())
            .kv_cache_memory_bytes(i64::MAX as u64)
            .load()
            .unwrap_err(),
        Engine::builder(missing_model())
            .num_blocks(1)
            .kv_cache_memory_bytes(4096)
            .gpu_memory_utilization(1.5)
            .load()
            .unwrap_err(),
    ];
    for error in errors {
        assert!(matches!(error, Error::ModelLoad { .. }), "{error:?}");
    }
}

#[test]
fn video_device_defaults_to_cpu_and_builder_validates_paths() {
    assert_eq!(VideoDevice::default(), VideoDevice::Cpu);

    for builder in [
        VideoEngineBuilder::new(""),
        VideoEngine::builder("dit.gguf"),
        VideoEngine::builder("dit.gguf").video_vae_path("video.safetensors"),
        VideoEngine::builder("dit.gguf")
            .video_vae_path("")
            .audio_vae_path("audio.safetensors"),
        VideoEngine::builder("dit.gguf")
            .video_vae_path("video.safetensors")
            .audio_vae_path(""),
        VideoEngine::builder("dit.gguf")
            .video_vae_path("video.safetensors")
            .audio_vae_path("audio.safetensors")
            .encoder_path(""),
    ] {
        let error = builder.load().err().expect("invalid video model paths");
        assert!(
            matches!(error, Error::InvalidConfiguration { .. }),
            "{error:?}"
        );
    }

    let error = VideoEngine::builder("dit\0.gguf")
        .video_vae_path("video.safetensors")
        .audio_vae_path("audio.safetensors")
        .load()
        .err()
        .expect("interior NUL video path");
    assert_eq!(
        error,
        Error::InteriorNul {
            field: "video DiT path"
        }
    );
}

#[test]
fn video_builder_accepts_all_safe_options_without_inference() {
    let error = VideoEngine::builder(missing_model())
        .encoder_path("/nonexistent/encoder.gguf")
        .tokenizer_path("/nonexistent/tokenizer.json")
        .video_vae_path("/nonexistent/video.safetensors")
        .video_vae_config_path("/nonexistent/video.json")
        .audio_vae_path("/nonexistent/audio.safetensors")
        .audio_vae_config_path("/nonexistent/audio.json")
        .prompt_embeds_path("/nonexistent/prompt.f32")
        .partition(VideoPartition::Fl2va)
        .device(VideoDevice::Cuda)
        .dequant_bf16(true)
        .fp4_resident(true)
        .load()
        .err()
        .expect("missing video model");
    assert!(matches!(error, Error::ModelLoad { .. }), "{error:?}");
}

#[test]
fn video_generation_validation_covers_numeric_reference_and_path_rules() {
    VideoGenerationParams::new("", "out")
        .dimensions(1, 33)
        .num_frames(2)
        .steps(1)
        .seed(0)
        .noise_augmentation(f32::MIN_POSITIVE)
        .validate()
        .expect("valid explicit generation parameters");

    let invalid = [
        VideoGenerationParams::new("", ""),
        VideoGenerationParams::new("bad\0prompt", "out"),
        VideoGenerationParams::new("", "out").dimensions(0, 1),
        VideoGenerationParams::new("", "out").dimensions(i32::MAX as u32 + 1, 1),
        VideoGenerationParams::new("", "out").num_frames(1),
        VideoGenerationParams::new("", "out").num_frames(i32::MAX as u32 + 1),
        VideoGenerationParams::new("", "out").steps(0),
        VideoGenerationParams::new("", "out").steps(i32::MAX as u32 + 1),
        VideoGenerationParams::new("", "out").noise_augmentation(f32::NAN),
        VideoGenerationParams::new("", "out").noise_augmentation(f32::INFINITY),
        VideoGenerationParams::new("", "out").noise_augmentation(0.0),
        VideoGenerationParams::new("", "out")
            .first_frame("first.ppm")
            .reference_image("reference.ppm"),
        VideoGenerationParams::new("", "out")
            .last_frame("last.ppm")
            .reference_audio("reference.wav"),
        VideoGenerationParams::new("", "out")
            .reference_image("reference.ppm")
            .reference_video("reference-frames"),
        VideoGenerationParams::new("", "x".repeat(482)),
        VideoGenerationParams::new("", "out").reference_video("x".repeat(482)),
        VideoGenerationParams::new("", "out").first_frame(""),
    ];
    for params in invalid {
        assert!(params.validate().is_err(), "unexpectedly valid: {params:?}");
    }

    for params in [
        VideoGenerationParams::new("", "out").first_frame("first.ppm"),
        VideoGenerationParams::new("", "out").last_frame("last.ppm"),
        VideoGenerationParams::new("", "out")
            .first_frame("first.ppm")
            .last_frame("last.ppm"),
        VideoGenerationParams::new("", "out")
            .reference_image("reference.ppm")
            .reference_audio("reference.wav"),
        VideoGenerationParams::new("", "out")
            .reference_video("reference-frames")
            .reference_audio("reference.wav"),
        VideoGenerationParams::new("", "out").reference_audio("reference.wav"),
    ] {
        params.validate().expect("allowed reference combination");
    }
}

fn os_args(values: &[&str]) -> Vec<OsString> {
    values.iter().map(OsString::from).collect()
}

#[test]
fn video_mux_validation_defaults_order_and_no_io_are_exact() {
    for params in [
        VideoMuxParams::new("", "out.mp4"),
        VideoMuxParams::new("frames_%06d.ppm", ""),
        VideoMuxParams::new("frames_%06d.ppm", "out.mp4").audio_path(""),
        VideoMuxParams::new("frames_%06d.ppm", "out.mp4").fps(0),
        VideoMuxParams::new("frames_%06d.ppm", "out.mp4").fps(i32::MAX as u32 + 1),
        VideoMuxParams::new("frames_%06d.ppm", "out.mp4").crf(0),
        VideoMuxParams::new("frames_%06d.ppm", "out.mp4").crf(i32::MAX as u32 + 1),
    ] {
        let error = compose_video_mux_argv(&params).expect_err("invalid mux parameters");
        assert!(
            matches!(error, Error::InvalidConfiguration { .. }),
            "{error:?}"
        );
    }

    let unique = format!(
        "vllm-cpp-rs-video-mux-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos()
    );
    let root = std::env::temp_dir().join(unique);
    let frames = root.join("frames $HOME;*.ppm");
    let audio = root.join("audio & track.wav");
    let output = root.join("video $(name).mp4");
    assert!(!root.exists());

    let silent = compose_video_mux_argv(&VideoMuxParams::new(&frames, &output))
        .expect("silent mux composition");
    let expected_silent = [
        OsString::from("ffmpeg"),
        OsString::from("-y"),
        OsString::from("-loglevel"),
        OsString::from("error"),
        OsString::from("-framerate"),
        OsString::from("24"),
        OsString::from("-i"),
        frames.as_os_str().to_owned(),
        OsString::from("-c:v"),
        OsString::from("libx264"),
        OsString::from("-pix_fmt"),
        OsString::from("yuv420p"),
        OsString::from("-crf"),
        OsString::from("18"),
        OsString::from("-movflags"),
        OsString::from("+faststart"),
        output.as_os_str().to_owned(),
    ];
    assert_eq!(silent.args(), expected_silent);

    let audio_mux = compose_video_mux_argv(
        &VideoMuxParams::new(&frames, &output)
            .audio_path(&audio)
            .fps(30)
            .crf(51),
    )
    .expect("audio mux composition");
    let expected_audio = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        "30",
        "-i",
    ];
    assert_eq!(&audio_mux.args()[..7], os_args(&expected_audio));
    assert_eq!(audio_mux.args()[7], frames.as_os_str());
    assert_eq!(audio_mux.args()[8], OsString::from("-i"));
    assert_eq!(audio_mux.args()[9], audio.as_os_str());
    assert_eq!(
        &audio_mux.args()[10..],
        os_args(&[
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "51",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            output.to_str().expect("UTF-8 temp output"),
        ])
    );
    assert!(
        !root.exists(),
        "mux composition must not create directories or files"
    );
    assert!(!Path::new(&output).exists());
    assert_eq!(silent.clone().into_args(), silent.args());
}
