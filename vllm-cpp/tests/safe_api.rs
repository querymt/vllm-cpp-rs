use static_assertions::{assert_impl_all, assert_not_impl_any};
use vllm_cpp::{
    Device, EmbeddingEngine, EmbeddingResult, Engine, EngineBuilder, Error, HuggingFaceError,
    HuggingFaceModel, Request, SchedulerPolicy, Toggle, TokenCompletion, Transcription,
    TranscriptionEngine, TranscriptionInput,
};

assert_impl_all!(Device: Clone, Copy, std::fmt::Debug, Default, Eq, PartialEq, Send, Sync);
assert_impl_all!(Engine: Send, Sync, Clone);
assert_impl_all!(EngineBuilder: Clone, std::fmt::Debug, Send, Sync);
assert_impl_all!(HuggingFaceError: Clone, std::fmt::Debug, Eq, PartialEq);
assert_impl_all!(HuggingFaceModel: Clone, std::fmt::Debug);
assert_impl_all!(Request: Send);
assert_impl_all!(vllm_cpp::SamplingParams: Clone, Send, Sync);
assert_impl_all!(TokenCompletion: Clone, std::fmt::Debug, Eq, PartialEq, Send, Sync);
assert_impl_all!(Transcription: Clone, std::fmt::Debug, Eq, PartialEq, Send, Sync);
assert_impl_all!(EmbeddingResult: Clone, std::fmt::Debug, PartialEq, Send, Sync);
assert_impl_all!(TranscriptionInput<'static>: Clone, Copy, std::fmt::Debug, Send, Sync);
assert_not_impl_any!(Request: Sync);
assert_not_impl_any!(TranscriptionEngine: Send, Sync);
assert_not_impl_any!(EmbeddingEngine: Send, Sync);

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
fn missing_model_is_typed_for_every_task_owner() {
    let errors = [
        Engine::load(missing_model()).unwrap_err(),
        TranscriptionEngine::load(missing_model())
            .err()
            .expect("missing transcription model error"),
        EmbeddingEngine::load(missing_model())
            .err()
            .expect("missing embedding model error"),
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
    let error = Engine::builder("bad\0model").load().unwrap_err();
    assert_eq!(
        error,
        Error::InteriorNul {
            field: "model path"
        }
    );
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
        let error = Engine::builder(missing_model())
            .gpu_memory_utilization(value)
            .load()
            .unwrap_err();
        assert!(
            matches!(error, Error::InvalidConfiguration { .. }),
            "{value:?}: {error:?}"
        );
    }
}

#[test]
fn rejects_invalid_kv_cache_memory_bytes() {
    for value in [0, i64::MAX as u64 + 1, u64::MAX] {
        let error = Engine::builder(missing_model())
            .kv_cache_memory_bytes(value)
            .load()
            .unwrap_err();
        assert!(
            matches!(error, Error::InvalidConfiguration { .. }),
            "{value}: {error:?}"
        );
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
