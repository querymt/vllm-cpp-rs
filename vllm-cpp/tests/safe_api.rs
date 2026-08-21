use static_assertions::{assert_impl_all, assert_not_impl_any};
use vllm_cpp::{
    Engine, Error, HuggingFaceError, HuggingFaceModel, Request, SchedulerPolicy, Toggle,
};

assert_impl_all!(Engine: Send, Sync, Clone);
assert_impl_all!(HuggingFaceError: Clone, std::fmt::Debug, Eq, PartialEq);
assert_impl_all!(HuggingFaceModel: Clone, std::fmt::Debug);
assert_impl_all!(Request: Send);
assert_impl_all!(vllm_cpp::SamplingParams: Clone, Send, Sync);
assert_not_impl_any!(Request: Sync);

fn missing_model() -> &'static str {
    "/nonexistent/vllm-cpp-rs-safe-api-model"
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
    assert_eq!(vllm_cpp::expected_abi_version(), 10);
    assert_eq!(vllm_cpp::abi_version(), 10);
    assert!(!vllm_cpp::version().expect("native version").is_empty());
}

#[test]
fn missing_model_is_typed() {
    let error = Engine::load(missing_model()).unwrap_err();
    assert!(matches!(error, Error::ModelLoad { .. }), "{error:?}");
    assert!(!error.to_string().is_empty());
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
        .load()
        .unwrap_err();
    assert!(matches!(error, Error::ModelLoad { .. }), "{error:?}");
}
