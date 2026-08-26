use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::ptr::{self, NonNull};
use std::rc::Rc;
use std::sync::Arc;

use vllm_cpp_sys as ffi;

use crate::abi::Compatibility;
use crate::callback::{
    callback_trampoline, CallbackState, StreamControl, StreamEvent, StreamOutcome,
};
use crate::error::{invalid_configuration, status_result, Error};
use crate::params::{Device, SamplingParams, SchedulerPolicy, Toggle};

/// A cloneable, `Send + Sync` vllm.cpp text-generation engine.
#[derive(Clone)]
pub struct Engine {
    pub(crate) inner: Arc<EngineInner>,
}

pub(crate) struct TextTask;
struct TranscriptionTask;
struct EmbeddingTask;

pub(crate) struct OwnedEngine<Task> {
    pub(crate) raw: NonNull<ffi::vllm_engine>,
    pub(crate) compatibility: Compatibility,
    _task: PhantomData<Task>,
    _not_send_sync: PhantomData<Rc<()>>,
}

struct LoadedEngine<Task> {
    raw: NonNull<ffi::vllm_engine>,
    compatibility: Compatibility,
    _task: PhantomData<Task>,
}

pub(crate) type EngineInner = OwnedEngine<TextTask>;

/// A thread-local RAII owner for a native transcription-task engine.
///
/// ABI 17 has no task query, so loading cannot prove that a checkpoint supports
/// transcription. Native task selection and future wrong-task diagnostics remain
/// authoritative. This owner intentionally exposes no operation yet and is
/// neither `Send` nor `Sync`.
pub struct TranscriptionEngine {
    _inner: OwnedEngine<TranscriptionTask>,
}

/// A thread-local RAII owner for a native embedding-task engine.
///
/// ABI 17 has no task query, so loading cannot prove that a checkpoint supports
/// embeddings. Native task selection and future wrong-task diagnostics remain
/// authoritative. This owner intentionally exposes no operation yet and is
/// neither `Send` nor `Sync`.
pub struct EmbeddingEngine {
    _inner: OwnedEngine<EmbeddingTask>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Engine")
            .field("raw", &self.inner.raw)
            .finish_non_exhaustive()
    }
}

/// Builder for one complete text-generation engine.
#[derive(Clone, Debug)]
pub struct EngineBuilder {
    config: ModelConfig,
}

#[derive(Clone, Debug)]
struct ModelConfig {
    model_path: PathBuf,
    tokenizer_config_path: Option<PathBuf>,
    block_size: Option<u32>,
    num_blocks: Option<u32>,
    max_model_len: Option<u32>,
    max_num_seqs: Option<u32>,
    tool_parser: Option<String>,
    reasoning_parser: Option<String>,
    speculative_config: Option<String>,
    prefix_caching: Toggle,
    max_num_batched_tokens: Option<u32>,
    scheduler: Option<SchedulerPolicy>,
    kv_transfer_config: Option<String>,
    jump_forward: Toggle,
    device: Option<Device>,
    gpu_memory_utilization: Option<f64>,
    kv_cache_memory_bytes: Option<u64>,
}

struct MarshaledModelParams {
    raw: Option<ffi::vllm_model_params>,
    model_path: CString,
    tokenizer_config_path: Option<CString>,
    block_size: Option<i32>,
    num_blocks: Option<i32>,
    max_model_len: Option<i32>,
    max_num_seqs: Option<i32>,
    tool_parser: Option<CString>,
    reasoning_parser: Option<CString>,
    speculative_config: Option<CString>,
    prefix_caching: Toggle,
    max_num_batched_tokens: Option<i32>,
    scheduling_policy: Option<CString>,
    kv_transfer_config: Option<CString>,
    jump_forward: Toggle,
    device: Option<Device>,
    gpu_memory_utilization: Option<f64>,
    kv_cache_memory_bytes: Option<i64>,
}

impl ModelConfig {
    fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            tokenizer_config_path: None,
            block_size: None,
            num_blocks: None,
            max_model_len: None,
            max_num_seqs: None,
            tool_parser: None,
            reasoning_parser: None,
            speculative_config: None,
            prefix_caching: Toggle::Default,
            max_num_batched_tokens: None,
            scheduler: None,
            kv_transfer_config: None,
            jump_forward: Toggle::Default,
            device: None,
            gpu_memory_utilization: None,
            kv_cache_memory_bytes: None,
        }
    }
}

impl MarshaledModelParams {
    fn new(config: ModelConfig) -> Result<Self, Error> {
        let gpu_memory_utilization = match config.gpu_memory_utilization {
            Some(value) if !value.is_finite() || value <= 0.0 => {
                return Err(invalid_configuration(
                    "gpu_memory_utilization must be finite and strictly positive",
                ));
            }
            value => value,
        };
        let kv_cache_memory_bytes = match config.kv_cache_memory_bytes {
            Some(0) => {
                return Err(invalid_configuration(
                    "kv_cache_memory_bytes must be greater than zero",
                ));
            }
            Some(value) => Some(i64::try_from(value).map_err(|_| {
                invalid_configuration("kv_cache_memory_bytes exceeds native i64 range")
            })?),
            None => None,
        };

        Ok(Self {
            raw: None,
            model_path: path_to_cstring(&config.model_path, "model path")?,
            tokenizer_config_path: config
                .tokenizer_config_path
                .as_deref()
                .map(|path| path_to_cstring(path, "tokenizer config path"))
                .transpose()?,
            block_size: optional_u32_to_i32(config.block_size, "block_size")?,
            num_blocks: optional_u32_to_i32(config.num_blocks, "num_blocks")?,
            max_model_len: optional_u32_to_i32(config.max_model_len, "max_model_len")?,
            max_num_seqs: optional_u32_to_i32(config.max_num_seqs, "max_num_seqs")?,
            tool_parser: optional_cstring(config.tool_parser.as_deref(), "tool parser")?,
            reasoning_parser: optional_cstring(
                config.reasoning_parser.as_deref(),
                "reasoning parser",
            )?,
            speculative_config: optional_cstring(
                config.speculative_config.as_deref(),
                "speculative configuration",
            )?,
            prefix_caching: config.prefix_caching,
            max_num_batched_tokens: optional_u32_to_i32(
                config.max_num_batched_tokens,
                "max_num_batched_tokens",
            )?,
            scheduling_policy: config
                .scheduler
                .map(|value| to_cstring(value.as_str(), "scheduler policy"))
                .transpose()?,
            kv_transfer_config: optional_cstring(
                config.kv_transfer_config.as_deref(),
                "KV transfer configuration",
            )?,
            jump_forward: config.jump_forward,
            device: config.device,
            gpu_memory_utilization,
            kv_cache_memory_bytes,
        })
    }

    fn apply_defaults(&mut self, mut raw: ffi::vllm_model_params) {
        raw.model_path = self.model_path.as_ptr();
        if let Some(value) = &self.tokenizer_config_path {
            raw.tokenizer_config_path = value.as_ptr();
        }
        if let Some(value) = self.block_size {
            raw.block_size = value;
        }
        if let Some(value) = self.num_blocks {
            raw.num_blocks = value;
        }
        if let Some(value) = self.max_model_len {
            raw.max_model_len = value;
        }
        if let Some(value) = self.max_num_seqs {
            raw.max_num_seqs = value;
        }
        if let Some(value) = &self.tool_parser {
            raw.tool_parser = value.as_ptr();
        }
        if let Some(value) = &self.reasoning_parser {
            raw.reasoning_parser = value.as_ptr();
        }
        if let Some(value) = &self.speculative_config {
            raw.speculative_config = value.as_ptr();
        }
        if self.prefix_caching != Toggle::Default {
            raw.enable_prefix_caching = self.prefix_caching.as_native();
        }
        if let Some(value) = self.max_num_batched_tokens {
            raw.max_num_batched_tokens = value;
        }
        if let Some(value) = &self.scheduling_policy {
            raw.scheduling_policy = value.as_ptr();
        }
        if let Some(value) = &self.kv_transfer_config {
            raw.kv_transfer_config = value.as_ptr();
        }
        if self.jump_forward != Toggle::Default {
            raw.enable_jump_forward = self.jump_forward.as_native();
        }
        if let Some(value) = self.device {
            raw.device = value.as_native();
        }
        if let Some(value) = self.gpu_memory_utilization {
            raw.gpu_memory_utilization = value;
        }
        if let Some(value) = self.kv_cache_memory_bytes {
            raw.kv_cache_memory_bytes = value;
        }
        self.raw = Some(raw);
    }

    fn raw(&self) -> &ffi::vllm_model_params {
        self.raw
            .as_ref()
            .expect("native defaults must be applied before model loading")
    }
}

/// Why native generation finished.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum FinishReason {
    Stop,
    Length,
    Abort,
    Error,
    Repetition,
    Unknown,
    Other(String),
}

/// A Rust-owned blocking completion result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Completion {
    pub text: String,
    pub finish_reason: Option<FinishReason>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

impl Engine {
    /// Starts configuring an engine for a model directory or GGUF file.
    pub fn builder(model_path: impl Into<PathBuf>) -> EngineBuilder {
        EngineBuilder::new(model_path)
    }

    /// Loads an engine using native defaults.
    pub fn load(model_path: impl Into<PathBuf>) -> Result<Self, Error> {
        Self::builder(model_path).load()
    }

    /// Runs one blocking text completion.
    pub fn complete(&self, prompt: &str, params: &SamplingParams) -> Result<Completion, Error> {
        let prompt = to_cstring(prompt, "prompt")?;
        let params = params.marshal(&self.inner.compatibility)?;
        let mut raw = MaybeUninit::<ffi::vllm_completion>::uninit();
        // SAFETY: the engine is owned and live, all pointers remain valid for the
        // call, and out storage is initialized by native code on success.
        let status = unsafe {
            ffi::vllm_complete(
                self.inner.raw.as_ptr(),
                prompt.as_ptr(),
                params.raw(),
                raw.as_mut_ptr(),
            )
        };
        if status != ffi::vllm_status_VLLM_OK {
            if let Some(error) = params.logits_processor_error() {
                return Err(error);
            }
            status_result(status)?;
            unreachable!("non-OK native status unexpectedly succeeded");
        }
        // SAFETY: VLLM_OK initializes every completion field.
        let raw = unsafe { raw.assume_init() };
        let guard = CompletionGuard(raw);
        if let Some(error) = params.logits_processor_error() {
            return Err(error);
        }
        completion_from_raw(&guard.0)
    }

    /// Runs one blocking streaming text completion.
    ///
    /// A callback panic is resumed only after native code has aborted the request
    /// and returned across the FFI boundary.
    pub fn complete_stream<F>(
        &self,
        prompt: &str,
        params: &SamplingParams,
        mut callback: F,
    ) -> Result<StreamOutcome, Error>
    where
        F: FnMut(StreamEvent) -> StreamControl,
    {
        let prompt = to_cstring(prompt, "prompt")?;
        let params = params.marshal(&self.inner.compatibility)?;
        let mut state = CallbackState::new(&mut callback);
        // SAFETY: state has a stable stack address for this blocking call; the C
        // API does not retain user_data after returning.
        let status = unsafe {
            ffi::vllm_complete_stream(
                self.inner.raw.as_ptr(),
                prompt.as_ptr(),
                params.raw(),
                Some(callback_trampoline::<F>),
                ptr::from_mut(&mut state).cast(),
            )
        };
        if let Some(payload) = state.take_panic() {
            std::panic::resume_unwind(payload);
        }
        if let Some(error) = state.take_error() {
            return Err(error);
        }
        if let Some(error) = params.logits_processor_error() {
            return Err(error);
        }
        status_result(status)?;
        Ok(StreamOutcome {
            stopped_by_callback: state.stopped(),
        })
    }

    /// Runs one blocking OpenAI-style chat request and returns response JSON.
    pub fn chat_json(&self, request_json: &str) -> Result<String, Error> {
        let request = to_cstring(request_json, "chat request JSON")?;
        let mut output: *mut c_char = ptr::null_mut();
        // SAFETY: the engine and request pointers are valid for the call and the
        // returned string is released by NativeStringGuard.
        let status =
            unsafe { ffi::vllm_chat(self.inner.raw.as_ptr(), request.as_ptr(), &mut output) };
        status_result(status)?;
        let output = NonNull::new(output).ok_or_else(|| Error::Runtime {
            message: "vllm_chat succeeded without a response".to_owned(),
        })?;
        let guard = NativeStringGuard(output);
        c_string_to_owned(guard.0.as_ptr(), "chat response")
    }

    /// Runs one blocking OpenAI-style streaming chat request.
    pub fn chat_stream_json<F>(
        &self,
        request_json: &str,
        mut callback: F,
    ) -> Result<StreamOutcome, Error>
    where
        F: FnMut(StreamEvent) -> StreamControl,
    {
        let request = to_cstring(request_json, "chat request JSON")?;
        let mut state = CallbackState::new(&mut callback);
        // SAFETY: state remains valid for this blocking call and native code does
        // not retain it after returning.
        let status = unsafe {
            ffi::vllm_chat_stream(
                self.inner.raw.as_ptr(),
                request.as_ptr(),
                Some(callback_trampoline::<F>),
                ptr::from_mut(&mut state).cast(),
            )
        };
        if let Some(payload) = state.take_panic() {
            std::panic::resume_unwind(payload);
        }
        if let Some(error) = state.take_error() {
            return Err(error);
        }
        status_result(status)?;
        Ok(StreamOutcome {
            stopped_by_callback: state.stopped(),
        })
    }

    #[cfg(feature = "serde")]
    pub fn chat(&self, request: &serde_json::Value) -> Result<serde_json::Value, Error> {
        let request_json = serde_json::to_string(request).map_err(|error| Error::Json {
            context: "failed to serialize chat request",
            message: error.to_string(),
        })?;
        let response = self.chat_json(&request_json)?;
        serde_json::from_str(&response).map_err(|error| Error::Json {
            context: "failed to parse chat response",
            message: error.to_string(),
        })
    }
}

impl<Task> Drop for OwnedEngine<Task> {
    fn drop(&mut self) {
        // SAFETY: each OwnedEngine exclusively owns one live handle. Native
        // teardown joins engine workers before returning.
        unsafe { ffi::vllm_engine_free(self.raw.as_ptr()) };
    }
}

impl<Task> From<LoadedEngine<Task>> for OwnedEngine<Task> {
    fn from(loaded: LoadedEngine<Task>) -> Self {
        Self {
            raw: loaded.raw,
            compatibility: loaded.compatibility,
            _task: PhantomData,
            _not_send_sync: PhantomData,
        }
    }
}

// SAFETY: vllm.cpp documents concurrent text completion submissions as
// thread-safe, and Arc keeps the text engine live through active operations.
unsafe impl Send for OwnedEngine<TextTask> {}
// SAFETY: shared text owners submit through native AsyncLLM; Arc prevents
// destruction from racing an operation. No other task owner receives this impl.
unsafe impl Sync for OwnedEngine<TextTask> {}

impl EngineBuilder {
    #[must_use]
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            config: ModelConfig::new(model_path),
        }
    }

    #[must_use]
    pub fn tokenizer_config_path(mut self, value: impl Into<PathBuf>) -> Self {
        self.config.tokenizer_config_path = Some(value.into());
        self
    }

    #[must_use]
    pub fn block_size(mut self, value: u32) -> Self {
        self.config.block_size = Some(value);
        self
    }

    #[must_use]
    pub fn num_blocks(mut self, value: u32) -> Self {
        self.config.num_blocks = Some(value);
        self
    }

    #[must_use]
    pub fn max_model_len(mut self, value: u32) -> Self {
        self.config.max_model_len = Some(value);
        self
    }

    #[must_use]
    pub fn max_num_seqs(mut self, value: u32) -> Self {
        self.config.max_num_seqs = Some(value);
        self
    }

    #[must_use]
    pub fn tool_parser(mut self, value: impl Into<String>) -> Self {
        self.config.tool_parser = Some(value.into());
        self
    }

    #[must_use]
    pub fn reasoning_parser(mut self, value: impl Into<String>) -> Self {
        self.config.reasoning_parser = Some(value.into());
        self
    }

    #[must_use]
    pub fn speculative_config(mut self, value: impl Into<String>) -> Self {
        self.config.speculative_config = Some(value.into());
        self
    }

    #[must_use]
    pub fn prefix_caching(mut self, value: Toggle) -> Self {
        self.config.prefix_caching = value;
        self
    }

    #[must_use]
    pub fn max_num_batched_tokens(mut self, value: u32) -> Self {
        self.config.max_num_batched_tokens = Some(value);
        self
    }

    /// Selects the native admission queue policy.
    ///
    /// Raw and serde chat request JSON can carry a `priority` field that the native
    /// OpenAI-compatible path parses and submits. Direct completion, completion
    /// streaming, and `Request` submissions currently default to priority zero and
    /// tie by arrival; caller-selected priorities for those direct APIs require a
    /// future C ABI/API change.
    #[must_use]
    pub fn scheduler(mut self, value: SchedulerPolicy) -> Self {
        self.config.scheduler = Some(value);
        self
    }

    #[must_use]
    pub fn kv_transfer_config(mut self, value: impl Into<String>) -> Self {
        self.config.kv_transfer_config = Some(value.into());
        self
    }

    #[must_use]
    pub fn jump_forward(mut self, value: Toggle) -> Self {
        self.config.jump_forward = value;
        self
    }

    /// Selects the required native device.
    ///
    /// [`Device::Cuda`] never silently falls back when CUDA is unavailable.
    #[must_use]
    pub fn device(mut self, value: Device) -> Self {
        self.config.device = Some(value);
        self
    }

    /// Sets the native fraction used by GPU memory profiling.
    ///
    /// The value must be finite and strictly positive. Values above `1.0` are
    /// forwarded because the native ABI does not impose an upper bound. An
    /// explicit block count takes precedence over absolute KV-cache bytes, which
    /// take precedence over this utilization/profile setting.
    #[must_use]
    pub fn gpu_memory_utilization(mut self, value: f64) -> Self {
        self.config.gpu_memory_utilization = Some(value);
        self
    }

    /// Sets an absolute KV-cache memory budget in bytes.
    ///
    /// The value must be nonzero and fit the native signed 64-bit field. Native
    /// code validates the model-dependent minimum. An explicit block count takes
    /// precedence over this budget, and this budget takes precedence over GPU
    /// utilization/profile sizing.
    #[must_use]
    pub fn kv_cache_memory_bytes(mut self, value: u64) -> Self {
        self.config.kv_cache_memory_bytes = Some(value);
        self
    }

    pub fn load(self) -> Result<Engine, Error> {
        Ok(Engine {
            inner: Arc::new(load_engine::<TextTask>(self.config)?),
        })
    }
}

impl TranscriptionEngine {
    /// Loads a native engine owner with a transcription-only Rust method surface.
    ///
    /// ABI 17 cannot inspect the resolved task at load time. This constructor does
    /// not probe or infer checkpoint architecture; native task selection remains
    /// authoritative for future operations and diagnostics.
    pub fn load(model_path: impl Into<PathBuf>) -> Result<Self, Error> {
        Ok(Self {
            _inner: load_engine::<TranscriptionTask>(ModelConfig::new(model_path))?,
        })
    }
}

impl EmbeddingEngine {
    /// Loads a native engine owner with an embedding-only Rust method surface.
    ///
    /// ABI 17 cannot inspect the resolved task at load time. This constructor does
    /// not probe or infer checkpoint architecture; native task selection remains
    /// authoritative for future operations and diagnostics.
    pub fn load(model_path: impl Into<PathBuf>) -> Result<Self, Error> {
        Ok(Self {
            _inner: load_engine::<EmbeddingTask>(ModelConfig::new(model_path))?,
        })
    }
}

fn load_engine<Task>(config: ModelConfig) -> Result<OwnedEngine<Task>, Error> {
    load_engine_with(
        config,
        Compatibility::check,
        |compatibility| compatibility.model_params_default(),
        |params, output| {
            // SAFETY: the marshaled storage backing every pointer remains live for
            // the call and output points to writable handle storage.
            unsafe { ffi::vllm_engine_load(params, output) }
        },
    )
    .map(OwnedEngine::from)
}

fn load_engine_with<Task>(
    config: ModelConfig,
    check: impl FnOnce() -> Result<Compatibility, Error>,
    defaults: impl FnOnce(&Compatibility) -> ffi::vllm_model_params,
    load: impl FnOnce(&ffi::vllm_model_params, *mut *mut ffi::vllm_engine) -> ffi::vllm_status,
) -> Result<LoadedEngine<Task>, Error> {
    let mut params = MarshaledModelParams::new(config)?;
    let compatibility = check()?;
    params.apply_defaults(defaults(&compatibility));

    let mut output = ptr::null_mut();
    let status = load(params.raw(), &mut output);
    status_result(status)?;
    let raw = NonNull::new(output).ok_or_else(|| Error::ModelLoad {
        message: "vllm_engine_load succeeded without a handle".to_owned(),
    })?;
    Ok(LoadedEngine {
        raw,
        compatibility,
        _task: PhantomData,
    })
}

fn completion_from_raw(raw: &ffi::vllm_completion) -> Result<Completion, Error> {
    if raw.text.is_null() {
        return Err(Error::Runtime {
            message: "vllm_complete succeeded without text".to_owned(),
        });
    }
    let text = c_string_to_owned(raw.text, "completion text")?;
    let finish_reason = if raw.finish_reason.is_null() {
        None
    } else {
        Some(parse_finish_reason(c_string_to_owned(
            raw.finish_reason,
            "finish reason",
        )?))
    };
    Ok(Completion {
        text,
        finish_reason,
        prompt_tokens: count_to_u32(raw.prompt_tokens, "prompt token count")?,
        completion_tokens: count_to_u32(raw.completion_tokens, "completion token count")?,
    })
}

fn parse_finish_reason(value: String) -> FinishReason {
    match value.as_str() {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "abort" => FinishReason::Abort,
        "error" => FinishReason::Error,
        "repetition" => FinishReason::Repetition,
        "unknown" => FinishReason::Unknown,
        _ => FinishReason::Other(value),
    }
}

fn count_to_u32(value: i32, field: &'static str) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| invalid_configuration(format!("native {field} was negative")))
}

fn optional_u32_to_i32(value: Option<u32>, field: &'static str) -> Result<Option<i32>, Error> {
    value
        .map(|value| {
            i32::try_from(value)
                .map_err(|_| invalid_configuration(format!("{field} exceeds native i32 range")))
        })
        .transpose()
}

fn optional_cstring(value: Option<&str>, field: &'static str) -> Result<Option<CString>, Error> {
    value.map(|value| to_cstring(value, field)).transpose()
}

fn to_cstring(value: &str, field: &'static str) -> Result<CString, Error> {
    CString::new(value).map_err(|_| Error::InteriorNul { field })
}

#[cfg(unix)]
fn path_to_cstring(path: &Path, field: &'static str) -> Result<CString, Error> {
    use std::os::unix::ffi::OsStrExt;

    CString::new(path.as_os_str().as_bytes()).map_err(|_| Error::InteriorNul { field })
}

#[cfg(not(unix))]
fn path_to_cstring(path: &Path, field: &'static str) -> Result<CString, Error> {
    path.to_str()
        .ok_or(Error::PathEncoding)
        .and_then(|value| to_cstring(value, field))
}

fn c_string_to_owned(pointer: *const c_char, field: &'static str) -> Result<String, Error> {
    if pointer.is_null() {
        return Err(Error::InvalidUtf8 { field });
    }
    // SAFETY: callers pass a live native NUL-terminated string.
    unsafe { CStr::from_ptr(pointer) }
        .to_str()
        .map(str::to_owned)
        .map_err(|_| Error::InvalidUtf8 { field })
}

struct CompletionGuard(ffi::vllm_completion);

impl Drop for CompletionGuard {
    fn drop(&mut self) {
        // SAFETY: the native function initialized this completion and the guard
        // releases its owned members exactly once.
        unsafe { ffi::vllm_completion_free(&mut self.0) };
    }
}

struct NativeStringGuard(NonNull<c_char>);

impl Drop for NativeStringGuard {
    fn drop(&mut self) {
        // SAFETY: vllm_chat allocated this string and this guard owns it once.
        unsafe { ffi::vllm_string_free(self.0.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::ffi::CStr;
    use std::ptr::{self, NonNull};

    use vllm_cpp_sys as ffi;

    use super::{
        load_engine_with, Device, EmbeddingTask, MarshaledModelParams, ModelConfig,
        SchedulerPolicy, TextTask, Toggle, TranscriptionTask,
    };
    use crate::abi::Compatibility;
    use crate::Error;

    const NATIVE_STRING: &[u8] = b"native-default\0";

    fn native_defaults() -> ffi::vllm_model_params {
        let pointer = NATIVE_STRING.as_ptr().cast();
        ffi::vllm_model_params {
            model_path: pointer,
            tokenizer_config_path: pointer,
            block_size: 41,
            num_blocks: 42,
            max_model_len: 43,
            max_num_seqs: 44,
            tool_parser: pointer,
            reasoning_parser: pointer,
            speculative_config: pointer,
            enable_prefix_caching: 45,
            max_num_batched_tokens: 46,
            scheduling_policy: pointer,
            kv_transfer_config: pointer,
            enable_jump_forward: 47,
            device: 48,
            gpu_memory_utilization: 0.92,
            kv_cache_memory_bytes: 49,
        }
    }

    fn matching_compatibility() -> Result<Compatibility, Error> {
        Compatibility::check_with(|| ffi::VLLM_ABI_VERSION as i32)
    }

    fn c_string(pointer: *const std::os::raw::c_char) -> String {
        assert!(!pointer.is_null());
        // SAFETY: tests inspect pointers while their MarshaledModelParams owner is live.
        unsafe { CStr::from_ptr(pointer) }
            .to_str()
            .expect("UTF-8 test string")
            .to_owned()
    }

    #[test]
    fn default_application_preserves_every_unset_native_value() {
        let defaults = native_defaults();
        let mut params = MarshaledModelParams::new(ModelConfig::new("model-dir"))
            .expect("marshal default model config");
        params.apply_defaults(defaults);
        let raw = params.raw();

        assert_eq!(c_string(raw.model_path), "model-dir");
        assert_eq!(raw.tokenizer_config_path, defaults.tokenizer_config_path);
        assert_eq!(raw.block_size, defaults.block_size);
        assert_eq!(raw.num_blocks, defaults.num_blocks);
        assert_eq!(raw.max_model_len, defaults.max_model_len);
        assert_eq!(raw.max_num_seqs, defaults.max_num_seqs);
        assert_eq!(raw.tool_parser, defaults.tool_parser);
        assert_eq!(raw.reasoning_parser, defaults.reasoning_parser);
        assert_eq!(raw.speculative_config, defaults.speculative_config);
        assert_eq!(raw.enable_prefix_caching, defaults.enable_prefix_caching);
        assert_eq!(raw.max_num_batched_tokens, defaults.max_num_batched_tokens);
        assert_eq!(raw.scheduling_policy, defaults.scheduling_policy);
        assert_eq!(raw.kv_transfer_config, defaults.kv_transfer_config);
        assert_eq!(raw.enable_jump_forward, defaults.enable_jump_forward);
        assert_eq!(raw.device, defaults.device);
        assert_eq!(raw.gpu_memory_utilization, defaults.gpu_memory_utilization);
        assert_eq!(raw.kv_cache_memory_bytes, defaults.kv_cache_memory_bytes);
    }

    #[test]
    fn explicit_overrides_and_strings_reach_the_native_view() {
        let mut config = ModelConfig::new("model-dir");
        config.tokenizer_config_path = Some("tokenizer.json".into());
        config.block_size = Some(16);
        config.num_blocks = Some(32);
        config.max_model_len = Some(128);
        config.max_num_seqs = Some(2);
        config.tool_parser = Some("hermes".to_owned());
        config.reasoning_parser = Some("reasoning".to_owned());
        config.speculative_config = Some("{}".to_owned());
        config.prefix_caching = Toggle::Off;
        config.max_num_batched_tokens = Some(64);
        config.scheduler = Some(SchedulerPolicy::LongestPrefixMatch);
        config.kv_transfer_config = Some("{\"kv_role\":\"kv_both\"}".to_owned());
        config.jump_forward = Toggle::On;
        config.device = Some(Device::Cuda);
        config.gpu_memory_utilization = Some(1.25);
        config.kv_cache_memory_bytes = Some(4096);

        let mut params = MarshaledModelParams::new(config).expect("marshal explicit model config");
        params.apply_defaults(native_defaults());
        let raw = params.raw();

        assert_eq!(c_string(raw.model_path), "model-dir");
        assert_eq!(c_string(raw.tokenizer_config_path), "tokenizer.json");
        assert_eq!(raw.block_size, 16);
        assert_eq!(raw.num_blocks, 32);
        assert_eq!(raw.max_model_len, 128);
        assert_eq!(raw.max_num_seqs, 2);
        assert_eq!(c_string(raw.tool_parser), "hermes");
        assert_eq!(c_string(raw.reasoning_parser), "reasoning");
        assert_eq!(c_string(raw.speculative_config), "{}");
        assert_eq!(raw.enable_prefix_caching, Toggle::Off.as_native());
        assert_eq!(raw.max_num_batched_tokens, 64);
        assert_eq!(c_string(raw.scheduling_policy), "lpm");
        assert_eq!(
            c_string(raw.kv_transfer_config),
            "{\"kv_role\":\"kv_both\"}"
        );
        assert_eq!(raw.enable_jump_forward, Toggle::On.as_native());
        assert_eq!(raw.device, Device::Cuda.as_native());
        assert_eq!(raw.gpu_memory_utilization, 1.25);
        assert_eq!(raw.kv_cache_memory_bytes, 4096);
    }

    #[test]
    fn forwards_all_memory_settings_without_changing_native_precedence() {
        let mut config = ModelConfig::new("model-dir");
        config.num_blocks = Some(7);
        config.gpu_memory_utilization = Some(2.0);
        config.kv_cache_memory_bytes = Some(8192);

        let mut params = MarshaledModelParams::new(config).expect("marshal memory settings");
        params.apply_defaults(native_defaults());
        let raw = params.raw();
        assert_eq!(raw.num_blocks, 7);
        assert_eq!(raw.kv_cache_memory_bytes, 8192);
        assert_eq!(raw.gpu_memory_utilization, 2.0);
    }

    fn assert_shared_load_order<Task>() {
        let calls = RefCell::new(Vec::new());
        let loaded = load_engine_with::<Task>(
            ModelConfig::new("shared-model"),
            || {
                calls.borrow_mut().push("abi");
                matching_compatibility()
            },
            |_| {
                calls.borrow_mut().push("default");
                native_defaults()
            },
            |params, output| {
                calls.borrow_mut().push("load");
                assert_eq!(c_string(params.model_path), "shared-model");
                // SAFETY: output is writable storage supplied by load_engine_with;
                // the dangling non-null value is never dereferenced or freed.
                unsafe { *output = NonNull::<ffi::vllm_engine>::dangling().as_ptr() };
                ffi::vllm_status_VLLM_OK
            },
        )
        .expect("injected load");

        assert_eq!(*calls.borrow(), ["abi", "default", "load"]);
        assert_eq!(loaded.raw, NonNull::dangling());
    }

    #[test]
    fn shared_loader_checks_abi_before_defaults_for_every_task_marker() {
        assert_shared_load_order::<TextTask>();
        assert_shared_load_order::<TranscriptionTask>();
        assert_shared_load_order::<EmbeddingTask>();
    }

    #[test]
    fn rust_marshaling_failure_precedes_the_abi_probe() {
        let calls = RefCell::new(Vec::new());
        let result = load_engine_with::<TextTask>(
            ModelConfig::new("bad\0model"),
            || {
                calls.borrow_mut().push("abi");
                matching_compatibility()
            },
            |_| unreachable!("default helper must not run after marshaling failure"),
            |_, _| unreachable!("load must not run after marshaling failure"),
        );

        assert!(matches!(
            result,
            Err(Error::InteriorNul {
                field: "model path"
            })
        ));
        assert!(calls.borrow().is_empty());
    }

    #[test]
    fn successful_status_with_null_handle_is_rejected() {
        let result = load_engine_with::<TextTask>(
            ModelConfig::new("model-dir"),
            matching_compatibility,
            |_| native_defaults(),
            |_, output| {
                // SAFETY: output is writable storage supplied by load_engine_with.
                unsafe { *output = ptr::null_mut() };
                ffi::vllm_status_VLLM_OK
            },
        );

        assert!(matches!(result, Err(Error::ModelLoad { .. })));
    }
}
