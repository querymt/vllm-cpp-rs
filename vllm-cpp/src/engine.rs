use std::ffi::{CStr, CString};
use std::mem::MaybeUninit;
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::ptr::{self, NonNull};
use std::sync::Arc;

use vllm_cpp_sys as ffi;

use crate::callback::{
    callback_trampoline, CallbackState, StreamControl, StreamEvent, StreamOutcome,
};
use crate::error::{invalid_configuration, status_result, Error};
use crate::params::{SamplingParams, SchedulerPolicy, Toggle};

/// A cloneable vllm.cpp serving engine.
#[derive(Clone)]
pub struct Engine {
    pub(crate) inner: Arc<EngineInner>,
}

pub(crate) struct EngineInner {
    pub(crate) raw: NonNull<ffi::vllm_engine>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Engine")
            .field("raw", &self.inner.raw)
            .finish_non_exhaustive()
    }
}

/// Builder for one complete serving engine.
#[derive(Clone, Debug)]
pub struct EngineBuilder {
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
    scheduler: SchedulerPolicy,
    kv_transfer_config: Option<String>,
    jump_forward: Toggle,
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
        let params = params.marshal()?;
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
        let params = params.marshal()?;
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

impl Drop for EngineInner {
    fn drop(&mut self) {
        // SAFETY: EngineInner exclusively owns this live handle. Native teardown
        // joins engine workers before returning.
        unsafe { ffi::vllm_engine_free(self.raw.as_ptr()) };
    }
}

// SAFETY: vllm.cpp documents concurrent completion submissions as thread-safe,
// and EngineInner keeps the engine alive until the last shared owner is dropped.
unsafe impl Send for EngineInner {}
// SAFETY: shared references may submit concurrently through native AsyncLLM;
// destruction cannot race because Arc retains the handle for each active owner.
unsafe impl Sync for EngineInner {}

impl EngineBuilder {
    #[must_use]
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
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
            scheduler: SchedulerPolicy::Fcfs,
            kv_transfer_config: None,
            jump_forward: Toggle::Default,
        }
    }

    #[must_use]
    pub fn tokenizer_config_path(mut self, value: impl Into<PathBuf>) -> Self {
        self.tokenizer_config_path = Some(value.into());
        self
    }

    #[must_use]
    pub fn block_size(mut self, value: u32) -> Self {
        self.block_size = Some(value);
        self
    }

    #[must_use]
    pub fn num_blocks(mut self, value: u32) -> Self {
        self.num_blocks = Some(value);
        self
    }

    #[must_use]
    pub fn max_model_len(mut self, value: u32) -> Self {
        self.max_model_len = Some(value);
        self
    }

    #[must_use]
    pub fn max_num_seqs(mut self, value: u32) -> Self {
        self.max_num_seqs = Some(value);
        self
    }

    #[must_use]
    pub fn tool_parser(mut self, value: impl Into<String>) -> Self {
        self.tool_parser = Some(value.into());
        self
    }

    #[must_use]
    pub fn reasoning_parser(mut self, value: impl Into<String>) -> Self {
        self.reasoning_parser = Some(value.into());
        self
    }

    #[must_use]
    pub fn speculative_config(mut self, value: impl Into<String>) -> Self {
        self.speculative_config = Some(value.into());
        self
    }

    #[must_use]
    pub fn prefix_caching(mut self, value: Toggle) -> Self {
        self.prefix_caching = value;
        self
    }

    #[must_use]
    pub fn max_num_batched_tokens(mut self, value: u32) -> Self {
        self.max_num_batched_tokens = Some(value);
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
        self.scheduler = value;
        self
    }

    #[must_use]
    pub fn kv_transfer_config(mut self, value: impl Into<String>) -> Self {
        self.kv_transfer_config = Some(value.into());
        self
    }

    #[must_use]
    pub fn jump_forward(mut self, value: Toggle) -> Self {
        self.jump_forward = value;
        self
    }

    pub fn load(self) -> Result<Engine, Error> {
        let model_path = path_to_cstring(&self.model_path, "model path")?;
        let tokenizer_config_path = self
            .tokenizer_config_path
            .as_deref()
            .map(|path| path_to_cstring(path, "tokenizer config path"))
            .transpose()?;
        let tool_parser = optional_cstring(self.tool_parser.as_deref(), "tool parser")?;
        let reasoning_parser =
            optional_cstring(self.reasoning_parser.as_deref(), "reasoning parser")?;
        let speculative_config = optional_cstring(
            self.speculative_config.as_deref(),
            "speculative configuration",
        )?;
        let scheduling_policy = to_cstring(self.scheduler.as_str(), "scheduler policy")?;
        let kv_transfer_config = optional_cstring(
            self.kv_transfer_config.as_deref(),
            "KV transfer configuration",
        )?;

        let mut raw = checked_model_params_default()?;
        raw.model_path = model_path.as_ptr();
        raw.tokenizer_config_path = optional_pointer(tokenizer_config_path.as_ref());
        raw.block_size = optional_u32_to_i32(self.block_size, "block_size")?;
        raw.num_blocks = optional_u32_to_i32(self.num_blocks, "num_blocks")?;
        raw.max_model_len = optional_u32_to_i32(self.max_model_len, "max_model_len")?;
        raw.max_num_seqs = optional_u32_to_i32(self.max_num_seqs, "max_num_seqs")?;
        raw.tool_parser = optional_pointer(tool_parser.as_ref());
        raw.reasoning_parser = optional_pointer(reasoning_parser.as_ref());
        raw.speculative_config = optional_pointer(speculative_config.as_ref());
        raw.enable_prefix_caching = self.prefix_caching.as_native();
        raw.max_num_batched_tokens =
            optional_u32_to_i32(self.max_num_batched_tokens, "max_num_batched_tokens")?;
        raw.scheduling_policy = scheduling_policy.as_ptr();
        raw.kv_transfer_config = optional_pointer(kv_transfer_config.as_ref());
        raw.enable_jump_forward = self.jump_forward.as_native();

        let mut output = ptr::null_mut();
        // SAFETY: all string storage remains live for the call and output points
        // to writable handle storage.
        let status = unsafe { ffi::vllm_engine_load(&raw, &mut output) };
        status_result(status)?;
        let raw = NonNull::new(output).ok_or_else(|| Error::ModelLoad {
            message: "vllm_engine_load succeeded without a handle".to_owned(),
        })?;
        Ok(Engine {
            inner: Arc::new(EngineInner { raw }),
        })
    }
}

fn checked_model_params_default() -> Result<ffi::vllm_model_params, Error> {
    checked_model_params_default_with(
        || {
            // SAFETY: this base ABI function takes no pointers or versioned structs.
            unsafe { ffi::vllm_abi_version() }
        },
        || {
            // SAFETY: exact ABI equality was established immediately before this
            // by-value return of a versioned struct.
            unsafe { ffi::vllm_model_params_default() }
        },
    )
}

fn checked_model_params_default_with(
    abi_version: impl FnOnce() -> i32,
    model_params_default: impl FnOnce() -> ffi::vllm_model_params,
) -> Result<ffi::vllm_model_params, Error> {
    let actual = abi_version();
    let expected = ffi::VLLM_ABI_VERSION as i32;
    if actual != expected {
        return Err(Error::AbiMismatch { expected, actual });
    }
    Ok(model_params_default())
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

fn optional_u32_to_i32(value: Option<u32>, field: &'static str) -> Result<i32, Error> {
    match value {
        Some(0) | None => Ok(0),
        Some(value) => i32::try_from(value)
            .map_err(|_| invalid_configuration(format!("{field} exceeds native i32 range"))),
    }
}

fn optional_cstring(value: Option<&str>, field: &'static str) -> Result<Option<CString>, Error> {
    value.map(|value| to_cstring(value, field)).transpose()
}

fn optional_pointer(value: Option<&CString>) -> *const c_char {
    value.map_or(ptr::null(), |value| value.as_ptr())
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
    use super::checked_model_params_default_with;
    use crate::Error;
    use std::cell::RefCell;

    #[test]
    fn abi_mismatch_prevents_model_params_default_call() {
        let calls = RefCell::new(Vec::new());
        let result = checked_model_params_default_with(
            || {
                calls.borrow_mut().push("abi");
                10
            },
            || {
                calls.borrow_mut().push("default");
                unreachable!("default helper must not run after an ABI mismatch")
            },
        );

        assert!(matches!(
            result,
            Err(Error::AbiMismatch {
                expected: 17,
                actual: 10
            })
        ));
        assert_eq!(*calls.borrow(), ["abi"]);
    }
}
