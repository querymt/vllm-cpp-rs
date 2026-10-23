use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::mem::{align_of, size_of, MaybeUninit};
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

/// A thread-local RAII owner for blocking native transcription.
///
/// ABI 17 has no task query, so loading cannot prove that a checkpoint supports
/// transcription. Native task selection and wrong-task [`Error::InvalidArgument`]
/// diagnostics remain authoritative. This owner is neither `Send` nor `Sync` and
/// operations require exclusive access.
pub struct TranscriptionEngine {
    inner: OwnedEngine<TranscriptionTask>,
}

/// A thread-local RAII owner for blocking native embeddings.
///
/// ABI 17 has no task query, so loading cannot prove that a checkpoint supports
/// embeddings. Native task selection and wrong-task [`Error::InvalidArgument`]
/// diagnostics remain authoritative. Native embedding batches are serialized;
/// this Rust owner is neither `Send` nor `Sync` and operations require exclusive
/// access.
pub struct EmbeddingEngine {
    inner: OwnedEngine<EmbeddingTask>,
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

/// A Rust-owned pre-tokenized completion result.
///
/// [`Self::token_ids`] contains only IDs that fit the caller's reporting buffer.
/// [`Self::truncated`] reports whether native generation produced additional IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenCompletion {
    pub token_ids: Vec<i32>,
    pub completion: Option<Completion>,
    pub truncated: bool,
}

/// Borrowed audio for one blocking transcription call.
///
/// The input is borrowed only until [`TranscriptionEngine::transcribe`] returns.
/// Rust does not inspect, decode, or resample either input form. WAV decoding and
/// the requirement for 16 kHz mono audio remain native behavior.
#[derive(Debug, Clone, Copy)]
pub enum TranscriptionInput<'a> {
    WavFile(&'a Path),
    Pcm {
        samples: &'a [f32],
        sample_rate: u32,
    },
}

/// A Rust-owned transcription result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transcription {
    pub text: Option<String>,
    pub token_ids: Vec<i32>,
}

/// A Rust-owned row-major embedding batch.
///
/// Rows preserve input order. The flattened values and all row views remain
/// independent of native result storage.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingResult {
    values: Vec<f32>,
    dimension: usize,
    prompt_tokens: u32,
}

impl EmbeddingResult {
    /// Returns all row-major values.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Returns the number of embedding rows.
    #[must_use]
    pub fn n_embeddings(&self) -> usize {
        self.values.len() / self.dimension
    }

    /// Returns the number of values in each row.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the total number of native input tokens for the batch.
    #[must_use]
    pub fn prompt_tokens(&self) -> u32 {
        self.prompt_tokens
    }

    /// Returns one row, or `None` when `index` is out of bounds.
    #[must_use]
    pub fn row(&self, index: usize) -> Option<&[f32]> {
        self.rows().nth(index)
    }

    /// Iterates over rows in input order.
    pub fn rows(&self) -> std::slice::ChunksExact<'_, f32> {
        self.values.chunks_exact(self.dimension)
    }
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
        let guard = NativeResultGuard::new(raw, ffi::vllm_completion_free);
        if let Some(error) = params.logits_processor_error() {
            return Err(error);
        }
        completion_from_raw(guard.raw())
    }

    /// Runs one blocking completion from a pre-tokenized prompt.
    ///
    /// `prompt_tokens` is borrowed only for this call. `max_output_tokens` limits
    /// how many generated IDs are reported; it does not limit generation. Native
    /// completion metadata is always requested to determine truncation accurately,
    /// even when `include_completion` omits it from the Rust result. All returned
    /// data is Rust-owned.
    pub fn complete_tokens(
        &self,
        prompt_tokens: &[i32],
        params: &SamplingParams,
        max_output_tokens: usize,
        include_completion: bool,
    ) -> Result<TokenCompletion, Error> {
        validate_token_input(prompt_tokens.len(), max_output_tokens)?;
        let params = params.marshal(&self.inner.compatibility)?;
        complete_tokens_with(
            prompt_tokens,
            params.raw(),
            max_output_tokens,
            include_completion,
            |prompt, n_prompt, params, output, capacity, written, completion| {
                // SAFETY: all borrowed and output storage remains live for this
                // blocking call, and the native result is initialized on success.
                unsafe {
                    ffi::vllm_complete_tokens(
                        self.inner.raw.as_ptr(),
                        prompt,
                        n_prompt,
                        params,
                        output,
                        capacity,
                        written,
                        completion,
                    )
                }
            },
            ffi::vllm_completion_free,
            || params.logits_processor_error(),
        )
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
    /// not probe or infer checkpoint architecture; native task selection and
    /// wrong-task diagnostics remain authoritative.
    pub fn load(model_path: impl Into<PathBuf>) -> Result<Self, Error> {
        Ok(Self {
            inner: load_engine::<TranscriptionTask>(ModelConfig::new(model_path))?,
        })
    }

    /// Runs one blocking transcription and returns Rust-owned text and token IDs.
    ///
    /// Input storage is borrowed only for this call. Rust does not decode WAV
    /// files, inspect PCM values, or resample audio; those checks and diagnostics
    /// are native behavior.
    pub fn transcribe(&mut self, input: TranscriptionInput<'_>) -> Result<Transcription, Error> {
        let input = MarshaledTranscriptionInput::new(input, &self.inner.compatibility)?;
        transcribe_with(
            &input,
            |params, output| {
                // SAFETY: the engine and marshaled input are live for this
                // blocking call, and output is initialized on success.
                unsafe { ffi::vllm_transcribe(self.inner.raw.as_ptr(), params, output) }
            },
            ffi::vllm_transcription_free,
        )
    }
}

impl EmbeddingEngine {
    /// Loads a native engine owner with an embedding-only Rust method surface.
    ///
    /// ABI 17 cannot inspect the resolved task at load time. This constructor does
    /// not probe or infer checkpoint architecture; native task selection and
    /// wrong-task diagnostics remain authoritative.
    pub fn load(model_path: impl Into<PathBuf>) -> Result<Self, Error> {
        Ok(Self {
            inner: load_engine::<EmbeddingTask>(ModelConfig::new(model_path))?,
        })
    }

    /// Runs one blocking native embedding batch.
    ///
    /// Input strings are borrowed only until this call returns. Native execution
    /// is serialized per engine; exclusive `&mut self` access also keeps this
    /// thread-local owner from overlapping Rust calls. The Rust-owned row-major
    /// result preserves input order.
    pub fn embed<I, S>(&mut self, texts: I) -> Result<EmbeddingResult, Error>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let texts = MarshaledEmbeddingInput::new(texts)?;
        embed_with(
            &texts,
            |pointers, count, output| {
                // SAFETY: all C strings and the pointer array remain live for this
                // blocking call, and output is initialized on success.
                unsafe { ffi::vllm_embed(self.inner.raw.as_ptr(), pointers, count, output) }
            },
            ffi::vllm_embedding_result_free,
        )
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

fn validate_token_input(prompt_len: usize, capacity: usize) -> Result<(i32, i32), Error> {
    if prompt_len == 0 {
        return Err(invalid_configuration("prompt_tokens must not be empty"));
    }
    let prompt_len = i32::try_from(prompt_len)
        .map_err(|_| invalid_configuration("prompt_tokens length exceeds native i32 range"))?;
    let capacity = i32::try_from(capacity)
        .map_err(|_| invalid_configuration("max_output_tokens exceeds native i32 range"))?;
    Ok((prompt_len, capacity))
}

fn complete_tokens_with(
    prompt_tokens: &[i32],
    params: &ffi::vllm_sampling_params,
    max_output_tokens: usize,
    include_completion: bool,
    call: impl FnOnce(
        *const i32,
        i32,
        *const ffi::vllm_sampling_params,
        *mut i32,
        i32,
        *mut i32,
        *mut ffi::vllm_completion,
    ) -> ffi::vllm_status,
    free: unsafe extern "C" fn(*mut ffi::vllm_completion),
    processor_error: impl FnOnce() -> Option<Error>,
) -> Result<TokenCompletion, Error> {
    let (prompt_len, capacity) = validate_token_input(prompt_tokens.len(), max_output_tokens)?;
    let mut token_ids = Vec::new();
    token_ids
        .try_reserve_exact(max_output_tokens)
        .map_err(|_| invalid_configuration("max_output_tokens cannot be allocated"))?;
    token_ids.resize(max_output_tokens, 0);
    let output = if token_ids.is_empty() {
        ptr::null_mut()
    } else {
        token_ids.as_mut_ptr()
    };
    let mut written = 0;
    let mut raw = MaybeUninit::<ffi::vllm_completion>::uninit();
    let status = call(
        prompt_tokens.as_ptr(),
        prompt_len,
        params,
        output,
        capacity,
        &mut written,
        raw.as_mut_ptr(),
    );
    if status != ffi::vllm_status_VLLM_OK {
        let native_error = status_result(status)
            .expect_err("a non-OK native completion status must produce an error");
        if let Some(error) = processor_error() {
            return Err(error);
        }
        return Err(native_error);
    }

    // SAFETY: the successful native call initialized every completion field.
    let raw = unsafe { raw.assume_init() };
    let guard = NativeResultGuard::new(raw, free);
    if let Some(error) = processor_error() {
        return Err(error);
    }
    token_completion_from_raw(
        token_ids,
        written,
        prompt_tokens.len(),
        include_completion,
        guard.raw(),
    )
}

fn token_completion_from_raw(
    mut token_ids: Vec<i32>,
    written: i32,
    prompt_len: usize,
    include_completion: bool,
    raw: &ffi::vllm_completion,
) -> Result<TokenCompletion, Error> {
    let written = usize::try_from(written)
        .map_err(|_| invalid_native_output("written token count", "count is negative"))?;
    if written > token_ids.len() {
        return Err(invalid_native_output(
            "written token count",
            "count exceeds output capacity",
        ));
    }
    let native_prompt = usize::try_from(raw.prompt_tokens)
        .map_err(|_| invalid_native_output("prompt token count", "count is negative"))?;
    if native_prompt != prompt_len {
        return Err(invalid_native_output(
            "prompt token count",
            "count does not match the input prompt",
        ));
    }
    let total = usize::try_from(raw.completion_tokens)
        .map_err(|_| invalid_native_output("completion token count", "count is negative"))?;
    if written != total.min(token_ids.len()) {
        return Err(invalid_native_output(
            "written token count",
            "count does not match completion metadata and capacity",
        ));
    }

    let completion = include_completion
        .then(|| completion_from_raw(raw))
        .transpose()?;
    token_ids.truncate(written);
    Ok(TokenCompletion {
        token_ids,
        completion,
        truncated: written < total,
    })
}

struct MarshaledTranscriptionInput<'a> {
    raw: ffi::vllm_transcription_params,
    _path: Option<CString>,
    _samples: PhantomData<&'a [f32]>,
}

impl<'a> MarshaledTranscriptionInput<'a> {
    fn new(input: TranscriptionInput<'a>, compatibility: &Compatibility) -> Result<Self, Error> {
        Self::new_with(input, || compatibility.transcription_params_default())
    }

    fn new_with(
        input: TranscriptionInput<'a>,
        defaults: impl FnOnce() -> ffi::vllm_transcription_params,
    ) -> Result<Self, Error> {
        match input {
            TranscriptionInput::WavFile(path) => {
                let path = path_to_cstring(path, "WAV path")?;
                let mut raw = defaults();
                raw.audio_path = path.as_ptr();
                Ok(Self {
                    raw,
                    _path: Some(path),
                    _samples: PhantomData,
                })
            }
            TranscriptionInput::Pcm {
                samples,
                sample_rate,
            } => {
                let (n_samples, sample_rate) = validate_pcm_input(samples.len(), sample_rate)?;
                let mut raw = defaults();
                raw.pcm = samples.as_ptr();
                raw.n_samples = n_samples;
                raw.sample_rate = sample_rate;
                Ok(Self {
                    raw,
                    _path: None,
                    _samples: PhantomData,
                })
            }
        }
    }

    fn raw(&self) -> &ffi::vllm_transcription_params {
        &self.raw
    }
}

fn validate_pcm_input(sample_count: usize, sample_rate: u32) -> Result<(i64, i32), Error> {
    if sample_count == 0 {
        return Err(invalid_configuration("PCM samples must not be empty"));
    }
    let sample_count = i64::try_from(sample_count)
        .map_err(|_| invalid_configuration("PCM sample count exceeds native i64 range"))?;
    if sample_rate == 0 {
        return Err(invalid_configuration(
            "PCM sample rate must be greater than zero",
        ));
    }
    let sample_rate = i32::try_from(sample_rate)
        .map_err(|_| invalid_configuration("PCM sample rate exceeds native i32 range"))?;
    Ok((sample_count, sample_rate))
}

fn transcribe_with(
    input: &MarshaledTranscriptionInput<'_>,
    call: impl FnOnce(
        *const ffi::vllm_transcription_params,
        *mut ffi::vllm_transcription,
    ) -> ffi::vllm_status,
    free: unsafe extern "C" fn(*mut ffi::vllm_transcription),
) -> Result<Transcription, Error> {
    let mut raw = MaybeUninit::<ffi::vllm_transcription>::uninit();
    let status = call(input.raw(), raw.as_mut_ptr());
    status_result(status)?;
    // SAFETY: the successful native call initialized every result field.
    let raw = unsafe { raw.assume_init() };
    let guard = NativeResultGuard::new(raw, free);
    transcription_from_raw(guard.raw())
}

fn transcription_from_raw(raw: &ffi::vllm_transcription) -> Result<Transcription, Error> {
    let text = match raw.has_text {
        0 if raw.text.is_null() => None,
        0 => {
            return Err(invalid_native_output(
                "transcription text",
                "pointer is non-null when has_text is zero",
            ));
        }
        1 if raw.text.is_null() => {
            return Err(invalid_native_output(
                "transcription text",
                "pointer is null when has_text is one",
            ));
        }
        1 => Some(c_string_to_owned(raw.text, "transcription text")?),
        _ => {
            return Err(invalid_native_output(
                "transcription has_text",
                "value is not zero or one",
            ));
        }
    };
    let token_count = usize::try_from(raw.n_token_ids)
        .map_err(|_| invalid_native_output("transcription token count", "count is negative"))?;
    validate_pointer_count(raw.token_ids, token_count, "transcription token IDs")?;
    let token_ids = checked_copy_slice(raw.token_ids, token_count, "transcription token IDs")?;
    Ok(Transcription { text, token_ids })
}

struct MarshaledEmbeddingInput {
    strings: Vec<CString>,
    pointers: Vec<*const c_char>,
    count: i32,
}

impl MarshaledEmbeddingInput {
    fn new<I, S>(texts: I) -> Result<Self, Error>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let strings = texts
            .into_iter()
            .map(|text| to_cstring(text.as_ref(), "embedding text"))
            .collect::<Result<Vec<_>, _>>()?;
        let count = validate_embedding_count(strings.len())?;
        let pointers = strings.iter().map(|text| text.as_ptr()).collect();
        Ok(Self {
            strings,
            pointers,
            count,
        })
    }

    fn pointers(&self) -> *const *const c_char {
        debug_assert!(!self.strings.is_empty());
        self.pointers.as_ptr()
    }
}

fn validate_embedding_count(count: usize) -> Result<i32, Error> {
    if count == 0 {
        return Err(invalid_configuration("embedding batch must not be empty"));
    }
    i32::try_from(count)
        .map_err(|_| invalid_configuration("embedding batch exceeds native i32 range"))
}

fn embed_with(
    input: &MarshaledEmbeddingInput,
    call: impl FnOnce(*const *const c_char, i32, *mut ffi::vllm_embedding_result) -> ffi::vllm_status,
    free: unsafe extern "C" fn(*mut ffi::vllm_embedding_result),
) -> Result<EmbeddingResult, Error> {
    let mut raw = MaybeUninit::<ffi::vllm_embedding_result>::uninit();
    let status = call(input.pointers(), input.count, raw.as_mut_ptr());
    status_result(status)?;
    // SAFETY: the successful native call initialized every result field.
    let raw = unsafe { raw.assume_init() };
    let guard = NativeResultGuard::new(raw, free);
    embedding_from_raw(guard.raw(), input.strings.len())
}

fn embedding_from_raw(
    raw: &ffi::vllm_embedding_result,
    expected_rows: usize,
) -> Result<EmbeddingResult, Error> {
    let rows = usize::try_from(raw.n_embeddings)
        .map_err(|_| invalid_native_output("embedding row count", "count is negative"))?;
    if rows == 0 {
        return Err(invalid_native_output(
            "embedding row count",
            "count is zero",
        ));
    }
    if rows != expected_rows {
        return Err(invalid_native_output(
            "embedding row count",
            "count does not match the input batch",
        ));
    }
    let dimension = usize::try_from(raw.dim)
        .map_err(|_| invalid_native_output("embedding dimension", "dimension is negative"))?;
    if dimension == 0 {
        return Err(invalid_native_output(
            "embedding dimension",
            "dimension is zero",
        ));
    }
    let prompt_tokens = native_count_to_u32(raw.prompt_tokens, "embedding prompt token count")?;
    let value_count = checked_product(rows, dimension, "embedding values")?;
    validate_pointer_count(raw.values, value_count, "embedding values")?;
    let values = checked_copy_slice(raw.values, value_count, "embedding values")?;
    Ok(EmbeddingResult {
        values,
        dimension,
        prompt_tokens,
    })
}

fn checked_product(left: usize, right: usize, field: &'static str) -> Result<usize, Error> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_native_output(field, "element count overflows usize"))
}

fn validate_pointer_count<T>(
    pointer: *const T,
    length: usize,
    field: &'static str,
) -> Result<(), Error> {
    if length == 0 {
        if pointer.is_null() {
            return Ok(());
        }
        return Err(invalid_native_output(
            field,
            "pointer is non-null for an empty result",
        ));
    }
    if pointer.is_null() {
        return Err(invalid_native_output(
            field,
            "pointer is null for a non-empty result",
        ));
    }
    if (pointer as usize) % align_of::<T>() != 0 {
        return Err(invalid_native_output(field, "pointer is not aligned"));
    }
    if length > isize::MAX as usize / size_of::<T>() {
        return Err(invalid_native_output(
            field,
            "element count exceeds addressable slice size",
        ));
    }
    Ok(())
}

fn checked_copy_slice<T: Copy>(
    pointer: *const T,
    length: usize,
    field: &'static str,
) -> Result<Vec<T>, Error> {
    validate_pointer_count(pointer, length, field)?;
    if length == 0 {
        return Ok(Vec::new());
    }
    // SAFETY: validation established a non-null, aligned, addressable range and
    // native keeps it live until the result guard is dropped.
    Ok(unsafe { std::slice::from_raw_parts(pointer, length) }.to_vec())
}

fn invalid_native_output(field: &'static str, message: &'static str) -> Error {
    Error::InvalidNativeOutput { field, message }
}

fn completion_from_raw(raw: &ffi::vllm_completion) -> Result<Completion, Error> {
    if raw.text.is_null() {
        return Err(invalid_native_output("completion text", "pointer is null"));
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
        prompt_tokens: native_count_to_u32(raw.prompt_tokens, "prompt token count")?,
        completion_tokens: native_count_to_u32(raw.completion_tokens, "completion token count")?,
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

fn native_count_to_u32(value: i32, field: &'static str) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| invalid_native_output(field, "count is negative"))
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

struct NativeResultGuard<T> {
    raw: T,
    free: unsafe extern "C" fn(*mut T),
}

impl<T> NativeResultGuard<T> {
    fn new(raw: T, free: unsafe extern "C" fn(*mut T)) -> Self {
        Self { raw, free }
    }

    fn raw(&self) -> &T {
        &self.raw
    }
}

impl<T> Drop for NativeResultGuard<T> {
    fn drop(&mut self) {
        // SAFETY: guards are armed only after successful native initialization
        // and uniquely release their result storage exactly once.
        unsafe { (self.free)(&mut self.raw) };
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
    use std::mem::{align_of, size_of};
    use std::path::Path;
    use std::ptr::{self, NonNull};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use vllm_cpp_sys as ffi;

    use super::{
        checked_product, complete_tokens_with, embed_with, embedding_from_raw, load_engine_with,
        token_completion_from_raw, transcribe_with, transcription_from_raw,
        validate_embedding_count, validate_pcm_input, validate_token_input, Device, EmbeddingTask,
        MarshaledEmbeddingInput, MarshaledModelParams, MarshaledTranscriptionInput, ModelConfig,
        SchedulerPolicy, TextTask, Toggle, TranscriptionInput, TranscriptionTask,
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

    static COMPLETION_FREES: AtomicUsize = AtomicUsize::new(0);
    static TRANSCRIPTION_FREES: AtomicUsize = AtomicUsize::new(0);
    static EMBEDDING_FREES: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn count_completion_free(_: *mut ffi::vllm_completion) {
        COMPLETION_FREES.fetch_add(1, Ordering::SeqCst);
    }

    unsafe extern "C" fn count_transcription_free(_: *mut ffi::vllm_transcription) {
        TRANSCRIPTION_FREES.fetch_add(1, Ordering::SeqCst);
    }

    unsafe extern "C" fn count_embedding_free(_: *mut ffi::vllm_embedding_result) {
        EMBEDDING_FREES.fetch_add(1, Ordering::SeqCst);
    }

    fn zeroed_sampling_params() -> ffi::vllm_sampling_params {
        // SAFETY: zero is a valid bit pattern for every generated C field.
        unsafe { std::mem::zeroed() }
    }

    fn raw_completion(
        text: *mut std::os::raw::c_char,
        prompt_tokens: i32,
        completion_tokens: i32,
    ) -> ffi::vllm_completion {
        ffi::vllm_completion {
            text,
            finish_reason: ptr::null(),
            prompt_tokens,
            completion_tokens,
        }
    }

    #[test]
    fn token_input_validation_rejects_empty_and_native_range_overflow() {
        assert!(matches!(
            validate_token_input(0, 0),
            Err(Error::InvalidConfiguration { .. })
        ));
        assert!(matches!(
            validate_token_input(1, i32::MAX as usize + 1),
            Err(Error::InvalidConfiguration { .. })
        ));
        #[cfg(target_pointer_width = "64")]
        assert!(matches!(
            validate_token_input(i32::MAX as usize + 1, 0),
            Err(Error::InvalidConfiguration { .. })
        ));
    }

    #[test]
    fn token_zero_capacity_uses_null_and_hidden_completion_metadata() {
        COMPLETION_FREES.store(0, Ordering::SeqCst);
        let text = b"generated\0";
        let params = zeroed_sampling_params();
        let result = complete_tokens_with(
            &[9707],
            &params,
            0,
            false,
            |prompt, n_prompt, _, output, capacity, written, completion| {
                assert!(!prompt.is_null());
                assert_eq!(n_prompt, 1);
                assert!(output.is_null());
                assert_eq!(capacity, 0);
                // SAFETY: all pointers target writable caller storage.
                unsafe {
                    *written = 0;
                    *completion = raw_completion(text.as_ptr().cast_mut().cast(), 1, 3);
                }
                ffi::vllm_status_VLLM_OK
            },
            count_completion_free,
            || None,
        )
        .expect("zero-capacity completion");

        assert!(result.token_ids.is_empty());
        assert!(result.completion.is_none());
        assert!(result.truncated);
        assert_eq!(COMPLETION_FREES.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn token_completion_copies_truncated_ids_and_optional_completion() {
        COMPLETION_FREES.store(0, Ordering::SeqCst);
        let text = b"ok\0";
        let finish = b"length\0";
        let params = zeroed_sampling_params();
        let result = complete_tokens_with(
            &[1, 2],
            &params,
            2,
            true,
            |_, _, _, output, _, written, completion| {
                // SAFETY: all pointers target writable caller storage.
                unsafe {
                    *output = 10;
                    *output.add(1) = 11;
                    *written = 2;
                    *completion = ffi::vllm_completion {
                        text: text.as_ptr().cast_mut().cast(),
                        finish_reason: finish.as_ptr().cast(),
                        prompt_tokens: 2,
                        completion_tokens: 4,
                    };
                }
                ffi::vllm_status_VLLM_OK
            },
            count_completion_free,
            || None,
        )
        .expect("truncated completion");

        assert_eq!(result.token_ids, [10, 11]);
        assert!(result.truncated);
        let completion = result.completion.expect("included completion");
        assert_eq!(completion.text, "ok");
        assert_eq!(completion.prompt_tokens, 2);
        assert_eq!(completion.completion_tokens, 4);
        assert_eq!(COMPLETION_FREES.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn token_metadata_relationships_are_validated() {
        let text = b"ok\0";
        let base = raw_completion(text.as_ptr().cast_mut().cast(), 1, 2);
        for (written, capacity, prompt, total) in [
            (-1, 2, 1, 2),
            (3, 2, 1, 3),
            (2, 2, 9, 2),
            (0, 2, 1, -1),
            (1, 2, 1, 2),
        ] {
            let raw = raw_completion(base.text, prompt, total);
            let result = token_completion_from_raw(vec![0; capacity], written, 1, false, &raw);
            assert!(matches!(result, Err(Error::InvalidNativeOutput { .. })));
        }
    }

    #[test]
    fn token_native_failure_discards_partial_output_without_arming_guard() {
        COMPLETION_FREES.store(0, Ordering::SeqCst);
        let params = zeroed_sampling_params();
        let result = complete_tokens_with(
            &[1],
            &params,
            1,
            false,
            |_, _, _, output, _, written, _| {
                // SAFETY: output and written point to caller-owned storage.
                unsafe {
                    *output = 99;
                    *written = 1;
                }
                ffi::vllm_status_VLLM_ERR_INVALID_ARGUMENT
            },
            count_completion_free,
            || None,
        );
        assert!(matches!(result, Err(Error::InvalidArgument { .. })));
        assert_eq!(COMPLETION_FREES.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn token_conversion_error_still_frees_once() {
        COMPLETION_FREES.store(0, Ordering::SeqCst);
        let invalid_utf8 = [0xff_u8, 0];
        let params = zeroed_sampling_params();
        let result = complete_tokens_with(
            &[1],
            &params,
            1,
            true,
            |_, _, _, output, _, written, completion| {
                // SAFETY: all pointers target writable caller storage.
                unsafe {
                    *output = 2;
                    *written = 1;
                    *completion = raw_completion(invalid_utf8.as_ptr().cast_mut().cast(), 1, 1);
                }
                ffi::vllm_status_VLLM_OK
            },
            count_completion_free,
            || None,
        );
        assert_eq!(
            result,
            Err(Error::InvalidUtf8 {
                field: "completion text"
            })
        );
        assert_eq!(COMPLETION_FREES.load(Ordering::SeqCst), 1);
    }

    fn transcription_defaults() -> ffi::vllm_transcription_params {
        ffi::vllm_transcription_params {
            audio_path: ptr::null(),
            pcm: ptr::null(),
            n_samples: 37,
            sample_rate: 38,
        }
    }

    #[test]
    fn transcription_marshaling_selects_and_retains_one_pointer_family() {
        let path = MarshaledTranscriptionInput::new_with(
            TranscriptionInput::WavFile(Path::new("audio.wav")),
            transcription_defaults,
        )
        .expect("path input");
        assert!(!path.raw.audio_path.is_null());
        assert!(path.raw.pcm.is_null());
        assert_eq!(c_string(path.raw.audio_path), "audio.wav");
        assert_eq!(path.raw.n_samples, 37);
        assert_eq!(path.raw.sample_rate, 38);

        let samples = [0.25, -0.5];
        let pcm = MarshaledTranscriptionInput::new_with(
            TranscriptionInput::Pcm {
                samples: &samples,
                sample_rate: 44_100,
            },
            transcription_defaults,
        )
        .expect("PCM input");
        assert!(pcm.raw.audio_path.is_null());
        assert_eq!(pcm.raw.pcm, samples.as_ptr());
        assert_eq!(pcm.raw.n_samples, 2);
        assert_eq!(pcm.raw.sample_rate, 44_100);
    }

    #[test]
    fn transcription_input_validation_rejects_invalid_pcm_and_path() {
        assert!(validate_pcm_input(0, 16_000).is_err());
        assert!(validate_pcm_input(1, 0).is_err());
        assert!(validate_pcm_input(1, i32::MAX as u32 + 1).is_err());
        #[cfg(target_pointer_width = "64")]
        assert!(validate_pcm_input(i64::MAX as usize + 1, 16_000).is_err());
        #[cfg(unix)]
        assert!(matches!(
            MarshaledTranscriptionInput::new_with(
                TranscriptionInput::WavFile(Path::new("bad\0path")),
                transcription_defaults,
            ),
            Err(Error::InteriorNul { field: "WAV path" })
        ));
    }

    #[test]
    fn transcription_metadata_and_optional_outputs_are_validated() {
        let text = b"text\0";
        let id = 7;
        let cases = [
            ffi::vllm_transcription {
                text: ptr::null_mut(),
                token_ids: ptr::null_mut(),
                n_token_ids: 0,
                has_text: 2,
            },
            ffi::vllm_transcription {
                text: text.as_ptr().cast_mut().cast(),
                token_ids: ptr::null_mut(),
                n_token_ids: 0,
                has_text: 0,
            },
            ffi::vllm_transcription {
                text: ptr::null_mut(),
                token_ids: ptr::null_mut(),
                n_token_ids: 0,
                has_text: 1,
            },
            ffi::vllm_transcription {
                text: ptr::null_mut(),
                token_ids: ptr::from_ref(&id).cast_mut(),
                n_token_ids: 0,
                has_text: 0,
            },
            ffi::vllm_transcription {
                text: ptr::null_mut(),
                token_ids: ptr::null_mut(),
                n_token_ids: 1,
                has_text: 0,
            },
            ffi::vllm_transcription {
                text: ptr::null_mut(),
                token_ids: ptr::null_mut(),
                n_token_ids: -1,
                has_text: 0,
            },
        ];
        for raw in cases {
            assert!(matches!(
                transcription_from_raw(&raw),
                Err(Error::InvalidNativeOutput { .. })
            ));
        }

        let empty = ffi::vllm_transcription {
            text: ptr::null_mut(),
            token_ids: ptr::null_mut(),
            n_token_ids: 0,
            has_text: 0,
        };
        assert_eq!(
            transcription_from_raw(&empty).expect("IDs-only empty result"),
            super::Transcription {
                text: None,
                token_ids: vec![]
            }
        );
    }

    #[test]
    fn transcription_copies_outputs_and_frees_once_on_conversion_error() {
        TRANSCRIPTION_FREES.store(0, Ordering::SeqCst);
        let samples = [0.0];
        let input = MarshaledTranscriptionInput::new_with(
            TranscriptionInput::Pcm {
                samples: &samples,
                sample_rate: 16_000,
            },
            transcription_defaults,
        )
        .expect("PCM input");
        let text = b"copied\0";
        let mut ids = vec![3, 4, 3];
        let copied = transcribe_with(
            &input,
            |_, output| {
                // SAFETY: output targets writable caller storage and test data stays live.
                unsafe {
                    *output = ffi::vllm_transcription {
                        text: text.as_ptr().cast_mut().cast(),
                        token_ids: ids.as_mut_ptr(),
                        n_token_ids: 3,
                        has_text: 1,
                    };
                }
                ffi::vllm_status_VLLM_OK
            },
            count_transcription_free,
        )
        .expect("copied transcription");
        ids.fill(9);
        assert_eq!(copied.text.as_deref(), Some("copied"));
        assert_eq!(copied.token_ids, [3, 4, 3]);
        assert_eq!(TRANSCRIPTION_FREES.load(Ordering::SeqCst), 1);

        let invalid_utf8 = [0xff_u8, 0];
        let result = transcribe_with(
            &input,
            |_, output| {
                // SAFETY: output targets writable caller storage.
                unsafe {
                    *output = ffi::vllm_transcription {
                        text: invalid_utf8.as_ptr().cast_mut().cast(),
                        token_ids: ptr::null_mut(),
                        n_token_ids: 0,
                        has_text: 1,
                    };
                }
                ffi::vllm_status_VLLM_OK
            },
            count_transcription_free,
        );
        assert!(matches!(result, Err(Error::InvalidUtf8 { .. })));
        assert_eq!(TRANSCRIPTION_FREES.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn embedding_marshaling_retains_strings_and_allows_empty_text() {
        let input = MarshaledEmbeddingInput::new(["first", "", "third"]).expect("embedding input");
        assert_eq!(input.count, 3);
        for (index, expected) in ["first", "", "third"].iter().enumerate() {
            // SAFETY: pointers refer to CStrings retained by input.
            let actual = unsafe { CStr::from_ptr(*input.pointers.as_ptr().add(index)) };
            assert_eq!(actual.to_str().expect("UTF-8"), *expected);
        }
        assert!(matches!(
            MarshaledEmbeddingInput::new(std::iter::empty::<&str>()),
            Err(Error::InvalidConfiguration { .. })
        ));
        assert!(matches!(
            MarshaledEmbeddingInput::new(["bad\0text"]),
            Err(Error::InteriorNul {
                field: "embedding text"
            })
        ));
        assert!(validate_embedding_count(i32::MAX as usize + 1).is_err());
    }

    #[test]
    fn embedding_metadata_rejects_shape_pointer_and_size_errors() {
        let aligned = NonNull::<f32>::dangling().as_ptr();
        let cases = [
            ffi::vllm_embedding_result {
                values: ptr::null_mut(),
                n_embeddings: 0,
                dim: 1,
                prompt_tokens: 0,
            },
            ffi::vllm_embedding_result {
                values: ptr::null_mut(),
                n_embeddings: 2,
                dim: 1,
                prompt_tokens: 0,
            },
            ffi::vllm_embedding_result {
                values: ptr::null_mut(),
                n_embeddings: 1,
                dim: 0,
                prompt_tokens: 0,
            },
            ffi::vllm_embedding_result {
                values: aligned,
                n_embeddings: 1,
                dim: 1,
                prompt_tokens: -1,
            },
            ffi::vllm_embedding_result {
                values: ptr::null_mut(),
                n_embeddings: 1,
                dim: 1,
                prompt_tokens: 0,
            },
            ffi::vllm_embedding_result {
                values: (align_of::<f32>() - 1) as *mut f32,
                n_embeddings: 1,
                dim: 1,
                prompt_tokens: 0,
            },
            ffi::vllm_embedding_result {
                values: aligned,
                n_embeddings: i32::MAX,
                dim: i32::MAX,
                prompt_tokens: 0,
            },
        ];
        for (index, raw) in cases.iter().enumerate() {
            let expected_rows = if index == 1 {
                1
            } else {
                raw.n_embeddings.max(1) as usize
            };
            assert!(matches!(
                embedding_from_raw(raw, expected_rows),
                Err(Error::InvalidNativeOutput { .. })
            ));
        }
        assert!(checked_product(usize::MAX, 2, "test product").is_err());
        assert!(i32::MAX as usize * i32::MAX as usize > isize::MAX as usize / size_of::<f32>());
    }

    #[test]
    fn embedding_rows_preserve_order_own_values_and_free_once() {
        EMBEDDING_FREES.store(0, Ordering::SeqCst);
        let input = MarshaledEmbeddingInput::new(["a", "b"]).expect("embedding input");
        let mut native_values = vec![1.0, 2.0, 3.0, 4.0];
        let result = embed_with(
            &input,
            |pointers, count, output| {
                assert_eq!(count, 2);
                assert!(!pointers.is_null());
                // SAFETY: output targets writable caller storage and values stays live.
                unsafe {
                    *output = ffi::vllm_embedding_result {
                        values: native_values.as_mut_ptr(),
                        n_embeddings: 2,
                        dim: 2,
                        prompt_tokens: 5,
                    };
                }
                ffi::vllm_status_VLLM_OK
            },
            count_embedding_free,
        )
        .expect("embedding result");
        native_values.fill(9.0);

        assert_eq!(result.values(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.n_embeddings(), 2);
        assert_eq!(result.dimension(), 2);
        assert_eq!(result.prompt_tokens(), 5);
        assert_eq!(result.row(0), Some(&[1.0, 2.0][..]));
        assert_eq!(result.row(2), None);
        assert_eq!(
            result.rows().collect::<Vec<_>>(),
            [&[1.0, 2.0][..], &[3.0, 4.0][..]]
        );
        assert_eq!(EMBEDDING_FREES.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn embedding_conversion_error_frees_once() {
        EMBEDDING_FREES.store(0, Ordering::SeqCst);
        let input = MarshaledEmbeddingInput::new(["a"]).expect("embedding input");
        let result = embed_with(
            &input,
            |_, _, output| {
                // SAFETY: output targets writable caller storage.
                unsafe {
                    *output = ffi::vllm_embedding_result {
                        values: ptr::null_mut(),
                        n_embeddings: 1,
                        dim: 1,
                        prompt_tokens: 1,
                    };
                }
                ffi::vllm_status_VLLM_OK
            },
            count_embedding_free,
        );
        assert!(matches!(result, Err(Error::InvalidNativeOutput { .. })));
        assert_eq!(EMBEDDING_FREES.load(Ordering::SeqCst), 1);
    }
}
