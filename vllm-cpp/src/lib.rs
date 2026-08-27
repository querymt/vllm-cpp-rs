//! Safe model inference API for the stable vllm.cpp C boundary.
//!
//! # Entry points
//!
//! Resolve a Hub model with [`HuggingFaceModel`] (default `main`, or an explicit
//! revision), then create a text [`Engine`] with [`Engine::load`] or configure
//! native model settings through [`EngineBuilder`]. [`TranscriptionEngine`] and
//! [`EmbeddingEngine`] provide blocking task-specific operations. [`SamplingParams`]
//! owns sampling, stop-string, [`StructuredOutput`] settings, and an optional
//! host-side logits processor for completion calls. The text engine provides
//! blocking text and pre-tokenized completion, streaming, raw-JSON chat, and
//! [`Engine::submit`] for a concurrent [`Request`]. Enable `serde` for
//! `serde_json::Value` chat helpers.
//!
//! # Ownership and callbacks
//!
//! [`Engine`] is a cloneable RAII owner; clones share one reference-counted
//! native engine. Rust copies completion, transcription, embedding, stream,
//! chat, and error data before native storage is freed or reused. Pre-tokenized
//! prompts, audio, paths, and embedding strings are borrowed only for their
//! blocking calls. Blocking callbacks may borrow caller data.
//! Their panics are caught before the C boundary and resumed after the native
//! call returns. Custom logits processors are `Send + Sync`, may run concurrently
//! on native worker threads, and report contained panic through
//! [`Error::LogitsProcessorPanicked`].
//!
//! A [`Request`] retains its engine and asynchronous callback until native
//! free/join completes. Requests are `Send` but intentionally not `Sync`. The text
//! [`Engine`] is `Send + Sync`; transcription and embedding owners are
//! conservatively neither, must remain on their creating thread, and require
//! exclusive access for operations. Native embedding execution is serialized.
//! Asynchronous callbacks run on a native delivery thread, must be `Send + 'static`, and surface panic through
//! [`Error::CallbackPanicked`]. ABI version 17 forbids waiting for or freeing a
//! request from its callback thread; callback-thread drop delegates ownership to
//! a cleanup reaper instead. ABI 17 exposes no task-introspection API, so loading
//! cannot prove or infer a checkpoint's task. Native task selection and future
//! wrong-task diagnostics remain authoritative.
//!
//! # ABI, linking, and deployment
//!
//! Engine loading requires the linked native library's ABI to equal
//! [`expected_abi_version`] before versioned structs cross FFI. [`version`] copies
//! the linked library's diagnostic version string. The default
//! `bundled` feature builds the pinned native source. `system` selects a
//! caller-provided installation, `dynamic-link` selects shared linking, and
//! `serde` adds typed JSON helpers. The non-optional `hf-hub` dependency provides
//! synchronous, cache-aware model retrieval without an async runtime. CUDA,
//! CUTLASS, Triton AOT, Vulkan, Metal, and external MLX features are experimental
//! bundled build configuration.
//!
//! Dynamic linking does not deploy `libvllm.so` or `libvllm.dylib`; applications
//! must make it and its runtime dependencies visible through the platform loader,
//! such as `LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, or an application-owned rpath.
//! The supported runtime tier is native Linux x86_64 CPU.
//! Accelerator features are build/configuration surfaces with known runtime
//! blockers, not complete accelerator runtime support.

mod abi;
mod callback;
mod engine;
mod error;
mod hf;
mod params;
mod request;

pub use callback::{StreamControl, StreamEvent, StreamOutcome};
pub use engine::{
    Completion, EmbeddingEngine, EmbeddingResult, Engine, EngineBuilder, FinishReason,
    TokenCompletion, Transcription, TranscriptionEngine, TranscriptionInput,
};
pub use error::{Error, HuggingFaceError};
pub use hf::HuggingFaceModel;
pub use params::{Device, SamplingParams, SchedulerPolicy, StructuredOutput, Toggle};
pub use request::{Request, RequestOutcome};

/// Returns the compile-time C ABI expected by this crate.
#[must_use]
pub const fn expected_abi_version() -> i32 {
    vllm_cpp_sys::VLLM_ABI_VERSION as i32
}

/// Returns the C ABI reported by the linked vllm.cpp library.
///
/// Engine loading compares this value for exact equality before passing any
/// versioned native struct.
#[must_use]
pub fn abi_version() -> i32 {
    // SAFETY: this base ABI function takes no pointers and returns a plain i32.
    unsafe { vllm_cpp_sys::vllm_abi_version() }
}

/// Copies the version string reported by the linked vllm.cpp library.
///
/// This diagnostic does not replace [`abi_version`]: callers must still use the
/// numeric ABI for compatibility decisions.
pub fn version() -> Result<String, Error> {
    // SAFETY: the base ABI returns a borrowed, process-lifetime C string.
    let pointer = unsafe { vllm_cpp_sys::vllm_version() };
    if pointer.is_null() {
        return Err(Error::Runtime {
            message: "vllm_version returned a null pointer".to_owned(),
        });
    }
    // SAFETY: the native contract returns a live NUL-terminated string.
    unsafe { std::ffi::CStr::from_ptr(pointer) }
        .to_str()
        .map(str::to_owned)
        .map_err(|_| Error::InvalidUtf8 {
            field: "native version",
        })
}
