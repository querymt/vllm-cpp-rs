//! Safe model inference API for the stable vllm.cpp C boundary.
//!
//! # Entry points
//!
//! Create an [`Engine`] with [`Engine::load`] or configure native model settings
//! through [`EngineBuilder`]. [`SamplingParams`] owns sampling, stop-string, and
//! [`StructuredOutput`] settings for completion calls. The engine provides
//! blocking completion, streaming, raw-JSON chat, and [`Engine::submit`] for a
//! concurrent [`Request`]. Enable `serde` for `serde_json::Value` chat helpers.
//!
//! # Ownership and callbacks
//!
//! [`Engine`] is a cloneable RAII owner; clones share one reference-counted
//! native engine. Rust copies completion, stream, chat, and error text before
//! native storage is freed or reused. Blocking callbacks may borrow caller data.
//! Their panics are caught before the C boundary and resumed after the native
//! call returns.
//!
//! A [`Request`] retains its engine and asynchronous callback until native
//! free/join completes. Requests are `Send` but intentionally not `Sync`, while
//! engines are `Send + Sync`. Asynchronous callbacks run on a native delivery
//! thread, must be `Send + 'static`, and surface panic through
//! [`Error::CallbackPanicked`]. ABI version 10 forbids waiting for or freeing a
//! request from its callback thread; callback-thread drop delegates ownership to
//! a cleanup reaper instead.
//!
//! # ABI, linking, and deployment
//!
//! Engine loading requires the linked native library's ABI to equal
//! [`expected_abi_version`] before versioned structs cross FFI. The default
//! `bundled` feature builds the pinned native source. `system` selects a
//! caller-provided installation, `dynamic-link` selects shared linking, and
//! `serde` adds typed JSON helpers. CUDA, CUTLASS, Triton AOT, and Vulkan features
//! are experimental bundled build configuration.
//!
//! Dynamic linking does not deploy `libvllm.so`; applications must make it and
//! its runtime dependencies visible through `LD_LIBRARY_PATH`, rpath, or system
//! loader configuration. The supported runtime tier is native Linux x86_64 CPU.
//! Accelerator features are build/configuration surfaces with known runtime
//! blockers, not complete accelerator runtime support.

mod callback;
mod engine;
mod error;
mod params;
mod request;

pub use callback::{StreamControl, StreamEvent, StreamOutcome};
pub use engine::{Completion, Engine, EngineBuilder, FinishReason};
pub use error::Error;
pub use params::{SamplingParams, SchedulerPolicy, StructuredOutput, Toggle};
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
