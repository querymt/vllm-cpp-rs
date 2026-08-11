//! Safe Rust bindings for the stable vllm.cpp C API.
//!
//! [`Engine`] is a cloneable, shared owner of a complete native serving stack.
//! It provides blocking completion, streaming, and chat methods plus
//! [`Engine::submit`] for non-blocking requests. Each [`Request`] retains the
//! engine until native request free/join completes.

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
