//! Safe Rust bindings for the stable vllm.cpp C API.
//!
//! The central [`Engine`] owns a complete native serving stack. Construct
//! request parameters in Rust, then use blocking completion or chat methods
//! without handling native pointers or free functions.

mod callback;
mod engine;
mod error;
mod params;

pub use callback::{StreamControl, StreamEvent, StreamOutcome};
pub use engine::{Completion, Engine, EngineBuilder, FinishReason};
pub use error::Error;
pub use params::{SamplingParams, SchedulerPolicy, StructuredOutput, Toggle};

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
