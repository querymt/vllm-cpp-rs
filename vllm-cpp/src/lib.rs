//! Rust bindings for vllm.cpp.
//!
//! The high-level API will be added in follow-up work. This bootstrap crate establishes
//! the final workspace shape and verifies the linked C ABI.

/// Returns the C ABI version reported by the linked vllm.cpp library.
#[must_use]
pub fn abi_version() -> i32 {
    // SAFETY: this base ABI function takes no pointers and returns a plain i32.
    unsafe { vllm_cpp_sys::vllm_abi_version() }
}
