//! Raw bindings to the stable vllm.cpp C API.
//!
//! This crate exposes checked-in generated unsafe FFI declarations. Applications
//! should use the application-facing `vllm-cpp` bindings when implemented.

#![allow(non_camel_case_types, non_upper_case_globals)]

include!("bindings.rs");
