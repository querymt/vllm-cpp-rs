//! Raw FFI declarations for the stable vllm.cpp C API.
//!
//! The checked-in bindings are generated from `vllm.cpp/include/vllm.h` and
//! expose that header's 35-symbol C boundary, versioned structs, constants, and
//! callback signatures. They target ABI version 17 from pinned vllm.cpp commit
//! `7020de93652ca920424a10ac5255b34810dd2f24` (`v0.0.2`). This exported ABI is narrower than
//! the broader native C++ implementation and does not expose undocumented
//! vllm.cpp internals.
//!
//! # Safety
//!
//! The declarations are intentionally raw. Callers must uphold every pointer,
//! lifetime, aliasing, thread, callback, and NUL-termination contract from the C
//! header. They must check returned status values, copy thread-local error text
//! before another API call on that thread, and release engines, requests,
//! completions, and allocated strings with their matching `vllm_*_free`
//! functions. In particular, ABI version 17 prohibits waiting for or freeing a
//! request from that request's callback thread. Prefer the safe `vllm-cpp` crate
//! unless direct ABI access is required.
//!
//! # Build and link modes
//!
//! The default `bundled` feature compiles and statically links the packaged
//! pinned source. `system` links a caller-provided compatible installation, and
//! `dynamic-link` selects the platform `libvllm` shared library in either source
//! mode. Dynamic consumers must deploy that library and its dependencies through
//! the platform loader. CUDA, external CUTLASS, Triton AOT, Vulkan, Metal, and
//! external MLX features are experimental bundled build configuration and require
//! their documented targets and native inputs.
//!
//! Native Linux x86_64 CPU is the supported runtime tier. Building the bundled
//! source requires CMake 3.24 or newer, a build tool, C11 and C++20 compilers,
//! and a linker/C++ standard library. Documentation builds skip native
//! compilation; that does not validate a runtime library or accelerator.

#![allow(non_camel_case_types, non_upper_case_globals)]

include!("bindings.rs");
