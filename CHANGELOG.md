# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- Checked-in raw Rust declarations for the 19-symbol stable vllm.cpp C API at ABI version 10, with header, symbol, layout, and runtime conformance checks.
- A safe API for model loading, blocking completion and streaming, raw-JSON and optional serde chat, structured output, owned sampling parameters, and concurrent request submission, cancellation, waiting, and diagnostics.
- RAII ownership for native engines, requests, completions, and strings, including callback panic containment and callback-thread-safe deferred request cleanup.
- Linux x86_64 CPU builds for bundled and system libraries with static or dynamic linking.
- Experimental bundled Linux x86_64/aarch64 build integration for CUDA, external CUTLASS, Triton AOT, and Vulkan.

### Compatibility

- The bundled native source is pinned to vllm.cpp commit `34aedfbe8ed9779697905541a62e2160ccfd9c05` and the Rust declarations require its C ABI version 10.
- The Rust crates are versioned together. `vllm-cpp` depends on exactly the matching `vllm-cpp-sys` version.

### Known limitations

- The supported runtime tier is native Linux x86_64 CPU. Accelerator features are experimental build/configuration surfaces, not runtime-support claims.
- Known native blockers include a CUDA teardown SIGSEGV after otherwise successful tests, a CUDA bf16 numerical tolerance failure, CUTLASS concurrent-output differences, and incomplete Vulkan runtime coverage.
- Dynamic builds require callers to deploy `libvllm.so` and its runtime dependencies through a loader-visible path. System static builds must also provide the matching private BLAKE3 archive.
