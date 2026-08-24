# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

## [0.0.1] - 2026-08-22

### Added

- Checked-in raw Rust declarations for the 19-symbol stable vllm.cpp C API at ABI version 10, with header, symbol, layout, and runtime conformance checks.
- A safe API for model loading, blocking completion and streaming, raw-JSON and optional serde chat, structured output, owned sampling parameters, panic-contained custom logits processors with request-scoped callback state, native version diagnostics, and concurrent request submission, cancellation, waiting, and diagnostics.
- An always-available synchronous `hf-hub` resolver for standalone GGUF files and runtime-complete sparse Safetensors snapshots, defaulting to the Hub's mutable `main` revision with an explicit branch/tag/commit override, explicit/`HF_TOKEN`/cached authentication precedence, cache/progress/offline controls, and no async runtime.
- Consistent local, Hugging Face GGUF, and Hugging Face Safetensors model-source arguments across every runnable example, with cache reuse and optional revisions; plus a weather extraction example and model-backed test using JSON-Schema structured output.
- A Clap-based interactive `chat` example with prompt/file startup input, retained system/user/assistant history, supported sampling controls, default streaming or blocking output, and shared local/Hugging Face resolution.
- RAII ownership for native engines, requests, completions, and strings, including callback panic containment and callback-thread-safe deferred request cleanup.
- Linux x86_64 CPU builds for bundled and system libraries with static or dynamic linking.
- Experimental bundled Linux x86_64/aarch64 build integration for CUDA, external CUTLASS, Triton AOT, and Vulkan.
- Bundled Apple ARM64 CPU and Metal build/link integration, plus optional external MLX integration with deterministic target/root/file validation and no packaged MLX payload or rpath.
- Manual hosted exact Rust 1.85.0, Linux ARM64 CPU, Apple ARM64 CPU/Metal compile-link, and Mesa llvmpipe Vulkan lanes.

### Compatibility

- The bundled native source is pinned to vllm.cpp commit `34aedfbe8ed9779697905541a62e2160ccfd9c05` and the Rust declarations require its C ABI version 10.
- The Rust crates are versioned together. `vllm-cpp` depends on exactly the matching `vllm-cpp-sys` version.

### Known limitations

- The priority scheduler is selectable, and raw and serde chat request JSON can carry a `priority` field that the native OpenAI-compatible path parses and submits. Direct completion, completion streaming, and `Request` submissions currently default to priority zero and tie by arrival; caller-selected priorities for those direct APIs remain deferred until a future C ABI/API change.

- The supported runtime tier is native Linux x86_64 CPU. Accelerator features are experimental build/configuration surfaces, not runtime-support claims.
- Known native blockers include a CUDA teardown SIGSEGV after otherwise successful tests, a CUDA bf16 numerical tolerance failure, CUTLASS concurrent-output differences, incomplete Vulkan attention/model runtime, and external MLX deployment plus unvalidated release-lane model/runtime behavior.
- The hosted Metal lane checks compile/link only, the software Vulkan lane checks backend/ops only, and accelerator builds do not establish runtime correctness.
- Dynamic builds require callers to deploy `libvllm.so` or `libvllm.dylib` and its runtime dependencies through a loader-visible path. System static builds must also provide the matching private BLAKE3 archive.
