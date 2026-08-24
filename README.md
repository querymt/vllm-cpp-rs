# vllm.cpp Rust Bindings

Rust bindings for [vllm.cpp](https://github.com/mudler/vllm.cpp), organized as:

- `vllm-cpp-sys`: raw C API bindings and the pinned native source build.
- `vllm-cpp`: the application-facing safe inference API.

## Status

The safe crate provides a cloneable engine API for local model loading, blocking completion and streaming, non-blocking concurrent requests, structured output, and raw-JSON chat. It also provides an always-available synchronous Hugging Face resolver for standalone GGUF files and runtime-complete sparse Safetensors snapshots, plus a Clap-based interactive chat example using those APIs. An optional `serde` feature adds `serde_json::Value` chat helpers. The sys crate provides checked-in generated FFI declarations with C/Rust layout checks and coverage for all 19 exported C symbols.

Linux x86_64 CPU builds support bundled static, bundled dynamic, system static, and system dynamic linking. Bundled CPU builds also target Linux aarch64 and Apple ARM64. Experimental bundled builds expose Linux x86_64/aarch64 build configuration for CUDA, external CUTLASS, Triton AOT, and Vulkan, plus Apple ARM64 Metal and external MLX configuration. Accelerator features are build integration surfaces, not runtime-support claims. vllm.cpp is pinned at `34aedfbe8ed9779697905541a62e2160ccfd9c05`, which exposes C ABI version 10.

## Prerequisites

Native builds require:

- Rust and Cargo.
- CMake 3.24 or newer.
- Ninja or another CMake build tool.
- A C11 and C++20 compiler.
- A system linker and C++ standard library.
- Just 1.40 or newer for maintainer workflows, plus Git, `jq`, and GNU tar.

This repository provides a Nix development shell with the pinned development tools. Linux also has minimal CUDA and Vulkan shells:

```console
nix develop
nix develop .#cuda
nix develop .#vulkan
nix develop .#msrv
```

## Checkout

vllm.cpp is a Git submodule. Clone recursively, or initialize it after cloning:

```console
git submodule update --init --recursive
```

## Safe API

The packaged [`vllm-cpp` guide](vllm-cpp/README.md) covers local and Hugging Face model resolution, safe ownership, callbacks, concurrency, features, link modes, and deployment. The [`vllm-cpp-sys` guide](vllm-cpp-sys/README.md) documents the raw ABI and native build boundary.

`Engine::load` accepts a native-compatible model directory or standalone GGUF. `HuggingFaceModel` synchronously resolves into the normal Hugging Face cache before engine construction, defaulting to the Hub's mutable `main` revision; `.revision(...)` can pin a branch, tag, or commit. GGUF mode selects one safe root file. Safetensors mode pins downloads to repository metadata's commit SHA and retrieves only native runtime requirements: root configuration/tokenizer files and either unsharded weights or an index plus all root shards. Every inference example accepts a bare or explicit local path and both Hub artifact forms with optional `--revision`. Cached downloads are reused. Retrieval does not prove model/backend compatibility.

`EngineBuilder` owns model settings and converts them to temporary C strings only for the load call. `SamplingParams` owns stop strings, structured constraints, and optional `Send + Sync` custom logits processors. Processor panics are contained before the C boundary and reported through Rust errors; processor-backed generation must be bounded because ABI v10 has no callback abort channel. Processor state remains registered only through the blocking call or asynchronous request lifetime; stale native invocations after cleanup become no-ops. `version()` copies the linked native diagnostic version string. Completion and chat strings are copied into Rust values before the matching native free function runs.

`Engine` is `Clone + Send + Sync`; each `Request` retains the shared engine until native callback delivery has joined. A request is `Send` but deliberately not `Sync`. `submit` returns before generation finishes, and `Request` provides `is_done`, idempotent `cancel`, `wait`, and copied `native_error` diagnostics. `wait` classifies completion as `Completed`, `StoppedByCallback`, or `Cancelled`; an explicit asynchronous `Stop` is classified as `StoppedByCallback` even when returned for the terminal event.

All streaming callbacks receive copied UTF-8 deltas. Blocking callbacks may borrow stack data; their panics are caught before the C boundary and resumed only after the native call returns. Asynchronous callbacks must be `Send + 'static`, run on a native delivery thread, and report panic as `Error::CallbackPanicked` from `wait`. Waiting for or freeing a request from its own callback thread is prohibited by ABI v10: `wait` returns `Error::RequestCallbackThread`, while drop transfers cleanup to a prestarted reaper that owns the request, callback, and engine until native free/cancel/join completes. Chat methods accept raw OpenAI-compatible request JSON; enable `serde` for `serde_json::Value` request and response helpers. `SchedulerPolicy::Priority` selects the native queue. Raw and serde chat request JSON can carry a `priority` field that the native OpenAI-compatible path parses and submits. Direct completion, completion streaming, and `Request` submissions currently default to priority zero and tie by arrival; caller-selected priorities for those direct APIs require a future C ABI/API change.

See [the examples guide](vllm-cpp/examples/README.md) for ordinary Linux and optional Nix setup, commands for every example, and the interactive chat CLI's local/Hub model forms and generation options. Release-facing changes are recorded in the [changelog](CHANGELOG.md), and maintainers use the manual [release process](RELEASING.md).

## Build and Test

Inside the Nix shell, use the root `Justfile` (requires Just 1.40 or newer) for maintainer workflows:

```console
CMAKE_GENERATOR=Ninja cargo build --locked --release
CMAKE_GENERATOR=Ninja cargo test --locked --workspace --release
cargo test --locked -p vllm-cpp --release --features serde
just ci
```

Set `CMAKE_BUILD_PARALLEL_LEVEL` to control native parallelism. The default bundled build remains deterministic and CPU-only: native tests, examples, the HTTP server, CUDA, Metal, MLX, Vulkan, Triton, and CUTLASS fetching are disabled explicitly. Use `nix develop .#msrv -c just msrv` for the exact local Rust 1.85.0 policy check; the manual `platforms` workflow runs the same exact toolchain policy.

`build.rs` is consumer-only native build/link integration; it does not download dependencies or compile/execute the maintainer layout probe. Ordinary consumers do not need Just, bindgen, or libclang. Normal first-time Cargo dependency resolution may access crates.io; use Cargo's standard `--offline` mode after dependencies are cached. The high-level crate's required `hf-hub` 0.5 dependency uses only its synchronous `ureq` feature, without Tokio or another async runtime. Library download progress is disabled by default.

## Experimental Backend Builds

Backend features are bundled-only and mutually exclusive with `system`; CUDA and Vulkan are also mutually exclusive. CUDA/CUTLASS/Triton/Vulkan target Linux x86_64/aarch64, while Metal/MLX require exact `aarch64-apple-darwin`. Backend features do not enable `bundled`: normal default-feature commands may use `--features cuda`, while `--no-default-features` callers must include it explicitly, for example `--features bundled,cuda`. Use a fresh `CARGO_TARGET_DIR` for every backend and link mode.

- `cuda` requires `VLLM_CPP_CUDA_ARCHITECTURES` equal to `80`, `86`, `87`, `89`, `90a`, `100a`, `103a`, `110`, `120a`, `121a`, or `120a;121a`. Leave this variable unset when `cuda` is disabled, including CPU and system builds.
- `cuda-cutlass` implies `cuda`, requires an explicit canonical `VLLM_CPP_CUTLASS_DIR` containing CUTLASS >=4.5.0, disables fetching, and rejects `103a` and `110`. Plain CUDA uses a nonexistent sentinel CUTLASS root so an ambient checkout cannot alter the build.
- `triton-aot` implies `cuda`, enables only checked-in AOT artifacts for one of `80`, `86`, `89`, `90a`, `100a`, or `121a`, and forces regeneration off.
- `vulkan` uses packaged Khronos headers and checked-in SPIR-V. It does not link a Vulkan SDK library; the native library opens the runtime loader dynamically.
- `metal` enables the native Metal backend on Apple ARM64 and links Apple's `Metal` and `Foundation` frameworks. Its MSL is compiled at runtime.
- `mlx` implies `metal` and requires canonical `MLX_ROOT` containing `include/mlx/array.h`, `lib/libmlx.dylib`, and `lib/mlx.metallib`. MLX remains an external dependency: Cargo neither fetches nor packages it and emits no machine-local rpath.

For example:

```console
nix develop .#cuda
VLLM_CPP_CUDA_ARCHITECTURES=120a \
  CARGO_TARGET_DIR=target/cuda-static \
  cargo build --locked --release --features cuda
VLLM_CPP_CUDA_ARCHITECTURES=120a \
  CARGO_TARGET_DIR=target/cuda-dynamic \
  cargo build --locked --release --features cuda,dynamic-link

nix develop .#vulkan
CARGO_TARGET_DIR=target/vulkan-static cargo build --locked --release --features vulkan

# Apple ARM64 only
CARGO_TARGET_DIR=target/metal-static cargo build --locked --release --features metal
MLX_ROOT=/absolute/path/to/mlx CARGO_TARGET_DIR=target/mlx-static \
  cargo build --locked --release --features mlx
```

Static CUDA links the exact `cudart`, `cublasLt`, and, for Triton, CUDA driver locations selected by CMake. Static Apple builds link `libc++`; Metal adds the `Metal` and `Foundation` frameworks, while MLX adds its canonical `lib` search path before `dylib=mlx`. Dynamic builds rely on the shared native library's transitive dependencies instead of repeating them through Cargo. Deploy `libvllm.so`/`libvllm.dylib` and optional toolkit/MLX libraries through normal loader paths.

Compilation does not establish runtime correctness. Known native evidence blockers remain: CUDA teardown can SIGSEGV after otherwise successful tests; CUDA bf16 testing has a numerical tolerance failure; CUTLASS concurrent output differs from the non-concurrent path; Vulkan attention/model runtime is incomplete; and MLX is an external, numerically distinct provider without release-lane model evidence. No accelerator runtime support is claimed here.

## Test Model and Sanitizers

Model-backed tests use Apache-2.0 `Qwen/Qwen3-0.6B` at pinned revision `c1899de289a04d12100db370d81485cdf75e47ca`. Explicitly resolve its complete Safetensors snapshot into the standard Hugging Face cache, then run exactly 18 blocking and request-lifecycle model tests serially, including choice and JSON-Schema structured-output enforcement:

```console
model=$(just setup-test-model)
VLLM_CPP_TEST_MODEL="$model" \
  cargo test --locked -p vllm-cpp --release --test qwen3 -- --test-threads=1
```

`just setup-test-model` is the only explicit test-fixture acquisition step. It uses `HuggingFaceModel` with the immutable revision above, honors normal `HF_HOME` and Hugging Face authentication, reuses the standard cache, and prints the resolved directory. The approximately 1.5 GB model is not included in repository or crate packages. Ordinary tests, sanitizers, and TSan never resolve or download models: `VLLM_CPP_TEST_MODEL` must name an externally prepared model directory. Model-backed tests skip with an explanatory message when it is unset; when set, tests and instrumentation recipes require it to be a directory.

AddressSanitizer, UndefinedBehaviorSanitizer, and leak detection run the full safe/request/model suites with native instrumentation. The Linux x86_64 GCC ThreadSanitizer lane runs selected request lifecycle tests individually and instruments native C++ only; it does not claim race coverage for Rust or the Rust standard library. Callback-thread self-drop remains in the normal and ASan/leak suites because its handoff uses uninstrumented Rust synchronization.

```console
just sanitizers "$model"
just tsan "$model"
```

`VLLM_CPP_SANITIZE` is a bundled-build test input. System mode rejects it because Cargo cannot infer whether an externally built native library carries matching instrumentation.

## Link Modes

- The default `bundled` feature builds and statically links the pinned CPU-only source.
- `bundled,dynamic-link` builds and dynamically links the pinned shared library.
- `system` requires `--no-default-features` and links a prefix selected by `VLLM_CPP_ROOT`.
- `system,dynamic-link` dynamically links the selected system library.

System-mode consumer builds validate that `VLLM_CPP_ROOT/include/vllm.h` exists; they do not compare its layout. The maintainer `layout-test` recipe compiles `tests/layout.c` at test runtime against the bundled header and compares its C layouts with the generated Rust declarations. The `link-modes` recipe repeats layout conformance against bundled and system fixture headers and also runs the runtime ABI test. `VLLM_CPP_LIB_DIR` selects a nonstandard vllm library directory. Upstream's normal CMake install does not install `libblake3_vendored.a`, so a stock install works directly with `system,dynamic-link`; system static users must provision that archive separately and set `VLLM_CPP_BLAKE3_LIB_DIR` (or place it in the vllm library directory). Dynamic tests and applications must make `libvllm.so` loader-visible with `LD_LIBRARY_PATH`, rpath, or an installed loader path; Cargo neither deploys the library nor configures rpath.

## Packaging

Inspect and test both packaged crates with:

```console
cargo package -p vllm-cpp-sys --locked --list
cargo package -p vllm-cpp --locked --list
just package-test
```

The package gate validates deterministic inventories for both crates, package metadata, required source/docs/examples/tests/licenses/native inputs, forbidden payloads, and license provenance. It extracts and tests both crates offline, then runs independent sys and safe downstream consumers; the safe consumer resolves both extracted crates rather than this workspace. The sys package carries only native build inputs and required licenses/notices; upstream tests, large fixtures, media, benchmarks, fetched SDKs, external CUTLASS trees, and agent records are excluded.

`just publish-dry-run` performs a sys-then-safe workspace packaging dry-run without uploading; it uses `--no-verify` to avoid the pre-publication registry cycle. As required by [RELEASING.md](RELEASING.md), after `vllm-cpp-sys` is available from crates.io, run the full `cargo publish -p vllm-cpp --locked --dry-run` verification before publishing the safe crate.

## Platform and Backend Validation

The manual `platforms` workflow provides exact Rust 1.85.0, Linux ARM64 CPU, Apple ARM64 CPU, Apple ARM64 Metal compile/link, and Mesa llvmpipe Vulkan jobs without duplicating ordinary Linux x86_64 CPU CI. The Vulkan job requires a real llvmpipe device and `storageBuffer16BitAccess`, then runs native backend/op gates; its scope is backend/op checking, not attention or model-inference support. The hosted Metal job checks compile/link only, not runtime correctness.

## Support

The supported runtime target is native Linux x86_64 CPU. Maintainer tests cover the four bundled/system static/dynamic CPU link modes plus bundled blocking and concurrent request inference with the pinned Qwen fixture. Sanitizer evidence covers native ASan/UBSan/leak detection and selected native-only GCC TSan lifecycle paths as described above. The manual Linux ARM64 and Apple ARM64 CPU jobs are configured for model-free build/test coverage. CUDA/CUTLASS/Triton/Vulkan/Metal/MLX remain experimental surfaces with the evidence boundaries and limitations listed above; CPU is the only supported runtime family.

## Licensing and Affiliation

The Rust workspace is dual-licensed under MIT or Apache-2.0. The root `LICENSE` contains the MIT terms, including the original QueryMate copyright, and `LICENSE-APACHE` contains the Apache-2.0 terms. Bundled components retain their own licenses and notices under `vllm-cpp-sys`.

vllm.cpp and these bindings are independent community projects. They are not affiliated with, endorsed by, or sponsored by the vLLM project, the PyTorch Foundation, or the Linux Foundation.
