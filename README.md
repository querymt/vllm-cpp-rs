# vllm.cpp Rust Bindings

Rust bindings for [vllm.cpp](https://github.com/mudler/vllm.cpp), organized as:

- `vllm-cpp-sys`: raw C API bindings and the pinned native source build.
- `vllm-cpp`: the application-facing safe inference API.

## Status

The safe crate provides a cloneable engine API for model loading, blocking completion and streaming, non-blocking concurrent requests, structured output, and raw-JSON chat. An optional `serde` feature adds `serde_json::Value` chat helpers. The sys crate provides checked-in generated FFI declarations with C/Rust layout checks and coverage for all 19 exported C symbols.

Linux x86_64 CPU builds support bundled static, bundled dynamic, system static, and system dynamic linking. Experimental bundled builds also expose Linux x86_64/aarch64 build configuration for CUDA, external CUTLASS, Triton AOT, and Vulkan. These accelerator features are build-only integration surfaces, not runtime-support claims. vllm.cpp is pinned at `34aedfbe8ed9779697905541a62e2160ccfd9c05`, which exposes C ABI version 10.

## Prerequisites

Initial development and testing support Linux CPU builds. They require:

- Rust and Cargo.
- CMake 3.24 or newer.
- Ninja or another CMake build tool.
- A C11 and C++20 compiler.
- A system linker and C++ standard library.
- Just 1.40 or newer for maintainer workflows, plus Git, `jq`, GNU tar, and `curl` for the model fixture recipe.

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

The packaged [`vllm-cpp` guide](vllm-cpp/README.md) covers model requirements, safe ownership, callbacks, concurrency, features, link modes, and deployment. The [`vllm-cpp-sys` guide](vllm-cpp-sys/README.md) documents the raw ABI and native build boundary.

`EngineBuilder` owns model settings and converts them to temporary C strings only for the load call. `SamplingParams` owns stop strings and structured constraints. Completion and chat strings are copied into Rust values before the matching native free function runs.

`Engine` is `Clone + Send + Sync`; each `Request` retains the shared engine until native callback delivery has joined. A request is `Send` but deliberately not `Sync`. `submit` returns before generation finishes, and `Request` provides `is_done`, idempotent `cancel`, `wait`, and copied `native_error` diagnostics. `wait` classifies completion as `Completed`, `StoppedByCallback`, or `Cancelled`; an explicit asynchronous `Stop` is classified as `StoppedByCallback` even when returned for the terminal event.

All streaming callbacks receive copied UTF-8 deltas. Blocking callbacks may borrow stack data; their panics are caught before the C boundary and resumed only after the native call returns. Asynchronous callbacks must be `Send + 'static`, run on a native delivery thread, and report panic as `Error::CallbackPanicked` from `wait`. Waiting for or freeing a request from its own callback thread is prohibited by ABI v10: `wait` returns `Error::RequestCallbackThread`, while drop transfers cleanup to a prestarted reaper that owns the request, callback, and engine until native free/cancel/join completes. Chat methods accept raw OpenAI-compatible request JSON; enable `serde` for `serde_json::Value` request and response helpers.

See [the examples guide](vllm-cpp/examples/README.md) for ordinary Linux and optional Nix setup and commands for every example. Release-facing changes are recorded in the [changelog](CHANGELOG.md), and maintainers use the manual [release process](RELEASING.md).

## Build and Test

Inside the Nix shell, use the root `Justfile` (requires Just 1.40 or newer) for maintainer workflows:

```console
CMAKE_GENERATOR=Ninja cargo build --locked --release
CMAKE_GENERATOR=Ninja cargo test --locked --workspace --release
cargo test --locked -p vllm-cpp --release --features serde
just ci
```

Set `CMAKE_BUILD_PARALLEL_LEVEL` to control native parallelism. The default bundled build remains deterministic and CPU-only: native tests, examples, the HTTP server, CUDA, Metal, MLX, Vulkan, Triton, and CUTLASS fetching are disabled explicitly. Use `nix develop .#msrv -c just msrv` for the exact local Rust 1.85.0 policy check; hosted exact-MSRV validation is deferred to a later CI slice.

`build.rs` is consumer-only native build/link integration; it does not download dependencies or compile/execute the maintainer layout probe. Ordinary consumers do not need Just, bindgen, or libclang. Normal first-time Cargo dependency resolution may access crates.io; use Cargo's standard `--offline` mode after dependencies are cached.

## Experimental Backend Builds

Backend features apply to bundled Linux x86_64/aarch64 builds only and are mutually exclusive with `system`; CUDA and Vulkan are also mutually exclusive. Backend features do not enable `bundled`: normal default-feature commands may use `--features cuda`, while `--no-default-features` callers must include it explicitly, for example `--features bundled,cuda`. Use a fresh `CARGO_TARGET_DIR` for every backend and link mode.

- `cuda` requires `VLLM_CPP_CUDA_ARCHITECTURES` equal to `80`, `86`, `87`, `89`, `90a`, `100a`, `103a`, `110`, `120a`, `121a`, or `120a;121a`. Leave this variable unset when `cuda` is disabled, including CPU and system builds.
- `cuda-cutlass` implies `cuda`, requires an explicit canonical `VLLM_CPP_CUTLASS_DIR` containing CUTLASS >=4.5.0, disables fetching, and rejects `103a` and `110`. Plain CUDA uses a nonexistent sentinel CUTLASS root so an ambient checkout cannot alter the build.
- `triton-aot` implies `cuda`, enables only checked-in AOT artifacts for one of `80`, `86`, `89`, `90a`, `100a`, or `121a`, and forces regeneration off.
- `vulkan` uses packaged Khronos headers and checked-in SPIR-V. It does not link a Vulkan SDK library; the native library opens the runtime loader dynamically.

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
```

Static CUDA links the exact `cudart`, `cublasLt`, and, for Triton, CUDA driver locations selected by CMake. Dynamic builds rely on `libvllm.so` `DT_NEEDED` entries instead of repeating those transitive Cargo links; deploy the shared library and toolkit libraries through normal loader paths.

Compilation does not establish runtime correctness. Known native evidence blockers remain: CUDA teardown can SIGSEGV after otherwise successful tests; CUDA bf16 testing has a numerical tolerance failure; CUTLASS concurrent output differs from the non-concurrent path; Vulkan runtime coverage is incomplete. No runtime support is claimed here.

## Test Model and Sanitizers

Model-backed tests use Apache-2.0 `Qwen/Qwen3-0.6B` at pinned revision `c1899de289a04d12100db370d81485cdf75e47ca`. Download or reuse the cache and verify every file, then run exactly 14 blocking and request-lifecycle model tests serially:

```console
model=$(just setup-test-model)
VLLM_CPP_TEST_MODEL="$model" \
  cargo test --locked -p vllm-cpp --release --test qwen3 -- --test-threads=1
```

The approximately 1.5 GB model stays in the user cache and is not included in repository or crate packages. Model-backed tests skip with an explanatory message when `VLLM_CPP_TEST_MODEL` is unset. When it is set, the test helper and sanitizer gate require `model.safetensors`, `config.json`, `tokenizer.json`, and `tokenizer_config.json` and report every missing file.

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

## Support

The supported runtime target is native Linux x86_64 CPU. Maintainer tests cover the four bundled/system static/dynamic CPU link modes plus bundled blocking and concurrent request inference with the pinned Qwen fixture. Sanitizer evidence covers native ASan/UBSan/leak detection and selected native-only GCC TSan lifecycle paths as described above. Linux CUDA/CUTLASS/Triton/Vulkan features remain experimental build-only surfaces with the limitations listed above; Apple and other accelerator targets are out of scope.

## Licensing and Affiliation

The Rust workspace is dual-licensed under MIT or Apache-2.0. The root `LICENSE` contains the MIT terms, including the original QueryMate copyright, and `LICENSE-APACHE` contains the Apache-2.0 terms. Bundled components retain their own licenses and notices under `vllm-cpp-sys`.

vllm.cpp and these bindings are independent community projects. They are not affiliated with, endorsed by, or sponsored by the vLLM project, the PyTorch Foundation, or the Linux Foundation.
