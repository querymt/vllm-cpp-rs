# vllm.cpp Rust Bindings

Rust bindings for [vllm.cpp](https://github.com/mudler/vllm.cpp), organized as:

- `vllm-cpp-sys`: raw C API bindings and the pinned native source build.
- `vllm-cpp`: the application-facing safe blocking API.

## Status

The safe crate provides an owned blocking engine API for model loading, completion, streaming, structured output, and raw-JSON chat. An optional `serde` feature adds `serde_json::Value` chat helpers. The sys crate provides checked-in generated FFI declarations with C/Rust layout checks and coverage for all 19 exported C symbols.

Linux x86_64 CPU builds support bundled static, bundled dynamic, system static, and system dynamic linking. vllm.cpp is pinned at `34aedfbe8ed9779697905541a62e2160ccfd9c05`, which exposes C ABI version 10.

## Prerequisites

Initial development and testing support Linux CPU builds. They require:

- Rust and Cargo.
- CMake 3.24 or newer.
- Ninja or another CMake build tool.
- A C11 and C++20 compiler.
- A system linker and C++ standard library.
- Just 1.40 or newer for maintainer workflows, plus Git, `jq`, GNU tar, and `curl` for the model fixture recipe.

This repository provides a Nix development shell with the pinned development tools:

```console
nix develop
```

## Checkout

vllm.cpp is a Git submodule. Clone recursively, or initialize it after cloning:

```console
git submodule update --init --recursive
```

## Safe API

```rust
use vllm_cpp::{Engine, SamplingParams};

let engine = Engine::load("/models/Qwen3-0.6B")?;
let completion = engine.complete(
    "The capital of France is",
    &SamplingParams::greedy().max_tokens(16),
)?;
println!("{}", completion.text);
# Ok::<(), vllm_cpp::Error>(())
```

`EngineBuilder` owns model settings and converts them to temporary C strings only for the load call. `SamplingParams` owns stop strings and structured constraints. Completion and chat strings are copied into Rust values before the matching native free function runs.

Blocking streaming callbacks receive copied UTF-8 deltas. Callback panics are caught before the C boundary and resumed only after the native call has stopped and returned. Chat methods accept raw OpenAI-compatible request JSON; enable `serde` for `serde_json::Value` request and response helpers.

Run the practical examples with a model directory:

```console
cargo run -p vllm-cpp --example complete -- <model-directory>
cargo run -p vllm-cpp --example stream -- <model-directory>
cargo run -p vllm-cpp --example chat -- <model-directory>
cargo run -p vllm-cpp --example structured -- <model-directory>
```

## Build and Test

Inside the Nix shell, use the root `Justfile` (requires Just 1.40 or newer) for maintainer workflows:

```console
CMAKE_GENERATOR=Ninja cargo build --locked --release
CMAKE_GENERATOR=Ninja cargo test --locked --workspace --release
cargo test --locked -p vllm-cpp --release --features serde
just ci
```

Set `CMAKE_BUILD_PARALLEL_LEVEL` to control native parallelism. The bundled build is deterministic and CPU-only: native tests, examples, the HTTP server, CUDA, Metal, MLX, Vulkan, Triton, and CUTLASS fetching are disabled explicitly.

`build.rs` is consumer-only native build/link integration; it does not download dependencies or compile/execute the maintainer layout probe. Ordinary consumers do not need Just, bindgen, or libclang. Normal first-time Cargo dependency resolution may access crates.io; use Cargo's standard `--offline` mode after dependencies are cached.

## Test Model and Sanitizers

Model-backed tests use Apache-2.0 `Qwen/Qwen3-0.6B` at pinned revision `c1899de289a04d12100db370d81485cdf75e47ca`. Download or reuse the cache and verify every file, then run exactly the six blocking model tests serially:

```console
model=$(just setup-test-model)
VLLM_CPP_TEST_MODEL="$model" \
  cargo test --locked -p vllm-cpp --release --test qwen3 -- --test-threads=1
```

The approximately 1.5 GB model stays in the user cache and is not included in repository or crate packages. Model-backed tests skip with an explanatory message when `VLLM_CPP_TEST_MODEL` is unset. When it is set, the test helper and sanitizer gate require `model.safetensors`, `config.json`, `tokenizer.json`, and `tokenizer_config.json` and report every missing file.

Address, undefined-behavior, and leak checks run the blocking safe API and six model tests with native instrumentation:

```console
just sanitizers "$model"
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

The package gate preserves the sys crate inventory, tests the extracted sys crate and downstream fixture offline, then points the extracted safe crate at the extracted sys crate and tests it offline with `bundled,serde`. It also rejects local paths, build output, and model files in the safe package. The sys package carries only native build inputs and required licenses/notices; upstream tests, large fixtures, media, benchmarks, and agent records are excluded.

## Support

The supported target is native Linux x86_64 CPU. Maintainer tests cover the four bundled/system static/dynamic link modes and bundled blocking inference with the pinned Qwen fixture. Other operating systems, architectures, and accelerator builds are not supported by this Rust build.

## Licensing and Affiliation

The Rust workspace is dual-licensed under MIT or Apache-2.0. The root `LICENSE` contains the MIT terms, including the original QueryMate copyright, and `LICENSE-APACHE` contains the Apache-2.0 terms. Bundled components retain their own licenses and notices under `vllm-cpp-sys`.

vllm.cpp and these bindings are independent community projects. They are not affiliated with, endorsed by, or sponsored by the vLLM project, the PyTorch Foundation, or the Linux Foundation.
