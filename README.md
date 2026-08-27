# vllm.cpp Rust Bindings

Rust bindings for [vllm.cpp](https://github.com/mudler/vllm.cpp), organized as:

- `vllm-cpp-sys`: raw C API bindings and the pinned native source build.
- `vllm-cpp`: the application-facing safe inference API.

## Status

> **Work in progress:** these bindings track a pinned vllm.cpp release and may lag later upstream APIs or backend behavior. Check the exact native identity and support boundary before adoption.

The safe crate covers text completion, streaming, raw-JSON and optional serde chat, concurrent requests, structured output, custom logits processing, pre-tokenized completion, transcription, embeddings, video generation, and standalone video-mux argument composition. It also includes a synchronous Hugging Face resolver and text-focused examples. The sys crate exposes the complete 35-function stable C boundary at ABI 17 with checked-in generated bindings and C/Rust conformance checks.

The native source is independently pinned to vllm.cpp tag `v0.0.2`, commit `7020de93652ca920424a10ac5255b34810dd2f24`. Native Linux x86_64 CPU is the supported runtime target, including bundled/system and static/dynamic linking. Linux ARM64, Apple ARM64, CUDA, external CUTLASS, Triton AOT, Vulkan, Metal, and external MLX are configured build or optional validation surfaces; they are not accelerator runtime-support claims.

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

The packaged [`vllm-cpp` guide](vllm-cpp/README.md) covers the full API, ownership, callbacks, filesystem effects, link modes, and deployment. The [`vllm-cpp-sys` guide](vllm-cpp-sys/README.md) documents the unsafe ABI and native build boundary.

`Engine::load` accepts a native-compatible model directory or standalone GGUF. `HuggingFaceModel` synchronously resolves a GGUF or runtime-complete sparse Safetensors snapshot into the normal cache; retrieval does not prove native model, task, or backend compatibility.

Text `Engine` is a cloneable `Send + Sync` RAII owner. It provides blocking completion, streaming, chat, structured output, custom logits processing, and `complete_tokens`, plus non-blocking `Request` submission. Requests retain an `Arc` to the engine, are `Send` but not `Sync`, and expose completion probes, cancellation, waiting, and copied diagnostics. Callback panics are contained before crossing C; asynchronous callbacks run on a native delivery thread, and callback-thread wait/free is rejected or deferred under the stable lifecycle contract.

`EngineBuilder` starts from `vllm_model_params_default()` and overlays only explicitly selected settings, preserving native helper defaults such as `block_size=32`, `max_num_seqs=32`, and `gpu_memory_utilization=0.92`. Text/task [`Device`](vllm-cpp/src/params.rs) numbering is `Auto=0`, `Cpu=1`, and `Cuda=2`; explicit CUDA never silently falls back. KV sizing precedence is `num_blocks` over `kv_cache_memory_bytes` over `gpu_memory_utilization` and its native profile/fallback path.

`complete_tokens` borrows caller-provided token IDs for one blocking call. Its output capacity limits only reported/copied IDs, not native generation; `truncated` compares the copied count with native completion metadata. `include_completion = false` suppresses the Rust metadata copy, while the hidden native metadata request still occurs so truncation remains accurate.

`TranscriptionEngine` and `EmbeddingEngine` are separate, non-cloneable, conservatively thread-local RAII owners. Their operations take `&mut self`, block, borrow call inputs, and copy results into Rust-owned values; embeddings are row-major and preserve input order. ABI 17 has no task-introspection function, so none of the task-specific owners can prove a checkpoint's task at load time. Native wrong-task `InvalidArgument` diagnostics remain authoritative.

`VideoEngine` separately owns a MiniMax-H3 checkpoint set and performs exclusive blocking generation. Video device numbering is `Cpu=0` and `Cuda=1`, with no `Auto`; explicit CUDA never falls back. Generation writes frame/audio artifacts, may leave stale or partial output, and trusts caller paths without sandboxing. `VideoMuxParams` and `VideoMuxArgv` only compose ordered `OsString` argument boundaries. This crate never executes ffmpeg, an HTTP server, or another process; `vllm_server_main` remains available only through the raw crate.

The high-level crate intentionally adds no tokenizer, task-query, raw-handle, process-execution, or HTTP-server wrapper. See [the examples guide](vllm-cpp/examples/README.md) for the intentionally text-focused binaries. Release-facing changes are recorded in the [changelog](CHANGELOG.md), and maintainers use the manual [release process](RELEASING.md).

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

Backend features are bundled-only and mutually exclusive with `system`; CUDA and Vulkan also conflict. CUDA/CUTLASS/Triton/Vulkan target Linux x86_64/aarch64, while Metal/MLX require exact `aarch64-apple-darwin`. Features do not imply runtime support and do not enable `bundled` for `--no-default-features` callers. Use a fresh `CARGO_TARGET_DIR` for each backend/link combination.

- `cuda` requires `VLLM_CPP_CUDA_ARCHITECTURES` equal to `80`, `86`, `87`, `89`, `90a`, `100a`, `103a`, `110`, `120a`, `121a`, or `120a;121a`.
- `cuda-cutlass` requires caller-provided CUTLASS >=4.5.0, disables fetching, and rejects `103a` and `110`.
- `triton-aot` packages and embeds all six checked-in AOT trees: `sm_80`, `sm_86`, `sm_89`, `sm_90a`, `sm_100a`, and `sm_121a`. Runtime dispatch selects only an exact SM match; other accepted CUDA targets, including `87`, `103a`, `110`, and `120a`, retain the portable C++/CUDA fallback. Regeneration remains disabled for consumer builds.
- `vulkan` uses packaged headers and SPIR-V and opens the runtime loader dynamically.
- `metal` links the Apple Metal and Foundation frameworks on Apple ARM64.
- `mlx` implies `metal` and requires an external `MLX_ROOT`; Cargo neither fetches nor packages MLX and emits no machine-local rpath.

A configuration example, not candidate runtime evidence:

```console
nix develop .#cuda
VLLM_CPP_CUDA_ARCHITECTURES=80 \
  CARGO_TARGET_DIR=target/cuda-static \
  cargo build --locked --release --features cuda
```

Static CUDA links toolkit libraries selected by CMake. Static Apple builds link `libc++` and the selected frameworks/providers. Dynamic builds rely on the native shared library's transitive dependencies. Deploy `libvllm.so`/`libvllm.dylib` and optional toolkit/MLX libraries through normal loader paths.

Compilation does not establish runtime correctness. CUDA/CUTLASS/Triton, Vulkan, Metal, and MLX remain experimental build surfaces; no accelerator runtime support is claimed.

## Optional Model and Sanitizer Recipes

The required Linux x86_64 CPU candidate gate is model-free and uses committed native fixtures. An optional prepared-model lane uses Apache-2.0 `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`; the model is never included in repository or crate packages. No ordinary test or instrumentation recipe downloads a model implicitly.

```console
model=$(just setup-test-model)
VLLM_CPP_TEST_MODEL="$model" \
  cargo test --locked -p vllm-cpp --release --test qwen3 -- --test-threads=1
just sanitizers "$model"
just tsan "$model"
```

These commands are optional evidence and are recorded only when rerun against the exact candidate. The TSan recipe instruments native C++ only and does not claim Rust standard-library race coverage. `VLLM_CPP_SANITIZE` is a bundled-build test input; system mode rejects it.

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

## Validation Boundary

Mandatory candidate evidence is Linux x86_64 CPU and model-free: formatting, lint, docs, workspace tests, generated-binding/header/layout/signature/ABI/exact-export checks, all four CPU link modes, native C API fixtures, ASan/UBSan/leak checks over committed fixtures, package extraction/downstream checks, publication dry-run, and exact MSRV validation.

Prepared Qwen inference/sanitizers, native-only TSan, successful Rust MiniMax-H3 generation, Miri, Linux ARM64, Apple ARM64, Vulkan, CUDA/CUTLASS/Triton, Metal/MLX, and accelerator runtime are optional or deferred lanes. The manual platform workflows are configuration, not exact-candidate evidence unless separately dispatched and recorded.

## Support

Native Linux x86_64 CPU is the supported runtime family. All accelerator features remain experimental build/configuration surfaces. Cross-platform compile or workflow configuration alone does not establish runtime support.

## Licensing and Affiliation

The Rust workspace is dual-licensed under MIT or Apache-2.0. The root `LICENSE` contains the MIT terms, including the original QueryMate copyright, and `LICENSE-APACHE` contains the Apache-2.0 terms. Bundled components retain their own licenses and notices under `vllm-cpp-sys`.

vllm.cpp and these bindings are independent community projects. They are not affiliated with, endorsed by, or sponsored by the vLLM project, the PyTorch Foundation, or the Linux Foundation.
