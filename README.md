# vllm.cpp Rust Bindings

Rust bindings for [vllm.cpp](https://github.com/mudler/vllm.cpp), split into a safe application API and a low-level C ABI crate.

## Status and identity

> Work in progress: this project follows a pinned native release and may lag later upstream APIs or backend behavior.

The bundled source is vllm.cpp `v0.0.2`, commit `7020de93652ca920424a10ac5255b34810dd2f24`. The stable C boundary is ABI 17 with 35 functions. `vllm-cpp-sys` contains the checked-in raw bindings; `vllm-cpp` provides RAII-based completion, chat, requests, transcription, embeddings, and video APIs.

Native Linux x86_64 CPU is the only supported runtime family, with bundled or system libraries and static or dynamic linking. Other platforms and CUDA, CUTLASS, Triton, Vulkan, Metal, and MLX are build or optional-validation surfaces, not runtime-support claims. Compilation and model-free tests do not establish model inference or backend correctness.

## Crates and guides

- [`vllm-cpp`](vllm-cpp/README.md) is the safe crate guide, including ownership, callbacks, Hub resolution, features, and deployment.
- [`vllm-cpp-sys`](vllm-cpp-sys/README.md) documents the unsafe ABI, native build, conformance checks, and link modes.
- [`vllm-cpp/examples`](vllm-cpp/examples/README.md) is a standalone guide to the text and experimental local-checkpoint video examples.

These package-local guides are the detailed API references; this root page covers workspace setup and maintainer checks.

## Requirements and checkout

Native builds require Rust/Cargo, CMake 3.24+, a C11 and C++20 compiler, a linker, and Ninja or another CMake generator. Just 1.40+, Git, `jq`, and GNU tar are maintainer tools. The optional Nix shells provide pinned development tools:

```console
nix develop
nix develop .#msrv
nix develop .#cuda
nix develop .#vulkan
```

The native source is a Git submodule. Clone recursively or initialize it after checkout:

```console
git submodule update --init --recursive
```

## Build and test

```console
CMAKE_GENERATOR=Ninja cargo build --locked --release
just test
just ci
nix develop .#msrv -c just msrv
```

`just test` runs locked workspace all-target tests with `serde`; `just ci` adds formatting, lint/docs, ABI and layout conformance, backend integrity, CPU link modes, native fixtures, sanitizers, package/downstream checks, and a publish dry-run. The exact Rust 1.85.0 check is separate. Fast `just build-support-test` and `just backend-config` recipes remain available without configuring CMake.

The default bundled build is deterministic and CPU-only. Native tests, examples, server, accelerator backends, and native dependency fetching are disabled. `build.rs` performs consumer build/link integration without network access; ordinary consumers do not need Just, bindgen, or libclang. Cargo may access crates.io during normal dependency resolution. The safe crate's synchronous `hf-hub` dependency does not add an async runtime; ordinary tests and instrumentation use a model only when one is explicitly prepared and supplied.

## Backend and packaging boundaries

Backend features are bundled-only, have explicit platform and prerequisite rules, and do not imply runtime support. See the safe and sys crate guides for exact feature contracts and deployment requirements. Use a fresh `CARGO_TARGET_DIR` for each backend/link combination.

Cargo manifests define package contents. `just package-test` compares archives with `cargo package --list`, checks metadata, native inputs, licenses and forbidden payloads, then tests extracted crates and independent downstream consumers offline. File counts, compressed sizes, payload bytes, and SHA-256 values are diagnostics, not caps.

`just publish-dry-run` runs only a workspace Cargo publish dry-run with `--no-verify` to avoid the registry dependency cycle. It does not upload crates or create tags. `just ci` runs package verification separately through `just package-test`. This dry-run does not replace the full registry-resolved safe-crate dry-run after the sys crate is available. See [RELEASING.md](RELEASING.md) for the sys-first release sequence and exact gates.

## Licensing and affiliation

The Rust workspace is dual-licensed under MIT or Apache-2.0. Bundled components retain their own licenses and notices under `vllm-cpp-sys`.

vllm.cpp and these bindings are independent community projects, not affiliated with, endorsed by, or sponsored by the vLLM project, the PyTorch Foundation, or the Linux Foundation.
