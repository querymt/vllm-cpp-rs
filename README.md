# vllm.cpp Rust Bindings

Rust bindings for [vllm.cpp](https://github.com/mudler/vllm.cpp), organized as:

- `vllm-cpp-sys`: raw C API bindings and the pinned native source build.
- `vllm-cpp`: the application-facing Rust bindings.

## Status

This bootstrap PR establishes the workspace, reproducible bundled CPU build, source packaging, and link smoke tests. Later PRs add generated FFI and the high-level API.

vllm.cpp is pinned at `34aedfbe8ed9779697905541a62e2160ccfd9c05`, which exposes C ABI version 10.

## Prerequisites

Initial development and testing support Linux CPU builds. They require:

- Rust and Cargo.
- CMake 3.24 or newer.
- Ninja or another CMake build tool.
- A C11 and C++20 compiler.
- A system linker and C++ standard library.
- Git for source checkouts.

This repository provides a Nix development shell with the pinned development tools:

```console
nix develop
```

## Checkout

vllm.cpp is a Git submodule. Clone recursively, or initialize it after cloning:

```console
git submodule update --init --recursive
```

## Build and Test

Inside the Nix shell:

```console
CMAKE_GENERATOR=Ninja cargo build --release
CMAKE_GENERATOR=Ninja cargo test --workspace --release
```

Set `CMAKE_BUILD_PARALLEL_LEVEL` to control native parallelism. The bundled build is deterministic and CPU-only: native tests, examples, the HTTP server, CUDA, Metal, MLX, Vulkan, Triton, and CUTLASS fetching are disabled explicitly.

`build.rs` does not download native dependencies. Normal first-time Cargo dependency resolution may access crates.io; use Cargo's standard `--offline` mode after dependencies are cached.

## Packaging

Inspect and build the sys package with:

```console
cargo package -p vllm-cpp-sys --list
cargo package -p vllm-cpp-sys
```

The package carries only the native build inputs and required licenses/notices; upstream tests, large fixtures, media, benchmarks, and agent records are excluded. The bootstrap package measures approximately 30 MiB unpacked and 4.2 MiB compressed.

## Support

The bootstrap is validated on Linux x86_64 with a bundled static CPU build. Other operating systems, architectures, dynamic/system linking, and accelerator backends are not yet supported by the Rust build.

## Licensing and Affiliation

The Rust workspace is dual-licensed under MIT or Apache-2.0. The root `LICENSE` contains the MIT terms, including the original QueryMate copyright, and `LICENSE-APACHE` contains the Apache-2.0 terms. Bundled components retain their own licenses and notices under `vllm-cpp-sys`.

vllm.cpp and these bindings are independent community projects. They are not affiliated with, endorsed by, or sponsored by the vLLM project, the PyTorch Foundation, or the Linux Foundation.
