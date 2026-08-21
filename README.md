# vllm.cpp Rust Bindings

Rust bindings for [vllm.cpp](https://github.com/mudler/vllm.cpp), organized as:

- `vllm-cpp-sys`: raw C API bindings and the pinned native source build.
- `vllm-cpp`: the application-facing Rust bindings.

## Status

The sys crate provides checked-in generated raw FFI declarations. Conformance checks cover C/Rust layout, all 19 exported C symbols, and C ABI version 10. Linux x86_64 CPU builds support bundled static, bundled dynamic, system static, and system dynamic linking. A high-level API follows in a later PR.

vllm.cpp is pinned at `34aedfbe8ed9779697905541a62e2160ccfd9c05`, which exposes C ABI version 10.

## Prerequisites

Initial development and testing support Linux CPU builds. They require:

- Rust and Cargo.
- CMake 3.24 or newer.
- Ninja or another CMake build tool.
- A C11 and C++20 compiler.
- A system linker and C++ standard library.
- Just 1.40 or newer for maintainer workflows, plus Git, `jq`, and GNU tar.

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

Inside the Nix shell, use the root `Justfile` (requires Just 1.40 or newer) for maintainer workflows:

```console
CMAKE_GENERATOR=Ninja cargo build --locked --release
CMAKE_GENERATOR=Ninja cargo test --locked --workspace --release
just ci
```

Set `CMAKE_BUILD_PARALLEL_LEVEL` to control native parallelism. The bundled build is deterministic and CPU-only: native tests, examples, the HTTP server, CUDA, Metal, MLX, Vulkan, Triton, and CUTLASS fetching are disabled explicitly.

`build.rs` is consumer-only native build/link integration; it does not download dependencies or compile/execute the maintainer layout probe. Ordinary consumers do not need Just, bindgen, or libclang. Normal first-time Cargo dependency resolution may access crates.io; use Cargo's standard `--offline` mode after dependencies are cached. `just package-test` requires `jq` and GNU tar, fetches locked Cargo dependencies when `CARGO_NET_OFFLINE` is not `true`, and then validates the package offline.

## Link Modes

- The default `bundled` feature builds and statically links the pinned CPU-only source.
- `bundled,dynamic-link` builds and dynamically links the pinned shared library.
- `system` requires `--no-default-features` and links a prefix selected by `VLLM_CPP_ROOT`.
- `system,dynamic-link` dynamically links the selected system library.

System-mode consumer builds validate that `VLLM_CPP_ROOT/include/vllm.h` exists; they do not compare its layout. The maintainer `layout-test` recipe compiles `tests/layout.c` at test runtime against the bundled header and compares its C layouts with the generated Rust declarations. The `link-modes` recipe repeats layout conformance against bundled and system fixture headers and also runs the runtime ABI test. `VLLM_CPP_LIB_DIR` selects a nonstandard vllm library directory. Upstream's normal CMake install does not install `libblake3_vendored.a`, so a stock install works directly with `system,dynamic-link`; system static users must provision that archive separately and set `VLLM_CPP_BLAKE3_LIB_DIR` (or place it in the vllm library directory). Dynamic tests and applications must make `libvllm.so` loader-visible with `LD_LIBRARY_PATH`, rpath, or an installed loader path; Cargo neither deploys the library nor configures rpath. Native Linux CPU is the supported target; cross-compiling the layout integration test is unsupported because it executes the compiled target probe.

## Packaging

Inspect and build the sys package with:

```console
cargo package -p vllm-cpp-sys --locked --list
just package-test
```

The package carries only the native build inputs and required licenses/notices; upstream tests, large fixtures, media, benchmarks, and agent records are excluded. The package measures approximately 30 MiB unpacked and 4.2 MiB compressed.

## Support

CI runs Linux x86_64 CPU tests in all four link modes: bundled static/dynamic and system static/dynamic. The system prefix is a test fixture assembled from bundled build outputs; its separately copied BLAKE3 archive demonstrates the explicit system-static contract rather than claiming that a stock upstream install provides one. Other operating systems, architectures, and accelerator backends are not yet supported by the Rust build.

## Licensing and Affiliation

The Rust workspace is dual-licensed under MIT or Apache-2.0. The root `LICENSE` contains the MIT terms, including the original QueryMate copyright, and `LICENSE-APACHE` contains the Apache-2.0 terms. Bundled components retain their own licenses and notices under `vllm-cpp-sys`.

vllm.cpp and these bindings are independent community projects. They are not affiliated with, endorsed by, or sponsored by the vLLM project, the PyTorch Foundation, or the Linux Foundation.
