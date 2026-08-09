# vllm-cpp-sys

Raw Rust bindings and native linking for [vllm.cpp](https://github.com/mudler/vllm.cpp).

This crate exposes generated unsafe functions that mirror the stable C API. Applications should use the application-facing `vllm-cpp` bindings when implemented. Ordinary consumers do not need Just, bindgen, or libclang.

## Link Modes

- `bundled` is enabled by default. It builds the pinned vllm.cpp source as a static CPU library.
- `dynamic-link` makes either source mode link `libvllm` dynamically.
- `system` disables the bundled build and links a caller-provided installation. Disable default features when selecting it.

`bundled` and `system` are mutually exclusive. Initial support is limited to native Linux CPU builds.

System mode requires `VLLM_CPP_ROOT`, whose prefix must contain `include/vllm.h` plus a `lib` or `lib64` directory. `VLLM_CPP_LIB_DIR` can override the vllm library directory. Consumer builds validate that the selected system header exists but do not compare its layout. The maintainer integration test compiles its C probe at test runtime against that header, compares its C layouts with the generated Rust declarations, and the runtime test requires ABI version 10.

Upstream's normal CMake install provides `libvllm` and `vllm.h` but does not install the private `libblake3_vendored.a` target. A stock install therefore works directly with `system,dynamic-link`. System static mode requires callers to provision the matching `libblake3_vendored.a` separately and set `VLLM_CPP_BLAKE3_LIB_DIR`; when the variable is unset, the build script checks the selected vllm library directory for backward compatibility. It does not search arbitrary build trees.

`dynamic-link` does not copy or package the shared library. Tests and applications must install `libvllm` in a loader-visible location or configure `LD_LIBRARY_PATH`, rpath, or another loader search path.

## Generated Bindings

The bundled source is pinned to commit `34aedfbe8ed9779697905541a62e2160ccfd9c05` and exposes C ABI version 10. Bindings are generated with bindgen 0.72.1 from `wrapper.h`, which includes `vllm.cpp/include/vllm.h`, and committed to `src/bindings.rs`. Maintainers use Just 1.40 or newer from the repository root:

```console
just bindings
just sys
just link-modes
```

The conformance gate verifies the generated output, C and C++ header compatibility, C/Rust layout, the exact 19-symbol export set, and compile-time and runtime ABI 10. CI separately runs bundled static/dynamic and fixture-backed system static/dynamic tests; dynamic tests set the required loader path. `build.rs` only performs consumer native build/link integration and does not compile or execute the layout probe. `tests/layout.rs` compiles and executes `tests/layout.c` with the bundled header or `VLLM_CPP_ROOT/include/vllm.h` at test runtime using the Rust standard library. Native Linux CPU is the supported target; cross-compiling that integration test is unsupported.

The Rust crate is dual-licensed under MIT or Apache-2.0. The bundled vllm.cpp source retains its upstream Apache-2.0 license and notices.

vllm.cpp is an independent community project and is not affiliated with or endorsed by the vLLM project, the PyTorch Foundation, or the Linux Foundation.
