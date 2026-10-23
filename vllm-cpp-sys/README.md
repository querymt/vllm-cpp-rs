# vllm-cpp-sys

Raw Rust bindings and native linking for the stable C API of [vllm.cpp](https://github.com/mudler/vllm.cpp).

This crate exposes checked-in generated unsafe declarations for the 35 exported C symbols in ABI version 17. Callers are responsible for pointer validity, lifetimes, callback threading, status/error handling, and matching every native allocation with its documented free function. Applications should prefer the current safe [`vllm-cpp`](https://docs.rs/vllm-cpp) crate unless they require direct ABI access. Ordinary consumers do not need Just, bindgen, or libclang.

The package contains Rust declarations and conformance tests, native build/link integration, the pinned native source inputs required by the supported feature set, and their licenses/notices. It excludes upstream test trees except for one unconditional ROCm syntax-check translation unit, plus fixtures, models, examples, benchmarks, agent records, fetched SDKs, external CUTLASS trees, and build output. See the repository [changelog](https://github.com/querymt/vllm-cpp-rs/blob/main/CHANGELOG.md) and [release process](https://github.com/querymt/vllm-cpp-rs/blob/main/RELEASING.md) for the coordinated crate boundary.

## Link Modes

- `bundled` is enabled by default. It builds the pinned vllm.cpp source as a static CPU library.
- `dynamic-link` makes either source mode link `libvllm` dynamically.
- `system` disables the bundled build and links a caller-provided installation. Disable default features when selecting it.

`bundled` and `system` are mutually exclusive. CPU runtime support is limited to native Linux x86_64; Linux aarch64 and Apple ARM64 have model-free build/test surfaces. Accelerator features below are experimental build configuration.

System mode requires `VLLM_CPP_ROOT`, whose prefix must contain `include/vllm.h` plus a `lib` or `lib64` directory. `VLLM_CPP_LIB_DIR` can override the vllm library directory. Consumer builds validate that the selected system header exists but do not compare its layout. The maintainer integration test compiles its C probe at test runtime against that header, compares its C layouts with the generated Rust declarations, and the runtime test requires ABI version 17.

Upstream's normal CMake install provides `libvllm` and `vllm.h` but does not install the private `libblake3_vendored.a` target. A stock install therefore works directly with `system,dynamic-link`. System static mode requires callers to provision the matching `libblake3_vendored.a` separately and set `VLLM_CPP_BLAKE3_LIB_DIR`; when the variable is unset, the build script checks the selected vllm library directory for backward compatibility. It does not search arbitrary build trees.

`dynamic-link` does not copy or package the shared library. Tests and applications must install `libvllm` in a loader-visible location or configure `LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, an application-owned rpath, or another loader search path. This crate emits no machine-local rpath.

## Backend Features

Feature selection is build configuration, not runtime or hardware evidence. Accelerator features are bundled-only but do not enable `bundled`: normal default-feature commands may use `--features cuda`, while `--no-default-features` callers must use, for example, `--features bundled,cuda`. `cuda` conflicts with `vulkan`, and host `VLLM_CPP_SANITIZE` settings conflict with CUDA.

| Feature | Build contract | Native linking |
|---|---|---|
| `cuda` | Linux x86_64/aarch64; requires exact `VLLM_CPP_CUDA_ARCHITECTURES` | Static mode links CMake's exact `CUDA_CUDART` and `CUDA_cublasLt_LIBRARY` results; dynamic mode relies on `libvllm.so` dependencies |
| `cuda-cutlass` | Implies CUDA; requires canonical `VLLM_CPP_CUTLASS_DIR` with CUTLASS >=4.5.0; fetch is OFF; `103a`/`110` rejected | Header-only external input |
| `triton-aot` | Implies CUDA; embeds all six checked-in AOT trees; regeneration is OFF | Static mode also links CMake's exact `CUDA_cuda_driver_LIBRARY`; dynamic mode relies on `libvllm.so` |
| `vulkan` | Linux x86_64/aarch64; packaged Khronos headers and checked-in SPIR-V | No Vulkan SDK link; runtime loader uses `dlopen` |
| `metal` | Exact `aarch64-apple-darwin`; native MSL backend | Static mode links `c++`, then `Metal` and `Foundation` frameworks |
| `mlx` | Implies Metal; requires canonical external `MLX_ROOT` with `include/mlx/array.h`, `lib/libmlx.dylib`, and `lib/mlx.metallib` | Static mode adds `$MLX_ROOT/lib` before `dylib=mlx`; no rpath or packaged MLX payload |

CUDA architectures are exactly `80`, `86`, `87`, `89`, `90a`, `100a`, `103a`, `110`, `120a`, `121a`, or `120a;121a`. `triton-aot` accepts every one of those values, including the fat `120a;121a` build, and embeds the complete `sm_80`, `sm_86`, `sm_89`, `sm_90a`, `sm_100a`, and `sm_121a` vendored set. Runtime dispatch selects only an exact-SM cubin; architectures without an exact tree use the portable CUDA path rather than a neighboring cubin. Plain CUDA passes a nonexistent CUTLASS root to CMake so ambient source cannot silently change the build. Use separate `CARGO_TARGET_DIR` values for each backend/link combination.

Cargo builds explicitly disable HIP/ROCm, Tenstorrent, NCCL, native tests/examples, and the HTTP server. CUTLASS fetching and Triton regeneration remain OFF, so supported bundled builds never download native inputs. Marlin and FlashAttention remain explicitly enabled when their CUDA architecture and external CUTLASS prerequisites are satisfied.

Metal and MLX remain bundled-only and do not imply `bundled`; `--no-default-features` callers must explicitly select `bundled,metal` or `bundled,mlx`. Setting `MLX_ROOT` without `mlx` is rejected. MLX is never downloaded, copied, or packaged, and applications are responsible for making `libmlx.dylib` and its metallib deployment visible at runtime.

These features do not claim runtime support. Known native blockers remain: a CUDA teardown SIGSEGV after otherwise successful tests, a CUDA bf16 numerical tolerance failure, CUTLASS concurrent output differences, incomplete Vulkan attention/model runtime, and external MLX provider behavior without release-lane model/runtime validation. The hosted Metal lane checks compile/link only; the software Vulkan lane checks backend/ops only. CPU is the only supported runtime family.

## Generated Bindings

The bundled source is pinned to vllm.cpp `v0.0.2` at commit `7020de93652ca920424a10ac5255b34810dd2f24` and exposes C ABI version 17. Bindings are generated with bindgen 0.72.1 from `wrapper.h`, which includes `vllm.cpp/include/vllm.h`, and committed to `src/bindings.rs`. The exported stable C boundary is narrower than the broader native C++ implementation; these declarations do not promise access to undocumented internals. Maintainers use Just 1.40 or newer from the repository root:

```console
just bindings
just sys
just link-modes
```

The conformance gate verifies the generated output, C and C++ header compatibility, C/Rust layout, the exact 35-symbol export set, pure backend plans/cache parsing, Apple/MLX validation and link ordering, pinned CUDA architecture mappings, Triton AOT drift, and compile-time/runtime ABI 17. CI separately runs bundled static/dynamic and fixture-backed system static/dynamic CPU tests; dynamic tests set the required loader path. `build.rs` only performs consumer native build/link integration and does not compile or execute the layout probe. `tests/layout.rs` compiles and executes `tests/layout.c` with the bundled header or `VLLM_CPP_ROOT/include/vllm.h` at test runtime using the Rust standard library. Native Linux CPU is the supported runtime target; cross-compiling that integration test is unsupported.

The Rust crate is dual-licensed under MIT or Apache-2.0. The bundled vllm.cpp source retains its upstream Apache-2.0 license and notices.

vllm.cpp is an independent community project and is not affiliated with or endorsed by the vLLM project, the PyTorch Foundation, or the Linux Foundation.
