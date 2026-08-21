# examples

these examples exercise the safe `vllm-cpp` api with fixed prompts and settings:

| example | behavior |
|---|---|
| [`complete`](complete.rs) | runs one blocking text completion |
| [`stream`](stream.rs) | prints one completion as token deltas arrive |
| [`concurrent`](concurrent.rs) | submits two asynchronous streaming requests and waits for both |
| [`chat`](chat.rs) | sends a fixed raw-json chat request; the optional `serde` feature is not required |
| [`structured`](structured.rs) | constrains one completion to the choice `red` or `blue` |

each example reads the first positional argument as a model directory and implements no additional options. a usable directory must contain the runtime files `model.safetensors`, `config.json`, `tokenizer.json`, and `tokenizer_config.json`; the pinned [test model fixture and layout](https://github.com/querymt/vllm-cpp-rs#test-model-and-sanitizers) is the known-good reference. model compatibility depends on the native engine, so an arbitrary model directory is not guaranteed to work.

## ordinary linux

nix and nixos are not required. for the default bundled cpu build, install:

- rust and cargo
- cmake 3.24 or newer
- ninja or another cmake build tool
- a c11 compiler, a c++20 compiler, a linker, and a c++ standard library
- git, so the pinned vllm.cpp submodule can be initialized

`just`, `jq`, bindgen, and libclang are maintainer tools and are not required to run these examples. from the workspace root, initialize the native source once:

```console
git submodule update --init --recursive
```

the commands below select ninja explicitly with `CMAKE_GENERATOR=Ninja`; merely installing ninja does not configure cmake to use it. if you choose another cmake generator and build tool, install them and set or otherwise configure `CMAKE_GENERATOR` accordingly.

the common command shape is:

```console
CMAKE_GENERATOR=Ninja \
  cargo run --locked --release -p vllm-cpp --features bundled --example EXAMPLE -- /path/to/model
```

`bundled` is the default feature; it is shown explicitly here to identify the cpu backend. run any of the five examples with:

```console
CMAKE_GENERATOR=Ninja \
  cargo run --locked --release -p vllm-cpp --features bundled --example complete -- /path/to/model
CMAKE_GENERATOR=Ninja \
  cargo run --locked --release -p vllm-cpp --features bundled --example stream -- /path/to/model
CMAKE_GENERATOR=Ninja \
  cargo run --locked --release -p vllm-cpp --features bundled --example concurrent -- /path/to/model
CMAKE_GENERATOR=Ninja \
  cargo run --locked --release -p vllm-cpp --features bundled --example chat -- /path/to/model
CMAKE_GENERATOR=Ninja \
  cargo run --locked --release -p vllm-cpp --features bundled --example structured -- /path/to/model
```

## optional nix shell

nix is optional and works on supported linux installations with nix; it does not require nixos. the default development shell supplies the cpu build dependencies. from the workspace root, run:

```console
CMAKE_GENERATOR=Ninja \
  nix develop -c cargo run --locked --release -p vllm-cpp --features bundled --example complete -- /path/to/model
```

replace `complete` with any other example name from the table.

## experimental cuda

plain `cuda` is the baseline accelerator feature. read the root [experimental backend build details](https://github.com/querymt/vllm-cpp-rs#experimental-backend-builds) before using it: cuda is a bundled linux experimental/build-only integration surface, and successful compilation does not guarantee model inference or backend correctness. a non-nix build needs a compatible cuda toolkit and driver installation in addition to the ordinary linux prerequisites.

set `VLLM_CPP_CUDA_ARCHITECTURES` to an architecture supported by this crate and the target gpu, and use a fresh `CARGO_TARGET_DIR` for the backend/link combination. for example, the tested RTX 5080 setup uses `120a`; do not use that value for unrelated hardware:

```console
arch=120a
CMAKE_GENERATOR=Ninja \
  VLLM_CPP_CUDA_ARCHITECTURES="$arch" \
  CARGO_TARGET_DIR="$PWD/target/cuda-examples" \
  cargo run --locked --release -p vllm-cpp --features cuda --example complete -- /path/to/model
```

with nix installed, the cuda shell supplies the pinned toolkit dependencies but still requires an explicit target architecture:

```console
arch=120a
CMAKE_GENERATOR=Ninja \
  VLLM_CPP_CUDA_ARCHITECTURES="$arch" \
  CARGO_TARGET_DIR="$PWD/target/cuda-examples-nix" \
  nix develop .#cuda -c cargo run --locked --release -p vllm-cpp --features cuda --example complete -- /path/to/model
```

other accelerator features have stricter limits:

- `cuda-cutlass` is an optional cuda variant; follow the [exact external cutlass prerequisites and known blockers](https://github.com/querymt/vllm-cpp-rs#experimental-backend-builds) before selecting it.
- `triton-aot` requires a target architecture with matching checked-in artifacts; `120a` is not supported by those artifacts.
- `vulkan` is currently for backend build/testing work. its model attention path is absent, so it cannot run these full-model examples.

## troubleshooting

- if the native source or cmake inputs are missing, rerun `git submodule update --init --recursive`.
- the first bundled build compiles the native c++ library and can take substantially longer than later runs.
- if model loading fails, verify the directory argument and its model, configuration, and tokenizer files, then confirm that vllm.cpp supports the model.
- for `dynamic-link` or `system` builds, follow the root [link mode and loader-path requirements](https://github.com/querymt/vllm-cpp-rs#link-modes); cargo does not deploy `libvllm.so` or configure its runtime search path.
- for accelerator configuration errors, use a fresh target directory and consult the root [experimental backend build details](https://github.com/querymt/vllm-cpp-rs#experimental-backend-builds).
