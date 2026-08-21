# examples

five examples exercise the safe `vllm-cpp` api: four use fixed prompts and settings, and `chat` is an interactive command-line application. one additional maintainer utility prepares the pinned test fixture:

| example | behavior |
|---|---|
| [`complete`](complete.rs) | runs one blocking text completion |
| [`stream`](stream.rs) | prints one completion as token deltas arrive |
| [`concurrent`](concurrent.rs) | submits two asynchronous streaming requests and waits for both |
| [`chat`](chat.rs) | runs a Clap-based interactive chat with conversation history and streaming output |
| [`structured`](structured.rs) | extracts a fixed weather report under a JSON Schema |
| [`setup_test_model`](setup_test_model.rs) | resolves the pinned Qwen test fixture for `just setup-test-model`; not a general inference CLI |

The four fixed examples (`complete`, `stream`, `concurrent`, and `structured`) accept the same manual model-source forms:

```console
EXAMPLE <model-directory-or-gguf>
EXAMPLE local <model-directory-or-gguf>
EXAMPLE hf-gguf <repo> <filename> [--revision <revision>]
EXAMPLE hf-safetensors <repo> [--revision <revision>]
```

The bare path remains an alias for `local`. Hub forms enable download progress, use `main` when `--revision` is omitted, and reuse files already present in Hugging Face's normal cache. `Engine::load` receives the local or resolved `PathBuf` unchanged. It accepts either a model directory or a standalone GGUF file understood by the pinned native engine. A known-good Safetensors directory contains `model.safetensors`, `config.json`, `tokenizer.json`, and `tokenizer_config.json`; indexed models instead use `model.safetensors.index.json` and all referenced root shards. Model compatibility depends on the native engine, so an arbitrary directory, GGUF, or Hub repository is not guaranteed to work.

## hugging face loading

Omitting `--revision` follows the Hub's mutable default `main` branch. Use `.revision(...)` in the library or `--revision` in an example to select a branch, tag, or commit; an immutable commit SHA is recommended for reproducibility. The resolver is synchronous and always available because `hf-hub` is a normal dependency, not a Cargo feature.

By default the resolver uses Hugging Face's normal cache, honoring `HF_HOME`, and selects the cached token created by Hugging Face login. Library callers can use `.revision(...)`, `.cache_dir(...)`, `.token(...)`, `.progress(...)`, and `.offline(true)`. An explicit token overrides the selected cached token and is redacted from resolver `Debug`. Offline default resolution reads only the cached `main` ref; an explicit revision reads only that cached ref. Offline mode constructs no API. The official Hugging Face endpoint remains fixed; `HF_ENDPOINT` is not used.

GGUF mode retrieves one safe root-level filename ending in lowercase `.gguf`; split GGUF sets are unsupported. Safetensors mode queries metadata for `main` or the explicit revision, pins file retrieval to the returned commit SHA, and creates that revision's cache ref after complete success. It retrieves only `config.json`, `tokenizer.json`, optional `tokenizer_config.json`, and either root `model.safetensors` or root `model.safetensors.index.json` plus every unique root shard in `weight_map`. Unrelated repository assets are not downloaded. Successful retrieval establishes a complete sparse snapshot for the pinned loader, not model-family or backend compatibility.

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

The common command shape is:

```console
CMAKE_GENERATOR=Ninja \
  cargo run --locked --release -p vllm-cpp --features bundled --example EXAMPLE -- MODEL_SOURCE
```

`bundled` is the default feature; it is shown explicitly here to identify the CPU backend. These commands demonstrate all five examples and the shared source syntax:

```console
CMAKE_GENERATOR=Ninja cargo run --locked --release -p vllm-cpp --features bundled --example complete -- /path/to/model
CMAKE_GENERATOR=Ninja cargo run --locked --release -p vllm-cpp --features bundled --example stream -- local /path/to/model
CMAKE_GENERATOR=Ninja cargo run --locked --release -p vllm-cpp --features bundled --example concurrent -- hf-gguf owner/repository model.gguf
CMAKE_GENERATOR=Ninja cargo run --locked --release -p vllm-cpp --features bundled --example chat -- --system "Answer concisely." hf-safetensors owner/repository
CMAKE_GENERATOR=Ninja cargo run --locked --release -p vllm-cpp --features bundled --example structured -- hf-safetensors owner/repository --revision release
```

Replace the fixed examples' arguments after `--` with another model-source form from the syntax block. The chat command uses the same source names through Clap; its exact syntax and interactive options follow. No example requires a revision.

## interactive chat CLI

`chat` accepts global chat options before or after the model subcommand and uses one of these model forms:

```console
chat [OPTIONS] <model-directory-or-gguf>
chat [OPTIONS] local <model-directory-or-gguf> [OPTIONS]
chat [OPTIONS] hf-gguf <repo> <filename> [--revision <revision>] [OPTIONS]
chat [OPTIONS] hf-safetensors <repo> [--revision <revision>] [OPTIONS]
```

The bare path remains a local alias. The Hub subcommands reuse the shared resolver: progress is enabled, cached files are reused, omitted revisions follow mutable `main`, and `--revision` can select a branch, tag, or commit. These complete commands show local and Hub ordering:

```console
CMAKE_GENERATOR=Ninja cargo run --locked --release -p vllm-cpp --example chat -- \
  --system "Answer concisely." --prompt "Hello" local /path/to/model
CMAKE_GENERATOR=Ninja cargo run --locked --release -p vllm-cpp --example chat -- \
  hf-gguf owner/repository model.gguf --revision release --temperature 0.5
CMAKE_GENERATOR=Ninja cargo run --locked --release -p vllm-cpp --example chat -- \
  --file prompt.txt hf-safetensors owner/repository --no-stream
```

`--prompt/-p <text>` and `--file/-f <path>` are mutually exclusive optional first user messages; prompt files must be UTF-8. `--system <text>` adds a retained system message. Generation options are `--max-tokens` (default `256`, maximum `2147483647`), `--temperature` (default `0.7`), `--top-p` (default `1`), `--top-k` (default `0`), `--min-p` (default `0`), and optional `--seed`. Responses stream by default; `--no-stream` selects blocking `Engine::chat_json` output. The CLI maintains the complete user/assistant history, submits it on each turn, and prints only assistant content rather than raw response JSON. Native role, reasoning, tool-call, and finish metadata is ignored; a valid response with no content is stored as an empty assistant message.

At `user>` enter `/clear` to retain the system message while clearing other history, or `/quit`/`/exit` to stop. EOF also exits cleanly. A per-turn request or response error is reported as `chat: <error>`, the attempted turn is removed from history, and the prompt continues; terminal input/output errors and model startup failures still exit. This high-level example intentionally exposes only controls supported by the existing chat request and engine APIs; it does not add low-level model, device, context, batch, token, or timing controls from other runtimes.

## optional nix shell

nix is optional and works on supported linux installations with nix; it does not require nixos. the default development shell supplies the cpu build dependencies. from the workspace root, run:

```console
CMAKE_GENERATOR=Ninja \
  nix develop -c cargo run --locked --release -p vllm-cpp --features bundled --example complete -- /path/to/model
```

replace `complete` with another example and its arguments from the table.

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
- if local model loading fails, verify the directory or standalone GGUF argument and required files, then confirm that vllm.cpp supports the model.
- if Hub resolution fails offline, verify that the exact requested revision has a cache ref and a complete single snapshot; offline mode never contacts the network.
- successful Hub resolution does not establish architecture, tokenizer, quantization, or backend support in the pinned native engine.
- for `dynamic-link` or `system` builds, follow the root [link mode and loader-path requirements](https://github.com/querymt/vllm-cpp-rs#link-modes); cargo does not deploy `libvllm.so` or configure its runtime search path.
- for accelerator configuration errors, use a fresh target directory and consult the root [experimental backend build details](https://github.com/querymt/vllm-cpp-rs#experimental-backend-builds).
