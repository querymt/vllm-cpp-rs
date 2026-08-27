# vllm-cpp

Safe Rust API for the stable [vllm.cpp](https://github.com/mudler/vllm.cpp) C boundary. The crate owns native resources, checks ABI compatibility before loading, and covers text completion/streaming/chat/requests, pre-tokenized completion, transcription, embeddings, video generation, and mux argument composition. Use `vllm-cpp-sys` only for the unsafe raw ABI.

## Quick use

```rust
use vllm_cpp::{Engine, SamplingParams};

let engine = Engine::load("/models/Qwen3-0.6B")?;
let params = SamplingParams::greedy().max_tokens(32);
let completion = engine.complete("The capital of France is", &params)?;
println!("{}", completion.text);
# Ok::<(), vllm_cpp::Error>(())
```

`Engine::load` accepts either a model directory or a standalone GGUF file understood by the pinned native engine. `HuggingFaceModel::safetensors` resolves and validates a complete loader snapshot, including unsharded or indexed weights; model-family compatibility remains a native vllm.cpp concern. See the packaged [examples guide](examples/README.md) for local and Hugging Face loading, blocking completion, streaming, JSON-Schema output, concurrent-request commands, and the Clap-based interactive chat CLI with retained history and supported sampling controls.

## Hugging Face models

`HuggingFaceModel` is an always-available synchronous resolver backed by the required `hf-hub` 0.5 dependency. It returns a local `PathBuf` that can be passed unchanged to `Engine::load`:

```rust,no_run
use vllm_cpp::{Engine, HuggingFaceModel};

let path = HuggingFaceModel::gguf("owner/repository", "model.gguf")
    // Omit this builder to follow the Hub's mutable `main` revision.
    .revision("0123456789abcdef0123456789abcdef01234567")
    .resolve()?;
let engine = Engine::load(path)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

`HuggingFaceModel::gguf(repo, filename)` and `HuggingFaceModel::safetensors(repo)` default to the Hub's mutable `main` revision. The `.revision(...)` builder accepts a branch, tag, or commit; immutable commit SHAs are recommended for reproducibility. The default cache honors `HF_HOME` through the normal Hugging Face layout and uses the cached login token when available. Builders can select a Hub cache directory, override the token, enable progress, or require cache-only offline resolution. Explicit tokens are redacted from resolver `Debug` output. The official endpoint is fixed; `HF_ENDPOINT` is not used.

GGUF mode retrieves one root-level lowercase `.gguf` file and rejects split sets. Safetensors mode first reads repository metadata for `main` or the explicit revision, pins downloads to its commit SHA, and retrieves only the root files required by this native loader: `config.json`, `tokenizer.json`, optional `tokenizer_config.json`, and either `model.safetensors` or the root index plus every indexed root shard. It returns the shared `snapshots/<sha>` directory and creates that revision's cache ref only after a complete successful retrieval. It does not download unrelated repository assets. Offline mode constructs no network API, reads the cached `main` ref by default, and distinguishes a missing cache revision from an incomplete cached snapshot.

Retrieval validates cache and snapshot completeness; it does not prove that the pinned native engine supports the repository's model architecture, tokenizer, quantization, or backend.

In the repository checkout, `just setup-test-model` explicitly resolves `Qwen/Qwen3-0.6B` at immutable revision `c1899de289a04d12100db370d81485cdf75e47ca`, reusing the standard Hugging Face cache and honoring normal `HF_HOME` and authentication. Ordinary tests and instrumentation commands never resolve or download this fixture. `VLLM_CPP_TEST_MODEL` remains an external contract: when supplied, it must point to a prepared model directory; model tests skip when it is unset.

## API and ownership

| Surface | Owner and contract |
|---|---|
| Text | `EngineBuilder` loads `Engine`; `Engine` is `Clone + Send + Sync` and shares one `Arc`-retained native handle. Completion, streaming, chat, structured output, logits processors, and `complete_tokens` are blocking; `submit` returns a non-blocking `Request`. |
| Tokens | `complete_tokens` borrows prompt IDs for the call and returns `TokenCompletion`. Output capacity limits reported IDs, not generation. `truncated` comes from native completion metadata; `include_completion = false` suppresses only the Rust metadata copy. |
| Transcription | `TranscriptionEngine` is non-cloneable and neither `Send` nor `Sync`; `transcribe(&mut self, TranscriptionInput)` blocks, borrows a WAV path or PCM slice, and returns Rust-owned optional text and token IDs. |
| Embeddings | `EmbeddingEngine` has the same conservative thread-local/exclusive contract. `embed(&mut self, ...)` blocks and returns a Rust-owned row-major `EmbeddingResult` preserving input order. |
| Video | `VideoEngineBuilder` loads a separate checkpoint set. Non-cloneable `VideoEngine` is neither `Send` nor `Sync`; `generate(&mut self, ...)` is blocking and exclusive and returns Rust-owned paths, dimensions, rates, counts, and mux argv. |
| Mux | `compose_video_mux_argv(&VideoMuxParams)` returns owned `VideoMuxArgv` argument boundaries. It performs no filesystem I/O and never locates or executes ffmpeg. |

`EngineBuilder` obtains native defaults first and overlays only explicit Rust settings. Unset values preserve the helper defaults, including `block_size=32`, `max_num_seqs=32`, and `gpu_memory_utilization=0.92`. `Device` uses `Auto=0`, `Cpu=1`, and `Cuda=2`; explicit CUDA never falls back. KV-memory precedence is `num_blocks > kv_cache_memory_bytes > gpu_memory_utilization` and its native profile/fallback path. Video uses independent `VideoDevice` numbering, `Cpu=0` and `Cuda=1`, with no automatic mode.

All native completion, chat, transcription, embedding, video, stream, argv, and diagnostic data is copied into Rust-owned values before native storage is released or reused. Blocking calls borrow their input storage only until return. Blocking callback panics are contained before C and resumed afterward. Custom logits processors are `Send + Sync`, may run on native worker threads, and report contained panic through `Error::LogitsProcessorPanicked`.

`Engine::submit` returns a `Request` that retains its engine, callback, and optional logits processor until native free/join completes. A request is `Send` but not `Sync`; dropping a live request cancels and joins it. Asynchronous callbacks are `Send + 'static` and run on a native delivery thread. Callback-thread wait/free is rejected or delegated to the cleanup reaper under the stable ABI lifecycle contract.

ABI 17 exposes no task query. Loading `Engine`, `TranscriptionEngine`, or `EmbeddingEngine` does not inspect or infer the checkpoint task; native wrong-task `Error::InvalidArgument` diagnostics remain authoritative. Video model format, partition, task, media, and capability checks are also native authority.

Video generation creates or truncates frame/audio artifacts, may leave stale files or partial output after failure, and has no cancellation, timeout, quota, rollback, or sandbox. Paths are trusted as supplied; Rust does not canonicalize, confine, reject symlinks, or clean outputs. On Unix, paths and mux arguments preserve raw bytes through `OsString`; non-Unix native conversion requires valid UTF-8. Mux arguments remain separate process arguments and must not be shell-joined. This crate does not execute ffmpeg, a server, or any other process.

`SchedulerPolicy::Priority` selects the native queue. Raw and serde chat JSON may carry `priority`; direct completion, streaming, and `Request` submission currently use priority zero and tie by arrival.

## Features and linking

| Feature | Purpose |
|---|---|
| `bundled` (default) | Build and statically link the pinned CPU native source |
| `system` | Link a caller-provided installation; use with `--no-default-features` |
| `dynamic-link` | Link `libvllm` dynamically in bundled or system mode |
| `serde` | Add `serde_json::Value` chat helpers; JSON parsing for Hub resolution is always present |
| `cuda` | Experimental bundled CUDA build configuration |
| `cuda-cutlass` | Experimental CUDA build with a caller-provided CUTLASS >=4.5.0 tree |
| `triton-aot` | Experimental CUDA build using checked-in Triton AOT artifacts |
| `vulkan` | Experimental bundled Vulkan build configuration |
| `metal` | Experimental native Metal build on Apple ARM64 |
| `mlx` | Experimental external MLX provider on Apple ARM64; implies `metal` |

Hugging Face resolution is not a Cargo feature: synchronous `hf-hub` support is a normal dependency in every build and does not add Tokio or another async runtime.

`bundled` and `system` conflict. CUDA and Vulkan conflict, and accelerator features are bundled-only but do not implicitly enable `bundled` for `--no-default-features` builds. Metal/MLX require exact `aarch64-apple-darwin`; MLX additionally requires an external `MLX_ROOT` with its headers, dylib, and metallib. The workspace [backend documentation](https://github.com/querymt/vllm-cpp-rs#experimental-backend-builds) records exact environment variables, supported build architectures, and current blockers.

## ABI and deployment

This crate uses the exact matching `vllm-cpp-sys =0.0.2`, whose bundled source is independently pinned to vllm.cpp tag `v0.0.2`, commit `7020de93652ca920424a10ac5255b34810dd2f24`. The checked-in bindings cover all 35 stable C functions at ABI 17. Loading checks exact runtime ABI equality before any versioned struct crosses FFI. `version()` is diagnostic; `abi_version()` is the compatibility authority. A system library must implement ABI 17 with matching layouts and signatures; ABI-10 libraries are incompatible.

Static bundled builds include the native archive in the application link. Dynamic bundled or system builds do not deploy `libvllm.so`/`libvllm.dylib`: install it and its backend/toolkit dependencies in a loader-visible location using `LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, rpath supplied by the application, or the system loader configuration. System mode uses `VLLM_CPP_ROOT`; `VLLM_CPP_LIB_DIR` can choose a nonstandard library directory. System static linking also requires the matching `libblake3_vendored.a` through `VLLM_CPP_BLAKE3_LIB_DIR` or the selected vllm library directory.

## Support boundary

The supported runtime tier is native Linux x86_64 CPU, covering bundled/system and static/dynamic link modes. Mandatory candidate evidence is model-free and includes native fixtures, sanitizers, exact link/export checks, packages, and downstream consumers. Prepared-Qwen inference/sanitizers, TSan, successful Rust MiniMax-H3 generation, Miri, Linux ARM64, Apple ARM64, Vulkan, CUDA/CUTLASS/Triton, Metal/MLX, and accelerator runtime are optional or deferred unless rerun against the exact candidate. Accelerator features remain experimental build/configuration surfaces; CPU is the only supported runtime family.

See the repository [changelog](https://github.com/querymt/vllm-cpp-rs/blob/main/CHANGELOG.md), [release process](https://github.com/querymt/vllm-cpp-rs/blob/main/RELEASING.md), and [root support details](https://github.com/querymt/vllm-cpp-rs#support) for the current release boundary.

The crate is dual-licensed under MIT or Apache-2.0.
