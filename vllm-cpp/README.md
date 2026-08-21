# vllm-cpp

Safe Rust API for the stable [vllm.cpp](https://github.com/mudler/vllm.cpp) C boundary. The crate owns native resources, checks ABI compatibility before model loading, and provides blocking completion/streaming/chat plus concurrent requests. Use `vllm-cpp-sys` directly only when an application needs the unsafe raw ABI.

## Quick use

```rust
use vllm_cpp::{Engine, SamplingParams};

let engine = Engine::load("/models/Qwen3-0.6B")?;
let params = SamplingParams::greedy().max_tokens(32);
let completion = engine.complete("The capital of France is", &params)?;
println!("{}", completion.text);
# Ok::<(), vllm_cpp::Error>(())
```

The model argument is a directory understood by the pinned native engine, not a single weights file. The known-good test layout contains `model.safetensors`, `config.json`, `tokenizer.json`, and `tokenizer_config.json`; model-family compatibility remains a native vllm.cpp concern. See the packaged [examples guide](examples/README.md) for blocking completion, streaming, chat, structured output, and concurrent-request commands.

## API and ownership

- `EngineBuilder` configures and loads a model. `Engine` is `Clone + Send + Sync`; clones share one reference-counted native engine.
- `SamplingParams` owns stop strings and structured constraints. Completion, chat, error, and stream text is copied into Rust-owned values before native storage is released or reused.
- Blocking `complete`, `complete_stream`, `chat_json`, and `chat_stream_json` calls keep borrowed callbacks alive only for the call. Callback panics are caught before crossing C and resumed after the native call returns.
- `Engine::submit` returns a `Request` before generation finishes. A request retains its engine and callback until native free/join completes, is `Send`, and is deliberately not `Sync`.
- Asynchronous callbacks run on a native delivery thread and must be `Send + 'static`. `wait` reports callback panics as `Error::CallbackPanicked`; waiting or freeing from that same callback thread is prohibited by ABI v10, so callback-thread drop transfers cleanup to a prestarted reaper.
- Dropping a live request cancels and joins it. `cancel` is idempotent, `wait` reports the request outcome, and `native_error` copies the request-owned diagnostic after completion into an owned Rust `String`; the native storage remains valid until the request is dropped or freed.

## Features and linking

| Feature | Purpose |
|---|---|
| `bundled` (default) | Build and statically link the pinned CPU native source |
| `system` | Link a caller-provided installation; use with `--no-default-features` |
| `dynamic-link` | Link `libvllm` dynamically in bundled or system mode |
| `serde` | Add `serde_json::Value` chat helpers |
| `cuda` | Experimental bundled CUDA build configuration |
| `cuda-cutlass` | Experimental CUDA build with a caller-provided CUTLASS >=4.5.0 tree |
| `triton-aot` | Experimental CUDA build using checked-in Triton AOT artifacts |
| `vulkan` | Experimental bundled Vulkan build configuration |
| `metal` | Experimental native Metal build on Apple ARM64 |
| `mlx` | Experimental external MLX provider on Apple ARM64; implies `metal` |

`bundled` and `system` conflict. CUDA and Vulkan conflict, and accelerator features are bundled-only but do not implicitly enable `bundled` for `--no-default-features` builds. Metal/MLX require exact `aarch64-apple-darwin`; MLX additionally requires an external `MLX_ROOT` with its headers, dylib, and metallib. The workspace [backend documentation](https://github.com/querymt/vllm-cpp-rs#experimental-backend-builds) records exact environment variables, supported build architectures, and current blockers.

## ABI and deployment

This crate is tied to the exact same `vllm-cpp-sys` crate version and the pinned vllm.cpp commit `34aedfbe8ed9779697905541a62e2160ccfd9c05`. Model loading requires exact C ABI version 10 before any versioned struct crosses FFI. A system library must implement the same ABI; the consumer build checks for its header, while maintainer conformance tests check layout and symbols.

Static bundled builds include the native archive in the application link. Dynamic bundled or system builds do not deploy `libvllm.so`/`libvllm.dylib`: install it and its backend/toolkit dependencies in a loader-visible location using `LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, rpath supplied by the application, or the system loader configuration. System mode uses `VLLM_CPP_ROOT`; `VLLM_CPP_LIB_DIR` can choose a nonstandard library directory. System static linking also requires the matching `libblake3_vendored.a` through `VLLM_CPP_BLAKE3_LIB_DIR` or the selected vllm library directory.

## Support boundary

The supported runtime tier is native Linux x86_64 CPU, covering bundled/system and static/dynamic link modes. Linux ARM64 and Apple ARM64 CPU have manual hosted jobs configured for model-free build/test coverage. CUDA, external CUTLASS, Triton AOT, Vulkan, Metal, and MLX are experimental build/configuration surfaces. The hosted Metal job checks compilation/linking only; Vulkan software-device gates check backend/ops, not attention or model inference; MLX remains external and has no release-lane model/runtime evidence. Known native blockers include CUDA teardown failure, CUDA bf16 numerical tolerance failure, CUTLASS concurrent-output differences, incomplete Vulkan attention/model runtime, and MLX's numerically distinct provider behavior. CPU remains the only supported runtime family.

See the repository [changelog](https://github.com/querymt/vllm-cpp-rs/blob/main/CHANGELOG.md), [release process](https://github.com/querymt/vllm-cpp-rs/blob/main/RELEASING.md), and [root support details](https://github.com/querymt/vllm-cpp-rs#support) for the current release boundary.

The crate is dual-licensed under MIT or Apache-2.0.
