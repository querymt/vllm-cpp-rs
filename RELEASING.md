# Releasing

Releases are manual. Validation and dry-runs do not upload crates or create tags; those actions are separate, explicit release steps.

## Prepare

- Start from a clean checkout and recursive submodule. Confirm the gitlink, native HEAD, and peeled `v0.0.2` tag all resolve to `7020de93652ca920424a10ac5255b34810dd2f24`.
- Keep both Rust crates at `0.0.2`, `vllm-cpp` dependent on exactly `vllm-cpp-sys =0.0.2`, and native CMake at `0.0.2`.
- Preserve ABI 17 and all 35 stable C functions across bindings, headers, layouts, signatures, runtime checks, linking, and dynamic exports. ABI-10 system libraries are incompatible.
- Claim only tested support: native Linux x86_64 CPU is supported; accelerator backends remain experimental. Keep licenses/notices and Cargo's package boundaries accurate; packages must exclude models, fixtures, caches, build output, external CUTLASS, and local paths.

## Validate

Run the maintained gates with Just 1.40 or newer:

```console
just ci
RUSTUP_TOOLCHAIN=1.85.0 just msrv
git diff --check
```

`just ci` runs formatting, warnings-denied lint/docs, locked model-free workspace tests, binding/header/layout/ABI/symbol checks, backend integrity, four CPU link modes, native C API fixtures, ASan/UBSan/leak checks, package/downstream verification, and a publish dry-run. Workspace all-target tests include the build-support and backend-config suites. The separate MSRV gate requires Rust and Cargo exactly 1.85.0.

`just package-test` compares each archive with `cargo package --list`, checks metadata, native inputs, licenses and leaks, then runs offline extracted tests, four extracted sys link modes, and independent sys and safe downstream consumers. File counts, payload/compressed sizes, and SHA-256 values are diagnostics, not caps. Claim byte reproducibility only when two archive hashes actually match.

The native fixture gate clears `VT_ASYNC_SCHED` and `VT_ASYNC_RUNNER` and validates the default-scheduler result.

Nix is optional. Prepared-Qwen inference/sanitizers, native-only TSan, Rust MiniMax-H3 generation, Miri, other platforms, and accelerator runtime are optional or deferred. Report only evidence actually collected: model-free passes and successful compilation do not establish prepared-model inference or backend correctness.

## Publish

`just publish-dry-run` performs package verification followed by a workspace Cargo dry-run with `--no-verify` to avoid the registry dependency cycle. It does not upload crates or tag a release, and it is not the full registry-resolved safe-crate dry-run. Publish sys first, wait until it is available, then fully verify the safe crate against registry sys `0.0.2` before its upload:

```console
cargo publish -p vllm-cpp-sys --locked
cargo publish -p vllm-cpp --locked --dry-run
cargo publish -p vllm-cpp --locked
```

After explicit publication, verify registry metadata, docs.rs, licenses, and a clean downstream build before tagging the published commit. Published versions are immutable: never replace a version with different bytes. Check registry state before retrying a failed safe upload; yanking does not make a version reusable.
