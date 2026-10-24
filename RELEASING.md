# Releasing

Releases are manual. Review the candidate from a clean checkout before publishing;
validation and dry-runs do not upload crates or create tags.

## Prepare

- Check `git status --short` and `git submodule status --recursive`, including a
  clean native worktree. Confirm the gitlink, native HEAD, and peeled `v0.0.2` tag
  all name `7020de93652ca920424a10ac5255b34810dd2f24`:

  ```console
  git rev-parse HEAD:vllm-cpp-sys/vllm.cpp
  git -C vllm-cpp-sys/vllm.cpp rev-parse HEAD 'refs/tags/v0.0.2^{}'
  ```

- Confirm both Rust crates are `0.0.2`, with exactly `vllm-cpp-sys =0.0.2` in the
  safe crate, and native CMake reports `0.0.2`. Check manifests and lockfiles.
- Preserve ABI 17 and all 35 stable C functions: generated bindings, C11/C++20
  headers, layouts, signatures, runtime ABI, linking, and dynamic exports must
  agree. ABI-10 system libraries are incompatible.
- Keep `Unreleased` empty above the dated changelog entry. Describe only tested
  support: Linux x86_64 CPU is supported; accelerator backends remain experimental.
- Review packaged licenses/notices and `vllm-cpp-sys/THIRD_PARTY.md`, including
  imported license texts. Packages must not contain models, media fixtures, build
  output, caches, SDKs, external CUTLASS trees, internal records, or local paths.
  Cargo manifests define package inclusion and exclusion.

## Validate

Use the normal Rust/native build tools and Just 1.40 or newer:

```console
just ci
RUSTUP_TOOLCHAIN=1.85.0 just msrv
git diff --check
```

`just ci` runs formatting, warnings-denied lint/docs, model-free workspace tests,
bindings/header/layout/signature/ABI/export
checks, four workspace CPU link modes, native C API fixtures, ASan/UBSan/leak
checks, package verification, and a no-upload publish dry-run. The MSRV gate
separately requires Rust and Cargo exactly 1.85.0.

Run the package gate with a fresh target directory (the CI package run may serve
as this run). It compares archives with `cargo package --list`, checks metadata,
source inputs, licenses and leaks, then runs offline extracted tests, four
extracted sys link modes, and independent sys and safe downstream consumers.
File counts, payload/compressed sizes and SHA-256 hashes are diagnostics, not
caps. A second run is optional; claim byte reproducibility only if compared
archive hashes actually match.

The native fixture gate unsets `VT_ASYNC_SCHED` and `VT_ASYNC_RUNNER`. A default
failure remains a failure; alternate-scheduler diagnostics cannot satisfy it.
Record any failed gate rather than treating later individual passes as a CI pass.

Nix is optional: use `nix develop --no-write-lock-file -c just ci` and
`nix develop .#msrv --no-write-lock-file -c just msrv` instead if desired.
Prepared-Qwen inference/sanitizers, native-only TSan, Rust MiniMax-H3 generation,
Miri, other platforms, and accelerator runtime are optional or deferred lanes.
Report only what actually ran; model-free passes do not establish model inference.

## Publish

Check crates.io for the exact versions before uploading. An unavailable or
ambiguous registry state needs diagnosis, not a trial upload. Publish sys first,
wait for registry availability, then run the **full** safe dry-run against registry
sys `0.0.2` before uploading the safe crate:

```console
cargo publish -p vllm-cpp-sys --locked
cargo publish -p vllm-cpp --locked --dry-run
cargo publish -p vllm-cpp --locked
```

The pre-publication `just publish-dry-run` uses `--no-verify` to avoid the registry
cycle; it does not replace that full safe dry-run. After publication, verify
registry metadata, docs.rs, licenses, and a clean downstream build before tagging
the published commit and creating a release.

Accepted versions are immutable: never republish or replace them with different
bytes. If sys succeeds but safe fails, check registry state before retrying;
retry unchanged safe contents only for a transient failure, otherwise prepare a
new coordinated version. Stop on identity drift, failed gates, dirty release
state, or unexpected package content. Yanking does not erase a crate or make its
version reusable and is not an ordinary abort mechanism.
