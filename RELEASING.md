# Releasing

Releases are prepared and published manually. A candidate pass or dry-run does not authorize a Git tag, crates.io upload, merge, push, or GitHub release. Publication is an irreversible registry action and requires separate maintainer authorization.

## Prepare a candidate

1. Start from the exact independently reviewed release commit. Require a clean root worktree and index and a clean, detached native submodule:

   ```console
   test -z "$(git status --short --untracked-files=all)"
   git diff --quiet
   git diff --cached --quiet
   git submodule status --recursive
   test "$(git -C vllm-cpp-sys/vllm.cpp symbolic-ref -q HEAD || true)" = ""
   test -z "$(git -C vllm-cpp-sys/vllm.cpp status --short --untracked-files=all)"
   ```

2. Verify native identity directly, never with `git describe`:

   ```console
   native=vllm-cpp-sys/vllm.cpp
   test "$(git rev-parse HEAD:vllm-cpp-sys/vllm.cpp)" = 7020de93652ca920424a10ac5255b34810dd2f24
   test "$(git -C "$native" rev-parse HEAD)" = 7020de93652ca920424a10ac5255b34810dd2f24
   test "$(git -C "$native" rev-parse 'refs/tags/v0.0.2^{}')" = 7020de93652ca920424a10ac5255b34810dd2f24
   test "$(git -C "$native" rev-parse 'HEAD^{tree}')" = 28df226f0ef9924e67d563c3bef4712d0e628c5a
   ```

3. Use `cargo metadata --locked --no-deps --format-version 1` and normalized package manifests to require both Rust crates at `0.0.2` and the safe dependency requirement exactly `=0.0.2`. Confirm both local package records in `Cargo.lock`. Separately parse native `project(vllm_cpp VERSION 0.0.2 LANGUAGES CXX)` from `CMakeLists.txt`. Rust and native versions are independent release identities that happen to both be `0.0.2` here; equality is not a universal policy.
4. Require `VLLM_ABI_VERSION == 17` in the pinned header and generated bindings and exactly 35 stable C functions. Run binding-drift, C11/C++20 header, every C/Rust layout and signature, runtime ABI, all-function link, and exact dynamic-export checks. ABI-10 system libraries are incompatible.
5. Keep `Unreleased` empty above the dated release entry. Describe only validated support and preserve known limitations.
6. Audit root and crate dual-license metadata, `LICENSE-MIT`, `LICENSE-APACHE`, native `LICENSE`/`NOTICE`, `THIRD_PARTY.md`, and every imported license text against the exact package inventories. Reject models, media fixtures, build output, caches, SDKs, external CUTLASS trees, internal records, and repository-local paths.

## Validate

Run the mandatory Linux x86_64 CPU gates from the pinned shell:

```console
env -u VLLM_CPP_TEST_MODEL nix develop -c just ci
nix develop .#msrv -c just msrv
cargo check --locked --workspace --all-targets --features vllm-cpp/serde
RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --no-deps --features vllm-cpp/serde
git diff --check
```

`just ci` includes formatting, warnings-denied lint/docs, model-free workspace tests, generated bindings and ABI conformance, four CPU link modes, the native C API fixture, model-free ASan/UBSan/leak checks, package/extracted/downstream validation, and no-upload publish dry-run. Run the exact MSRV gate separately so stable-toolchain success cannot mask it.

Prepared-Qwen inference/sanitizers, native-only TSan, successful Rust MiniMax-H3 generation, Miri, Linux ARM64, Apple ARM64, Vulkan, CUDA/CUTLASS/Triton, Metal/MLX, and accelerator runtime are optional or deferred. Record one only when it ran against the exact candidate; configured workflows and older results are not candidate evidence.

## Inspect packages

Use two fresh, separate `CARGO_TARGET_DIR` values; do not trust existing `target/package` files:

```console
CARGO_NET_OFFLINE=true CARGO_TARGET_DIR=target/package-release-a just package-test
CARGO_NET_OFFLINE=true CARGO_TARGET_DIR=target/package-release-b just package-test
just publish-dry-run
```

For both runs, retain:

- sorted `cargo package --list` inventories and archive basenames;
- normalized `Cargo.toml` manifests, including exact `vllm-cpp-sys =0.0.2`;
- file counts and unpacked/compressed sizes, enforcing sys limits of 1,400 files, 40 MiB unpacked, and 6 MiB compressed and safe limits of 40 files, 512 KiB unpacked, and 128 KiB compressed;
- complete license/notice/provenance inventories and checks that current READMEs, source, tests, examples, and required native/backend inputs are present;
- SHA-256 for both archives from each run and successful extracted/offline builds, all four extracted sys link modes, and independent sys and safe downstream consumers.

Require identical sorted inventories and semantically identical normalized manifests between runs. Compare archive hashes and record both outcomes, but do not assume byte identity: Cargo-generated `.cargo_vcs_info.json`, archive metadata, or timestamps may differ. Claim reproducible bytes only when both hashes actually match and the comparison explains the metadata involved.

`just publish-dry-run` uses Cargo's sys-first workspace order with `--no-verify` and never uploads. It cannot provide the safe crate's full registry-resolution verification before exact sys `0.0.2` is available from crates.io. Check both crate-version slots are available before any future upload; do not reserve or publish them during candidate preparation.

## Publish

Only a separately authorized maintainer may publish from the exact reviewed commit with a clean root and detached submodule. Publish sys first:

```console
cargo publish -p vllm-cpp-sys --locked
# Wait until crates.io serves exact vllm-cpp-sys 0.0.2.
cargo publish -p vllm-cpp --locked --dry-run
cargo publish -p vllm-cpp --locked
```

The full safe dry-run must resolve registry sys `0.0.2` before the safe upload. After both uploads, verify registry metadata, archives, docs.rs, licenses, and a clean downstream build. Create a tag and GitHub release only after separate authorization and only for the exact published commit.

## Abort and recovery

- Before upload, abort on any mismatch, failed gate, unexpected file, dirty state, changed lockfile, native identity/ABI/export drift, inaccurate support statement, or unavailable version. Fix it in a separately reviewed commit and restart.
- Accepted crates.io versions cannot be replaced or deleted. Never rebuild different bytes under the same version.
- If sys publishes but safe fails, stop and diagnose. Retry unchanged safe bytes only for a transient failure; otherwise prepare a new coordinated version.
- Yank only when leaving a version selectable would harm users. Yanking does not erase the crate or make the version reusable.
- Never use `cargo yank` as an ordinary abort mechanism or publish merely to test credentials or packaging.
