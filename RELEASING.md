# Releasing

Releases are prepared and published manually. A candidate pass or dry-run does not authorize a Git tag, crates.io upload, merge, push, or GitHub release. Publication is an irreversible registry action and requires separate maintainer authorization.

## Prepare a candidate

1. Obtain `VLLM_CPP_RELEASE_COMMIT` from the independent approval of the exact release commit. Never derive the expected value from the current checkout. Require a lowercase, full 40-character SHA, verify that it names a commit object, and compare it with the root checkout before any candidate gate:

   ```bash
   set -euo pipefail
   : "${VLLM_CPP_RELEASE_COMMIT:?set this to the independently approved release commit}"
   if [[ ! $VLLM_CPP_RELEASE_COMMIT =~ ^[0-9a-f]{40}$ ]]; then
     echo 'VLLM_CPP_RELEASE_COMMIT must be exactly 40 lowercase hexadecimal characters' >&2
     exit 1
   fi
   git cat-file -e "$VLLM_CPP_RELEASE_COMMIT^{commit}"
   actual_release_commit=$(git rev-parse HEAD)
   if [[ $actual_release_commit != "$VLLM_CPP_RELEASE_COMMIT" ]]; then
     printf 'reviewed release commit mismatch: expected %s, found %s\n' \
       "$VLLM_CPP_RELEASE_COMMIT" "$actual_release_commit" >&2
     exit 1
   fi
   ```

   Preserve the externally supplied value and rerun this guard immediately before each separately authorized upload.

2. Require a clean root worktree and index and a clean, detached native submodule:

   ```console
   test -z "$(git status --short --untracked-files=all)"
   git diff --quiet
   git diff --cached --quiet
   git submodule status --recursive
   test "$(git -C vllm-cpp-sys/vllm.cpp symbolic-ref -q HEAD || true)" = ""
   test -z "$(git -C vllm-cpp-sys/vllm.cpp status --short --untracked-files=all)"
   ```

3. Verify native identity directly, never with `git describe`:

   ```console
   native=vllm-cpp-sys/vllm.cpp
   test "$(git rev-parse HEAD:vllm-cpp-sys/vllm.cpp)" = 7020de93652ca920424a10ac5255b34810dd2f24
   test "$(git -C "$native" rev-parse HEAD)" = 7020de93652ca920424a10ac5255b34810dd2f24
   test "$(git -C "$native" rev-parse 'refs/tags/v0.0.2^{}')" = 7020de93652ca920424a10ac5255b34810dd2f24
   test "$(git -C "$native" rev-parse 'HEAD^{tree}')" = 28df226f0ef9924e67d563c3bef4712d0e628c5a
   ```

4. Use `cargo metadata --locked --no-deps --format-version 1` and normalized package manifests to require both Rust crates at `0.0.2` and the safe dependency requirement exactly `=0.0.2`. Confirm both local package records in `Cargo.lock`. Separately parse native `project(vllm_cpp VERSION 0.0.2 LANGUAGES CXX)` from `CMakeLists.txt`. Rust and native versions are independent release identities that happen to both be `0.0.2` here; equality is not a universal policy.
5. Require `VLLM_ABI_VERSION == 17` in the pinned header and generated bindings and exactly 35 stable C functions. Run binding-drift, C11/C++20 header, every C/Rust layout and signature, runtime ABI, all-function link, and exact dynamic-export checks. ABI-10 system libraries are incompatible.
6. Keep `Unreleased` empty above the dated release entry. Describe only validated support and preserve known limitations.
7. Audit root and crate dual-license metadata, `LICENSE-MIT`, `LICENSE-APACHE`, native `LICENSE`/`NOTICE`, `THIRD_PARTY.md`, and every imported license text against the exact package inventories. Reject models, media fixtures, build output, caches, SDKs, external CUTLASS trees, internal records, and repository-local paths.

## Validate

Run the mandatory Linux x86_64 CPU gates directly with the native Cargo and Just workflows:

```console
env -u VLLM_CPP_TEST_MODEL just ci
RUSTUP_TOOLCHAIN=1.85.0 just msrv
# Equivalent exact-toolchain invocation: rustup run 1.85.0 just msrv
cargo check --locked --workspace --all-targets --features vllm-cpp/serde
RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --no-deps --features vllm-cpp/serde
git diff --check
```

`just ci` includes formatting, warnings-denied lint/docs, model-free workspace tests, generated bindings and ABI conformance, four CPU link modes, the native C API fixture, model-free ASan/UBSan/leak checks, package/extracted/downstream validation, and no-upload publish dry-run. Run the exact MSRV gate separately so stable-toolchain success cannot mask it.

Nix support remains an optional convenience, not a prerequisite. Maintainers who choose it may run `nix develop -c env -u VLLM_CPP_TEST_MODEL just ci` and `nix develop .#msrv -c just msrv`; `nix flake check --no-build` is an additional Nix-specific evaluation check, not an ordinary release gate.

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

## Inspect registry state

Use the exact-version crates.io API read-only. This helper classifies only HTTP 200 with matching crate/version JSON as `accepted` and only HTTP 404 as `absent`; network errors, redirects that do not finish in either status, other HTTP statuses, and malformed or mismatched JSON are `ambiguous`. Stop on `ambiguous`.

```bash
set -euo pipefail
command -v curl >/dev/null
command -v jq >/dev/null
registry_tmp=$(mktemp -d)
trap 'rm -rf "$registry_tmp"' EXIT HUP INT TERM

registry_state() {
  local crate=$1
  local version=$2
  local body status
  if [[ ! $crate =~ ^[a-z0-9][a-z0-9_-]*$ || ! $version =~ ^[0-9A-Za-z.+-]+$ ]]; then
    printf '%s\n' ambiguous
    return
  fi
  if ! body=$(mktemp "$registry_tmp/response.XXXXXX"); then
    printf '%s\n' ambiguous
    return
  fi
  if ! status=$(curl --silent --show-error --location \
      --connect-timeout 10 --max-time 30 --retry 0 \
      --output "$body" --write-out '%{http_code}' -- \
      "https://crates.io/api/v1/crates/$crate/$version"); then
    printf '%s\n' ambiguous
    return
  fi
  case $status in
    200)
      if jq -e --arg crate "$crate" --arg version "$version" \
          '.version.crate == $crate and .version.num == $version' \
          "$body" >/dev/null 2>&1; then
        printf '%s\n' accepted
      else
        printf '%s\n' ambiguous
      fi
      ;;
    404) printf '%s\n' absent ;;
    *) printf '%s\n' ambiguous ;;
  esac
}

sys_state=$(registry_state vllm-cpp-sys 0.0.2)
safe_state=$(registry_state vllm-cpp 0.0.2)
rm -rf "$registry_tmp"
trap - EXIT HUP INT TERM
printf 'vllm-cpp-sys 0.0.2: %s\nvllm-cpp 0.0.2: %s\n' \
  "$sys_state" "$safe_state"
```

The helper performs no registry mutation. Preserve its output with the release evidence. Before an initial sys upload, both states must be `absent`. Before any safe upload or retry, sys must be `accepted` and safe must be `absent`. If safe is already `accepted`, never republish it. Any other combination requires stopping for diagnosis.

## Publish

Only a separately authorized maintainer may publish from the exact reviewed commit with a clean root and detached submodule. Rerun the reviewed-root guard immediately before each upload. Publish sys first:

```console
cargo publish -p vllm-cpp-sys --locked
# Wait until the exact-version helper reports sys accepted and safe absent.
cargo publish -p vllm-cpp --locked --dry-run
cargo publish -p vllm-cpp --locked
```

The full safe dry-run must resolve registry sys `0.0.2` before the safe upload. After both uploads, verify registry metadata, archives, docs.rs, licenses, and a clean downstream build. Create a tag and GitHub release only after separate authorization and only for the exact published commit.

### Retry or verify the safe crate

A safe upload retry is allowed only for the same independently approved archive bytes. Obtain the approved hash externally; never copy a dirty-candidate hash into this document or derive the expected value from a rebuilt archive.

```bash
set -euo pipefail
: "${VLLM_CPP_SAFE_ARCHIVE_SHA256:?set this to the approved safe archive SHA-256}"
if [[ ! $VLLM_CPP_SAFE_ARCHIVE_SHA256 =~ ^[0-9a-f]{64}$ ]]; then
  echo 'VLLM_CPP_SAFE_ARCHIVE_SHA256 must be exactly 64 lowercase hexadecimal characters' >&2
  exit 1
fi
safe_archive=target/package/vllm-cpp-0.0.2.crate
actual_safe_sha256=$(sha256sum "$safe_archive" | awk '{print $1}')
if [[ $actual_safe_sha256 != "$VLLM_CPP_SAFE_ARCHIVE_SHA256" ]]; then
  echo 'safe archive differs from the approved bytes; prepare a new coordinated version' >&2
  exit 1
fi
```

Rerun the registry helper after this check. Retry only when `sys_state=accepted` and `safe_state=absent`, and only after separate upload authorization. `ambiguous` means stop. If `safe_state=accepted`, do not upload; instead verify the accepted archive against the same approved hash before post-publication checks:

```bash
set -euo pipefail
accepted_safe=$(mktemp)
trap 'rm -f "$accepted_safe"' EXIT HUP INT TERM
curl --fail --silent --show-error --location --connect-timeout 10 --max-time 60 \
  --retry 0 --output "$accepted_safe" -- \
  'https://crates.io/api/v1/crates/vllm-cpp/0.0.2/download'
printf '%s  %s\n' "$VLLM_CPP_SAFE_ARCHIVE_SHA256" "$accepted_safe" \
  | sha256sum --check --strict
```

A registry archive mismatch or any need to change the safe bytes requires a new coordinated version; never reuse `0.0.2` for different bytes.

## Abort and recovery

- Before upload, abort on any mismatch, failed gate, unexpected file, dirty state, changed lockfile, native identity/ABI/export drift, inaccurate support statement, or unavailable version. Fix it in a separately reviewed commit and restart.
- Accepted crates.io versions cannot be replaced or deleted. Never rebuild different bytes under the same version.
- If sys publishes but safe fails, stop and diagnose. Retry unchanged safe bytes only for a transient failure; otherwise prepare a new coordinated version.
- Yank only when leaving a version selectable would harm users. Yanking does not erase the crate or make the version reusable.
- Never use `cargo yank` as an ordinary abort mechanism or publish merely to test credentials or packaging.
