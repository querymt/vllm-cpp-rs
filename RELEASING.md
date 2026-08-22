# Releasing

Releases are prepared and published manually. The repository does not tag, publish, or create a GitHub release automatically. A successful local candidate or dry-run is not a release; crates.io publication is an irreversible registry action.

## Prepare a candidate

1. Start from the reviewed release commit and verify the branch, `HEAD`, and intended remote identity.
2. Require a clean worktree and index, including initialized submodules:

   ```console
   test -z "$(git status --short --untracked-files=all)"
   git diff --quiet
   git diff --cached --quiet
   git submodule status --recursive
   test "$(git -C vllm-cpp-sys/vllm.cpp rev-parse HEAD)" = 34aedfbe8ed9779697905541a62e2160ccfd9c05
   test -z "$(git -C vllm-cpp-sys/vllm.cpp status --short --untracked-files=all)"
   ```

3. Confirm the release version in the workspace manifest, both normalized package manifests, `Cargo.lock`, and the pinned native `project(vllm_cpp VERSION ...)` declaration. Both crates and the native CMake project must use the same version, and `vllm-cpp` must depend on exactly that `vllm-cpp-sys` version. The CMake project declaration is the native release version authority; do not derive the crate version from `git describe` or the nearest native tag.
4. Confirm the native gitlink is `34aedfbe8ed9779697905541a62e2160ccfd9c05`, `VLLM_ABI_VERSION` is 10 in the pinned public C header and checked-in bindings, and generated bindings have no drift.
5. Move the relevant entries from `Unreleased` to a dated version section. Describe only validated support; preserve known backend/runtime blockers.
6. Audit dual-license metadata, crate license files, `NOTICE`, `THIRD_PARTY.md`, imported license texts, and the package inventory. Do not publish models, fixtures, build output, caches, SDKs, external CUTLASS trees, or repository-local paths.
7. Run the complete maintainer validation from the pinned development shell. At minimum run formatting, lint, model-free tests, docs, sys conformance, all CPU link modes, package extraction/downstream tests, and the exact MSRV gate.

## Inspect and dry-run

Build fresh archives; do not trust old files under `target/package`:

```console
cargo package -p vllm-cpp-sys --locked --list
cargo package -p vllm-cpp --locked --list
just package-test
just publish-dry-run
```

Inspect both sorted inventories and extracted normalized `Cargo.toml` files. Confirm the packages contain their READMEs, dual licenses, notices and provenance where applicable, source, tests, examples, and every required native/backend input. Confirm extracted builds and independent downstream consumers pass offline and that the safe consumer resolves the extracted sys crate rather than the workspace.

`just publish-dry-run` uses Cargo's workspace dry-run in sys-first order without uploading. The preceding package gate provides the full extracted/offline verification; the workspace command uses `--no-verify` to avoid a registry-resolution cycle before the exact sys version exists on crates.io. After sys is published, run `cargo publish -p vllm-cpp --locked --dry-run` and require its full verification to pass before the safe upload.

## Publish

Only an authorized maintainer should publish, from the exact reviewed commit with a clean worktree and index. Verify crates.io credentials and ownership, then publish one crate at a time:

```console
cargo publish -p vllm-cpp-sys --locked
# Wait until crates.io serves the exact sys version.
cargo publish -p vllm-cpp --locked --dry-run
cargo publish -p vllm-cpp --locked
```

The sys crate must be accepted and available from crates.io before publishing the safe crate because the safe archive declares an exact registry dependency. After both uploads, verify the registry metadata, package contents, docs.rs results, and a clean downstream build. Create the Git tag and release notes only for the exact published commit and version.

## Abort and recovery

- Before an upload succeeds, abort on any mismatch, validation failure, unexpected file, dirty state, changed lockfile, changed native pin/ABI, or inaccurate release note. Fix the issue in a separately reviewed commit and restart the checklist.
- After crates.io accepts a version, that version cannot be replaced or deleted. Never rebuild a different archive under the same version.
- If the sys crate publishes but the safe crate fails, stop and diagnose. Retry the unchanged safe version only when the failure is transient and the exact reviewed archive remains valid; otherwise prepare a new coordinated version.
- Yank a published version only when leaving it selectable would harm users. Yanking prevents new resolution but does not erase the crate, undo existing lockfiles, or make the version reusable. Record the reason publicly and publish a corrected new version.
- Never use `cargo yank` as an ordinary abort mechanism, and never publish merely to test credentials or packaging.
