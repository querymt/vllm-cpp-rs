set shell := ["bash", "-euo", "pipefail", "-c"]

root := justfile_directory()
bindings_file := root + "/vllm-cpp-sys/src/bindings.rs"
model_revision := "c1899de289a04d12100db370d81485cdf75e47ca"

# Maintainer workflows require Just 1.40 or newer.

export CMAKE_GENERATOR := env_var_or_default("CMAKE_GENERATOR", "Ninja")
export CMAKE_BUILD_PARALLEL_LEVEL := env_var_or_default("CMAKE_BUILD_PARALLEL_LEVEL", "2")

# List available maintainer workflows.
default:
    @just --justfile {{ quote(root + "/Justfile") }} --list

# Regenerate the checked-in Rust bindings (bindgen 0.72.1 required).
bindings output=bindings_file:
    just --justfile '{{ root }}/Justfile' _bindings {{ quote(output) }}

[private]
_bindings output:
    #!/usr/bin/env bash
    set -euo pipefail
    output={{ quote(output) }}
    version=$(bindgen --version)
    if [[ $version != 'bindgen 0.72.1' ]]; then
      echo "expected bindgen 0.72.1, found: $version" >&2
      exit 1
    fi
    mkdir -p "$(dirname "$output")"
    output_dir=$(cd "$(dirname "$output")" && pwd -P)
    output="$output_dir/$(basename "$output")"
    cd {{ quote(root + "/vllm-cpp-sys") }}
    bindgen wrapper.h \
      --allowlist-function '^vllm_.*' \
      --allowlist-type '^vllm_.*' \
      --allowlist-var '^VLLM_.*' \
      --no-doc-comments \
      --no-layout-tests \
      --formatter rustfmt \
      --rust-target 1.85 \
      --output "$output" \
      -- \
      -std=c11 \
      -I.

# Check that the committed bindings match bindgen 0.72.1 output.
bindings-check:
    #!/usr/bin/env bash
    set -euo pipefail
    temp=$(mktemp)
    trap 'rm -f "$temp"' EXIT
    just --justfile {{ quote(root + "/Justfile") }} _bindings "$temp"
    cmp "$temp" {{ quote(bindings_file) }}

# Check that the wrapper is valid C11 and C++20.
header-check:
    #!/usr/bin/env bash
    set -euo pipefail
    "${CC:-cc}" -std=c11 -fsyntax-only -x c -I'{{ root }}/vllm-cpp-sys' \
      '{{ root }}/vllm-cpp-sys/wrapper.h'
    "${CXX:-c++}" -std=c++20 -fsyntax-only -x c++ -I'{{ root }}/vllm-cpp-sys' \
      '{{ root }}/vllm-cpp-sys/wrapper.h'

# Run focused tests for private build-script artifact selection helpers.
build-support-test:
    #!/usr/bin/env bash
    set -euo pipefail
    temp=$(mktemp -d)
    trap 'rm -rf "$temp"' EXIT
    rustc --edition=2021 --test -D warnings \
      {{ quote(root + "/vllm-cpp-sys/tests/build_support.rs") }} \
      -o "$temp/build-support-tests"
    "$temp/build-support-tests"

# Test the pure Linux backend build planner without configuring CMake.
backend-config:
    #!/usr/bin/env bash
    set -euo pipefail
    temp=$(mktemp -d)
    trap 'rm -rf "$temp"' EXIT
    rustc --edition=2021 --test -D warnings \
      {{ quote(root + "/vllm-cpp-sys/tests/build_config.rs") }} \
      -o "$temp/build-config-tests"
    "$temp/build-config-tests"

# Verify pinned CUDA architecture mappings and vendored Triton AOT inputs.
backend-integrity:
    cd {{ quote(root) }} && cmake -P vllm-cpp-sys/vllm.cpp/cmake/CudaArchFeaturesTest.cmake
    cd {{ quote(root) }} && bash vllm-cpp-sys/vllm.cpp/scripts/check-triton-aot-drift.sh

# Run the focused C/Rust layout conformance test.
layout-test:
    CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-{{ root }}/target/layout-test}" cargo test --locked -p vllm-cpp-sys --release --test layout

# Run generated-binding, header, layout, and backend configuration conformance checks.
sys: bindings-check header-check build-support-test backend-config backend-integrity layout-test

# Test all Linux CPU link modes and the exact shared-library exports.
link-modes:
    #!/usr/bin/env bash
    set -euo pipefail
    repo_root={{ quote(root) }}
    target_base=${CARGO_TARGET_DIR:-"$repo_root/target/link-modes"}
    mkdir -p "$target_base"
    target_base=$(cd "$target_base" && pwd -P)
    recipe_base="$target_base/vllm-cpp-sys-link-modes"
    mkdir -p "$recipe_base"
    recipe_base=$(cd "$recipe_base" && pwd -P)
    work=$(mktemp -d -- "$recipe_base/run.XXXXXXXXXX")
    cleanup() {
      case $work in
        "$recipe_base"/run.*) rm -rf -- "$work" ;;
        *) echo "refusing to remove unexpected link-modes work directory: $work" >&2 ;;
      esac
    }
    trap cleanup EXIT
    cd "$repo_root"
    export CARGO_TERM_COLOR=never

    find_one() {
      local root=$1
      local path_pattern=$2
      local matches=()
      mapfile -t matches < <(
        find -L "$root" -type f -path "$path_pattern" -print | LC_ALL=C sort
      )
      if ((${#matches[@]} != 1)); then
        printf 'expected exactly one %s below %s, found %d\n' \
          "$path_pattern" "$root" "${#matches[@]}" >&2
        printf '%s\n' "${matches[@]}" >&2
        exit 1
      fi
      printf '%s\n' "${matches[0]}"
    }

    find_installed_library() {
      local root=$1
      local filename=$2
      local matches=()
      mapfile -t matches < <(
        find -L "$root" \
          \( -path "*/out/lib/$filename" -o -path "*/out/lib64/$filename" \) \
          -type f -print | LC_ALL=C sort
      )
      if ((${#matches[@]} != 1)); then
        printf 'expected exactly one installed %s below %s, found %d\n' \
          "$filename" "$root" "${#matches[@]}" >&2
        printf '%s\n' "${matches[@]}" >&2
        exit 1
      fi
      printf '%s\n' "${matches[0]}"
    }

    check_exports() {
      local library=$1
      local actual="$work/exports.actual"
      local expected="$work/exports.expected"
      nm -D --defined-only "$library" \
        | awk '$2 != "A" { name=$3; sub(/@@.*/, "", name); print name }' \
        | LC_ALL=C sort > "$actual"
      cat > "$expected" <<'EOF'
    vllm_abi_version
    vllm_chat
    vllm_chat_stream
    vllm_complete
    vllm_complete_stream
    vllm_completion_free
    vllm_engine_free
    vllm_engine_load
    vllm_last_error
    vllm_model_params_default
    vllm_request_cancel
    vllm_request_done
    vllm_request_error
    vllm_request_free
    vllm_request_submit
    vllm_request_wait
    vllm_sampling_params_default
    vllm_string_free
    vllm_version
    EOF
      diff -u "$expected" "$actual"
    }

    bundled_static_target="$work/bundled-static"
    echo '==> bundled static'
    CARGO_TARGET_DIR="$bundled_static_target" \
      cargo test --locked -p vllm-cpp-sys --release --tests

    bundled_dynamic_target="$work/bundled-dynamic"
    echo '==> bundled dynamic build'
    CARGO_TARGET_DIR="$bundled_dynamic_target" \
      cargo build --locked -p vllm-cpp-sys --release --features dynamic-link
    bundled_dynamic_lib=$(find_installed_library \
      "$bundled_dynamic_target/release/build" libvllm.so)
    check_exports "$bundled_dynamic_lib"
    bundled_dynamic_lib_dir=$(dirname "$bundled_dynamic_lib")
    echo '==> bundled dynamic test'
    LD_LIBRARY_PATH="$bundled_dynamic_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      CARGO_TARGET_DIR="$bundled_dynamic_target" \
      cargo test --locked -p vllm-cpp-sys --release --tests --features dynamic-link

    prefix="$work/system-prefix"
    mkdir -p "$prefix/include" "$prefix/lib" "$prefix/lib64" "$prefix/blake3-lib"
    cp vllm-cpp-sys/vllm.cpp/include/vllm.h "$prefix/include/"

    bundled_static_lib=$(find_installed_library \
      "$bundled_static_target/release/build" libvllm.a)
    bundled_static_lib_dir=$(dirname "$bundled_static_lib")
    cp "$bundled_static_lib" "$prefix/lib/"
    cp -a "$bundled_static_lib_dir"/libvllm.so* "$prefix/lib/"
    blake3_lib=$(find_one \
      "$bundled_static_target/release/build" '*/out/build/libblake3_vendored.a')
    cp "$blake3_lib" "$prefix/blake3-lib/"

    system_static_target="$work/system-static"
    system_static_log="$work/system-static.log"
    echo '==> system static'
    VLLM_CPP_ROOT="$prefix" \
      VLLM_CPP_BLAKE3_LIB_DIR="$prefix/blake3-lib" \
      CARGO_TARGET_DIR="$system_static_target" \
      cargo test --locked -vv -p vllm-cpp-sys --release --tests \
        --no-default-features --features system 2>&1 | tee "$system_static_log"
    grep -Fq "cargo:rerun-if-changed=$prefix/lib/libvllm.a" "$system_static_log"
    grep -Fq \
      "cargo:rerun-if-changed=$prefix/blake3-lib/libblake3_vendored.a" \
      "$system_static_log"
    blake3_search_line=$(grep -nF -m1 \
      "cargo:rustc-link-search=native=$prefix/blake3-lib" "$system_static_log" \
      | cut -d: -f1 || true)
    vllm_search_line=$(grep -nF -m1 \
      "cargo:rustc-link-search=native=$prefix/lib" "$system_static_log" \
      | cut -d: -f1 || true)
    if [[ -z $blake3_search_line || -z $vllm_search_line \
      || $blake3_search_line -ge $vllm_search_line ]]; then
      echo 'system static build did not select lib over empty lib64 or preserve BLAKE3 link order' >&2
      exit 1
    fi

    echo '==> system static watched-archive rerun'
    replacement="$prefix/lib/.libvllm.a.replacement"
    cp "$prefix/lib/libvllm.a" "$replacement"
    touch "$replacement"
    mv "$replacement" "$prefix/lib/libvllm.a"
    system_static_rerun_log="$work/system-static-rerun.log"
    VLLM_CPP_ROOT="$prefix" \
      VLLM_CPP_BLAKE3_LIB_DIR="$prefix/blake3-lib" \
      CARGO_TARGET_DIR="$system_static_target" \
      cargo test --locked -vv -p vllm-cpp-sys --release --tests \
        --no-default-features --features system 2>&1 | tee "$system_static_rerun_log"
    grep -Fq 'Dirty vllm-cpp-sys' "$system_static_rerun_log"
    grep -Fq 'libvllm.a' "$system_static_rerun_log"
    grep -Fq "$prefix/lib/libvllm.a" "$system_static_rerun_log"
    grep -Fq "cargo:rerun-if-changed=$prefix/lib/libvllm.a" \
      "$system_static_rerun_log"

    system_override_static_target="$work/system-override-static"
    system_override_static_log="$work/system-override-static.log"
    echo '==> system static invalid override'
    if VLLM_CPP_ROOT="$prefix" \
      VLLM_CPP_LIB_DIR="$prefix/lib64" \
      VLLM_CPP_BLAKE3_LIB_DIR="$prefix/blake3-lib" \
      CARGO_TARGET_DIR="$system_override_static_target" \
      cargo test --locked -vv -p vllm-cpp-sys --release --tests \
        --no-default-features --features system 2>&1 \
      | tee "$system_override_static_log"; then
      echo 'expected system static VLLM_CPP_LIB_DIR override to reject the empty lib64 directory' >&2
      exit 1
    fi
    grep -Fq \
      "expected VLLM_CPP_LIB_DIR override artifact libvllm.a at $prefix/lib64/libvllm.a" \
      "$system_override_static_log"

    system_sanitize_target="$work/system-sanitize"
    system_sanitize_log="$work/system-sanitize.log"
    echo '==> system sanitizer rejection'
    if VLLM_CPP_ROOT="$prefix" \
      VLLM_CPP_BLAKE3_LIB_DIR="$prefix/blake3-lib" \
      VLLM_CPP_SANITIZE=address \
      CARGO_TARGET_DIR="$system_sanitize_target" \
      cargo test --locked -vv -p vllm-cpp-sys --release --tests \
        --no-default-features --features system 2>&1 \
      | tee "$system_sanitize_log"; then
      echo 'expected system mode to reject VLLM_CPP_SANITIZE' >&2
      exit 1
    fi
    grep -Fq \
      'VLLM_CPP_SANITIZE is supported only for bundled builds' \
      "$system_sanitize_log"

    system_dynamic_target="$work/system-dynamic"
    system_dynamic_log="$work/system-dynamic.log"
    echo '==> system dynamic'
    LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      VLLM_CPP_ROOT="$prefix" \
      CARGO_TARGET_DIR="$system_dynamic_target" \
      cargo test --locked -vv -p vllm-cpp-sys --release --tests \
        --no-default-features --features system,dynamic-link 2>&1 \
      | tee "$system_dynamic_log"
    grep -Fq "cargo:rerun-if-changed=$prefix/lib/libvllm.so" \
      "$system_dynamic_log"
    grep -Fq "cargo:rustc-link-search=native=$prefix/lib" \
      "$system_dynamic_log"

    system_override_dynamic_target="$work/system-override-dynamic"
    system_override_dynamic_log="$work/system-override-dynamic.log"
    echo '==> system dynamic invalid override'
    if LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      VLLM_CPP_ROOT="$prefix" \
      VLLM_CPP_LIB_DIR="$prefix/lib64" \
      CARGO_TARGET_DIR="$system_override_dynamic_target" \
      cargo test --locked -vv -p vllm-cpp-sys --release --tests \
        --no-default-features --features system,dynamic-link 2>&1 \
      | tee "$system_override_dynamic_log"; then
      echo 'expected system dynamic VLLM_CPP_LIB_DIR override to reject the empty lib64 directory' >&2
      exit 1
    fi
    grep -Fq \
      "expected VLLM_CPP_LIB_DIR override artifact libvllm.so at $prefix/lib64/libvllm.so" \
      "$system_override_dynamic_log"

    echo 'all four Linux CPU link modes passed'

# Test the packaged sys crate and an offline downstream fixture.
package-test:
    #!/usr/bin/env bash
    set -euo pipefail
    repo_root={{ quote(root) }}
    cd "$repo_root"
    package_target=${CARGO_TARGET_DIR:-$repo_root/target/package-gate}
    mkdir -p "$package_target"
    package_target=$(cd "$package_target" && pwd -P)
    export CARGO_TARGET_DIR="$package_target"

    command -v jq >/dev/null || {
      echo 'jq is required to parse Cargo metadata' >&2
      exit 1
    }
    if [[ ${CARGO_NET_OFFLINE:-false} != true ]]; then
      cargo fetch --locked
    fi
    export CARGO_NET_OFFLINE=true
    version=$(cargo metadata --locked --offline --no-deps --format-version 1 \
      | jq -er '[.packages[] | select(.name == "vllm-cpp-sys") | .version] | if length == 1 then .[0] else error("expected exactly one vllm-cpp-sys package") end')

    package_args=()
    if [[ -n $(git status --porcelain=v1 --untracked-files=all -- vllm-cpp-sys) ]]; then
      package_args+=(--allow-dirty)
    fi
    cargo package -p vllm-cpp-sys --locked --offline "${package_args[@]}"
    package_file="$package_target/package/vllm-cpp-sys-$version.crate"
    temp=$(mktemp -d)
    trap 'rm -rf "$temp"' EXIT
    tar -xzf "$package_file" -C "$temp"
    package_root="$temp/vllm-cpp-sys-$version"
    temp_target="$temp/target"

    package_size=$(stat -c '%s' "$package_file")
    unpacked_size=$(du -sb "$package_root" | cut -f1)
    entry_count=$(tar -tzf "$package_file" | wc -l)
    max_compressed=$((6 * 1024 * 1024))
    max_unpacked=$((36 * 1024 * 1024))
    max_entries=1300
    ((package_size <= max_compressed)) || {
      echo "package exceeds compressed budget: $package_size > $max_compressed" >&2
      exit 1
    }
    ((unpacked_size <= max_unpacked)) || {
      echo "package exceeds unpacked budget: $unpacked_size > $max_unpacked" >&2
      exit 1
    }
    ((entry_count <= max_entries)) || {
      echo "package exceeds entry budget: $entry_count > $max_entries" >&2
      exit 1
    }

    required_members=(
      Cargo.lock
      Cargo.toml
      LICENSE-APACHE
      LICENSE-MIT
      NOTICE
      README.md
      THIRD_PARTY.md
      build.rs
      licenses/FLASH-ATTENTION-BSD-3-CLAUSE.txt
      licenses/FLASH-LINEAR-ATTENTION-MIT.txt
      wrapper.h
      src/bindings.rs
      src/build_config.rs
      src/build_support.rs
      src/lib.rs
      tests/build_config.rs
      tests/build_support.rs
      tests/layout.c
      tests/layout.rs
      tests/symbols.rs
      vllm.cpp/CMakeLists.txt
      vllm.cpp/LICENSE
      vllm.cpp/NOTICE
      vllm.cpp/include/vllm.h
      vllm.cpp/src/capi/chat_prompt.cpp
      vllm.cpp/src/capi/chat_prompt.h
      vllm.cpp/src/capi/engine_handle.h
      vllm.cpp/src/capi/vllm_c.cpp
      vllm.cpp/src/vllm/version.cpp
      vllm.cpp/src/vt/cuda/triton_aot_vendored/sm_121a/MANIFEST
      vllm.cpp/scripts/triton-aot-compile.py
      vllm.cpp/triton_kernels/chunk_delta_h.py
      vllm.cpp/third_party/README.md
      vllm.cpp/third_party/blake3/LICENSE_A2
      vllm.cpp/third_party/blake3/LICENSE_CC0
      vllm.cpp/third_party/minja/LICENSE
      vllm.cpp/third_party/nlohmann/json.hpp
      vllm.cpp/third_party/vulkan/vulkan_core.h
    )
    for member in "${required_members[@]}"; do
      [[ -s $package_root/$member ]] || {
        echo "packaged crate is missing required member: $member" >&2
        exit 1
      }
    done

    native_inventory() {
      local base=$1
      local member
      shift
      for member in "$@"; do
        find "$base/$member" -type f -print
      done | sed "s#^$base/##" | LC_ALL=C sort
    }
    native_members=(
      vllm.cpp/CMakeLists.txt
      vllm.cpp/LICENSE
      vllm.cpp/NOTICE
      vllm.cpp/cmake
      vllm.cpp/include
      vllm.cpp/src
      vllm.cpp/scripts/triton-aot-compile.py
      vllm.cpp/triton_kernels
      vllm.cpp/third_party/README.md
      vllm.cpp/third_party/blake3
      vllm.cpp/third_party/minja
      vllm.cpp/third_party/nlohmann
      vllm.cpp/third_party/vulkan
    )
    diff -u \
      <(native_inventory "$repo_root/vllm-cpp-sys" "${native_members[@]}") \
      <(native_inventory "$package_root" "${native_members[@]}")
    diff -u \
      <(printf '%s\n' CMakeLists.txt LICENSE NOTICE cmake include scripts src third_party triton_kernels | LC_ALL=C sort) \
      <(find "$package_root/vllm.cpp" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)
    denied_members=(
      Justfile
      layout.c
      vllm.cpp/.agents
      vllm.cpp/.git
      vllm.cpp/.github
      vllm.cpp/assets
      vllm.cpp/benchmarks
      vllm.cpp/docs
      vllm.cpp/examples
      vllm.cpp/tests
      vllm.cpp/tools
      vllm.cpp/third_party/doctest
      vllm.cpp/third_party/httplib
    )
    for member in "${denied_members[@]}"; do
      [[ ! -e $package_root/$member ]] || {
        echo "denied upstream tree or record leaked into package: $member" >&2
        exit 1
      }
    done

    (
      cd "$package_root"
      CARGO_NET_OFFLINE=true CARGO_TARGET_DIR="$temp_target" \
        cargo test --locked --release --tests --offline
    )

    fixture="$temp/downstream"
    mkdir -p "$fixture/src"
    cat > "$fixture/Cargo.toml" <<EOF
    [package]
    name = "vllm-cpp-sys-package-smoke"
    version = "0.0.0"
    edition = "2021"
    publish = false

    [dependencies]
    vllm-cpp-sys = { path = "$package_root" }
    EOF
    cat > "$fixture/src/main.rs" <<'EOF'
    use std::ffi::CStr;

    use vllm_cpp_sys as ffi;

    fn main() {
        assert_eq!(ffi::VLLM_ABI_VERSION, 10);
        assert_eq!(unsafe { ffi::vllm_abi_version() }, 10);
        let version = unsafe { CStr::from_ptr(ffi::vllm_version()) };
        assert!(!version.to_bytes().is_empty());

        let mut params = unsafe { ffi::vllm_model_params_default() };
        params.model_path = c"/nonexistent/vllm-cpp-rs-package-smoke".as_ptr();
        let mut engine = std::ptr::null_mut();
        let status = unsafe { ffi::vllm_engine_load(&params, &mut engine) };
        assert_eq!(status, ffi::vllm_status_VLLM_ERR_MODEL_LOAD);
        assert!(engine.is_null());
    }
    EOF
    (
      cd "$fixture"
      CARGO_NET_OFFLINE=true cargo generate-lockfile --offline
      CARGO_NET_OFFLINE=true CARGO_TARGET_DIR="$temp_target" \
        cargo run --locked --release --offline
    )

    safe_version=$(cargo metadata --locked --offline --no-deps --format-version 1 \
      | jq -er '[.packages[] | select(.name == "vllm-cpp") | .version] | if length == 1 then .[0] else error("expected exactly one vllm-cpp package") end')
    safe_package_args=()
    if [[ -n $(git status --porcelain=v1 --untracked-files=all -- vllm-cpp vllm-cpp-sys) ]]; then
      safe_package_args+=(--allow-dirty)
    fi
    cargo package --workspace --locked --offline --no-verify "${safe_package_args[@]}"
    safe_package_file="$package_target/package/vllm-cpp-$safe_version.crate"
    tar -xzf "$safe_package_file" -C "$temp"
    safe_root="$temp/vllm-cpp-$safe_version"
    safe_manifest="$safe_root/Cargo.toml"
    sed -i \
      "/\[dependencies.vllm-cpp-sys\]/a path = \"$package_root\"" \
      "$safe_manifest"
    (
      cd "$safe_root"
      CARGO_NET_OFFLINE=true cargo generate-lockfile --offline
    )

    package_listing="$temp/safe-package.list"
    tar -tzf "$safe_package_file" > "$package_listing"
    if grep -Eq '(^|/)(target|model\.safetensors)(/|$)' "$package_listing"; then
      echo 'local build output or model fixture leaked into the safe crate package' >&2
      exit 1
    fi
    if grep -RIlF "$repo_root" "$safe_root" --exclude=Cargo.toml >/dev/null; then
      echo 'local repository path leaked into the safe crate package' >&2
      exit 1
    fi
    [[ ! -e $safe_root/target ]] || {
      echo 'local build output leaked into the safe crate package' >&2
      exit 1
    }
    [[ ! -e $safe_root/model.safetensors ]] || {
      echo 'model fixture leaked into the safe crate package' >&2
      exit 1
    }
    (
      cd "$safe_root"
      env -u VLLM_CPP_TEST_MODEL \
        CARGO_NET_OFFLINE=true CARGO_TARGET_DIR="$temp_target" \
        cargo test --locked --release --offline --features bundled,serde
    )

# Download and verify the pinned Qwen3-0.6B model fixture.
setup-test-model destination=env_var_or_default("VLLM_CPP_TEST_MODEL", env_var_or_default("XDG_CACHE_HOME", env_var("HOME") + "/.cache") + "/vllm-cpp-rs/Qwen3-0.6B-" + model_revision):
    #!/usr/bin/env bash
    set -euo pipefail
    revision={{ quote(model_revision) }}
    base="https://huggingface.co/Qwen/Qwen3-0.6B/resolve/$revision"
    destination={{ quote(destination) }}
    mkdir -p "$destination"

    files=(
      LICENSE
      config.json
      generation_config.json
      merges.txt
      model.safetensors
      tokenizer.json
      tokenizer_config.json
      vocab.json
    )
    for file in "${files[@]}"; do
      if [[ ! -f $destination/$file ]]; then
        echo "downloading $file" >&2
        curl --fail --location --retry 3 --continue-at - \
          "$base/$file" --output "$destination/$file"
      fi
    done

    cat > "$destination/SHA256SUMS.expected" <<'EOF'
    832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e  LICENSE
    660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd  config.json
    2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2  generation_config.json
    8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5  merges.txt
    f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b  model.safetensors
    aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4  tokenizer.json
    d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101  tokenizer_config.json
    ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910  vocab.json
    EOF
    (cd "$destination" && sha256sum --check SHA256SUMS.expected) >&2
    printf '%s\n' "$destination"

# Run the full safe/request/model suites with ASan, UBSan, and leak detection.
sanitizers model=env_var_or_default("VLLM_CPP_TEST_MODEL", ""):
    #!/usr/bin/env bash
    set -euo pipefail
    model={{ quote(model) }}
    if [[ -z $model ]]; then
      echo 'set VLLM_CPP_TEST_MODEL or pass model=<verified-model-directory>' >&2
      exit 1
    fi
    required_model_files=(
      model.safetensors
      config.json
      tokenizer.json
      tokenizer_config.json
    )
    missing=()
    for file in "${required_model_files[@]}"; do
      [[ -f $model/$file ]] || missing+=("$file")
    done
    if ((${#missing[@]})); then
      printf 'model fixture is incomplete at %s; missing:' "$model" >&2
      printf ' %s' "${missing[@]}" >&2
      printf '\n' >&2
      exit 1
    fi
    cd {{ quote(root) }}
    export VLLM_CPP_TEST_MODEL="$model"
    export VLLM_CPP_SANITIZE=address,undefined
    export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-{{ quote(root + "/target/sanitize") }}}

    cargo test --locked -p vllm-cpp --test safe_api --test qwen3 --no-run

    asan=$(gcc -print-file-name=libasan.so)
    ubsan=$(gcc -print-file-name=libubsan.so)
    if [[ ! -f $asan || ! -f $ubsan ]]; then
      echo 'GCC sanitizer runtimes are unavailable' >&2
      exit 1
    fi
    export LD_PRELOAD="$asan:$ubsan${LD_PRELOAD:+:$LD_PRELOAD}"
    export ASAN_OPTIONS=${ASAN_OPTIONS:-detect_leaks=1:halt_on_error=1}
    export UBSAN_OPTIONS=${UBSAN_OPTIONS:-halt_on_error=1:print_stacktrace=1}
    export VT_POOL_BYPASS=1

    for pattern in safe_api qwen3; do
      binary=$(find "$CARGO_TARGET_DIR/debug/deps" -maxdepth 1 -type f \
        -name "$pattern-*" -executable -printf '%T@ %p\n' \
        | sort -n | tail -1 | cut -d' ' -f2-) || true
      if [[ -z $binary ]]; then
        echo "could not find $pattern test binary" >&2
        exit 1
      fi
      "$binary" --test-threads=1
    done

# Run selected request lifecycle tests under native-only GCC TSan on Linux x86_64.
tsan model=env_var_or_default("VLLM_CPP_TEST_MODEL", ""):
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
      echo 'GCC TSan validation is supported only on Linux x86_64' >&2
      exit 1
    fi
    model={{ quote(model) }}
    if [[ -z $model ]]; then
      echo 'set VLLM_CPP_TEST_MODEL or pass model=<verified-model-directory>' >&2
      exit 1
    fi
    required_model_files=(
      model.safetensors
      config.json
      tokenizer.json
      tokenizer_config.json
    )
    missing=()
    for file in "${required_model_files[@]}"; do
      [[ -f $model/$file ]] || missing+=("$file")
    done
    if ((${#missing[@]})); then
      printf 'model fixture is incomplete at %s; missing:' "$model" >&2
      printf ' %s' "${missing[@]}" >&2
      printf '\n' >&2
      exit 1
    fi
    cd {{ quote(root) }}
    export VLLM_CPP_TEST_MODEL="$model"
    export VLLM_CPP_TEST_ISOLATED_ENGINE=1
    export VLLM_CPP_SANITIZE=thread
    export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-{{ quote(root + "/target/tsan") }}}

    cargo test --locked -p vllm-cpp --test qwen3 --no-run

    tsan=$(gcc -print-file-name=libtsan.so)
    if [[ ! -f $tsan ]]; then
      echo 'GCC ThreadSanitizer runtime is unavailable' >&2
      exit 1
    fi
    export LD_PRELOAD="$tsan${LD_PRELOAD:+:$LD_PRELOAD}"
    # Rust and its standard library are not instrumented in this GCC lane. Keep
    # native vllm.cpp races visible while ignoring uninstrumented Rust modules.
    export TSAN_OPTIONS=${TSAN_OPTIONS:-halt_on_error=1:ignore_noninstrumented_modules=1}
    export VT_POOL_BYPASS=1

    binary=$(find "$CARGO_TARGET_DIR/debug/deps" -maxdepth 1 -type f \
      -name 'qwen3-*' -executable -printf '%T@ %p\n' \
      | sort -n | tail -1 | cut -d' ' -f2-) || true
    if [[ -z $binary ]]; then
      echo 'could not find qwen3 test binary' >&2
      exit 1
    fi
    # Self-drop uses uninstrumented Rust synchronization, so normal and ASan/LSan
    # cover it. Every selected native lifecycle case runs in its own process.
    for test in \
      concurrent_requests_batch_with_correct_output \
      engine_clones_submit_and_wait_from_multiple_rust_threads \
      live_request_moves_to_rust_thread_for_cancel_wait_and_drop \
      request_outcomes_and_probes_are_precise \
      callback_panic_is_reported_and_engine_is_reusable \
      request_retains_engine_and_live_drop_is_safe \
      concurrent_request_lifecycle_stress; do
      "$binary" "$test" --exact --test-threads=1
    done

# Check Just and Rust formatting.
fmt-check:
    just --unstable --justfile {{ quote(root + "/Justfile") }} --fmt --check
    cargo fmt --all -- --check

# Lint all workspace targets with warnings denied.
lint:
    cargo clippy --locked --workspace --all-targets -- -D warnings

# Build API documentation with warnings denied.
docs:
    RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps --features vllm-cpp/serde

# Run the complete maintainer gate serially; do not pass --jobs.
ci: fmt-check lint docs sys link-modes package-test
