set shell := ["bash", "-euo", "pipefail", "-c"]

root := justfile_directory()
bindings_file := root + "/vllm-cpp-sys/src/bindings.rs"

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
    vllm_complete_tokens
    vllm_completion_free
    vllm_embed
    vllm_embedding_result_free
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
    vllm_server_main
    vllm_string_free
    vllm_transcribe
    vllm_transcription_free
    vllm_transcription_params_default
    vllm_version
    vllm_video_engine_free
    vllm_video_engine_load
    vllm_video_generate
    vllm_video_model_params_default
    vllm_video_mux_argv
    vllm_video_mux_argv_free
    vllm_video_mux_params_default
    vllm_video_params_default
    vllm_video_result_free
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

# Check locked model-free targets on the exact Rust 1.85.0 toolchain.
msrv:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{ quote(root) }}
    rust_version=$(rustc --version | awk '{print $2}')
    cargo_version=$(cargo --version | awk '{print $2}')
    if [[ $rust_version != 1.85.0 || $cargo_version != 1.85.0 ]]; then
      echo "msrv requires rustc and cargo 1.85.0; found rustc $rust_version and cargo $cargo_version" >&2
      exit 1
    fi
    export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-{{ quote(root + "/target/msrv") }}}
    env -u VLLM_CPP_TEST_MODEL \
      cargo check --locked --workspace --all-targets --features vllm-cpp/serde

# Validate both crate archives, extracted builds, and downstream consumers.
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

    temp=$(mktemp -d)
    trap 'rm -rf "$temp"' EXIT
    metadata="$temp/workspace-metadata.json"
    cargo metadata --locked --offline --no-deps --format-version 1 > "$metadata"
    version=$(jq -er '
      [.packages[] | select(.name == "vllm-cpp-sys") | .version]
      | if length == 1 then .[0] else error("expected exactly one vllm-cpp-sys package") end
    ' "$metadata")
    safe_version=$(jq -er '
      [.packages[] | select(.name == "vllm-cpp") | .version]
      | if length == 1 then .[0] else error("expected exactly one vllm-cpp package") end
    ' "$metadata")
    [[ $safe_version == "$version" ]] || {
      echo "crate versions differ: sys=$version safe=$safe_version" >&2
      exit 1
    }
    native_version=$(sed -nE \
      's/^project\(vllm_cpp VERSION ([0-9]+\.[0-9]+\.[0-9]+) LANGUAGES CXX\)$/\1/p' \
      "$repo_root/vllm-cpp-sys/vllm.cpp/CMakeLists.txt")
    [[ -n $native_version ]] || {
      echo 'could not read the native project version from vllm.cpp/CMakeLists.txt' >&2
      exit 1
    }
    [[ $native_version == "$version" ]] || {
      echo "native and crate versions differ: native=$native_version crates=$version" >&2
      exit 1
    }
    jq -e --arg version "$version" '
      [.packages[] | select(.name == "vllm-cpp" or .name == "vllm-cpp-sys")]
      | length == 2
        and all(.version == $version)
        and all(.rust_version == "1.85")
        and all(.license == "MIT OR Apache-2.0")
        and all(.repository == "https://github.com/querymt/vllm-cpp-rs")
        and all(.readme == "README.md")
        and (map(select(.name == "vllm-cpp" and .documentation == "https://docs.rs/vllm-cpp")) | length == 1)
        and (map(select(.name == "vllm-cpp-sys" and .documentation == "https://docs.rs/vllm-cpp-sys")) | length == 1)
        and (map(select(.name == "vllm-cpp") | .dependencies[] | select(.name == "vllm-cpp-sys" and .req == ("=" + $version) and .uses_default_features == false)) | length == 1)
        and (map(select(.name == "vllm-cpp") | .dependencies[] | select(.name == "hf-hub" and .req == "^0.5.0" and .optional == false and .uses_default_features == false and .features == ["ureq"])) | length == 1)
        and (map(select(.name == "vllm-cpp") | .dependencies[] | select(.name == "serde_json" and .optional == false and .kind == null)) | length == 1)
        and (map(select(.name == "vllm-cpp") | .dependencies[] | select(.name == "clap" and .req == "=4.6.1" and .kind == "dev" and .optional == false and .features == ["derive"])) | length == 1)
    ' "$metadata" >/dev/null

    sys_list="$temp/vllm-cpp-sys.list"
    safe_list="$temp/vllm-cpp.list"
    cargo package -p vllm-cpp-sys --locked --offline --allow-dirty --list \
      | LC_ALL=C sort -u > "$sys_list"
    cargo package -p vllm-cpp --locked --offline --allow-dirty --list \
      | LC_ALL=C sort -u > "$safe_list"

    cargo package -p vllm-cpp-sys --locked --offline --allow-dirty
    cargo package --workspace --locked --offline --allow-dirty --no-verify
    sys_package="$package_target/package/vllm-cpp-sys-$version.crate"
    safe_package="$package_target/package/vllm-cpp-$safe_version.crate"
    [[ -s $sys_package && -s $safe_package ]]

    tar -xzf "$sys_package" -C "$temp"
    tar -xzf "$safe_package" -C "$temp"
    sys_root="$temp/vllm-cpp-sys-$version"
    safe_root="$temp/vllm-cpp-$safe_version"
    temp_target="$temp/target"

    archive_inventory() {
      local archive=$1
      local prefix=$2
      tar -tzf "$archive" \
        | sed "s#^$prefix/##" \
        | grep -v '/$' \
        | LC_ALL=C sort
    }
    diff -u "$sys_list" <(archive_inventory "$sys_package" "vllm-cpp-sys-$version")
    diff -u "$safe_list" <(archive_inventory "$safe_package" "vllm-cpp-$safe_version")

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
    sys_expected="$temp/vllm-cpp-sys.expected"
    {
      printf '%s\n' \
        .cargo_vcs_info.json \
        Cargo.lock \
        Cargo.toml \
        Cargo.toml.orig \
        LICENSE-APACHE \
        LICENSE-MIT \
        NOTICE \
        README.md \
        THIRD_PARTY.md \
        build.rs \
        licenses/FLASH-ATTENTION-BSD-3-CLAUSE.txt \
        licenses/FLASH-LINEAR-ATTENTION-MIT.txt \
        src/bindings.rs \
        src/build_config.rs \
        src/build_support.rs \
        src/lib.rs \
        tests/build_config.rs \
        tests/build_support.rs \
        tests/layout.c \
        tests/layout.rs \
        tests/symbols.rs \
        wrapper.h
      native_inventory "$repo_root/vllm-cpp-sys" "${native_members[@]}"
    } | LC_ALL=C sort > "$sys_expected"
    diff -u "$sys_expected" "$sys_list"
    diff -u \
      <(native_inventory "$repo_root/vllm-cpp-sys" "${native_members[@]}") \
      <(native_inventory "$sys_root" "${native_members[@]}")
    diff -u \
      <(printf '%s\n' CMakeLists.txt LICENSE NOTICE cmake include scripts src third_party triton_kernels | LC_ALL=C sort) \
      <(find "$sys_root/vllm.cpp" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)

    safe_expected="$temp/vllm-cpp.expected"
    printf '%s\n' \
      .cargo_vcs_info.json \
      Cargo.lock \
      Cargo.toml \
      Cargo.toml.orig \
      LICENSE-APACHE \
      LICENSE-MIT \
      README.md \
      examples/README.md \
      examples/chat.rs \
      examples/common/mod.rs \
      examples/complete.rs \
      examples/concurrent.rs \
      examples/setup_test_model.rs \
      examples/stream.rs \
      examples/structured.rs \
      src/callback.rs \
      src/engine.rs \
      src/error.rs \
      src/hf.rs \
      src/lib.rs \
      src/params.rs \
      src/request.rs \
      tests/qwen3.rs \
      tests/safe_api.rs \
      | LC_ALL=C sort > "$safe_expected"
    diff -u "$safe_expected" "$safe_list"

    required_sys_members=(
      README.md
      THIRD_PARTY.md
      licenses/FLASH-ATTENTION-BSD-3-CLAUSE.txt
      licenses/FLASH-LINEAR-ATTENTION-MIT.txt
      vllm.cpp/include/vllm.h
      vllm.cpp/src/capi/chat_prompt.cpp
      vllm.cpp/src/capi/chat_prompt.h
      vllm.cpp/src/capi/engine_handle.h
      vllm.cpp/src/capi/vllm_c.cpp
      vllm.cpp/src/vllm/version.cpp
      vllm.cpp/src/vt/cuda/cuda_matmul_fp8_cutlass.cu
      vllm.cpp/src/vt/cuda/flash_attn/src/flash.h
      vllm.cpp/src/vt/cuda/marlin/core/scalar_type.hpp
      vllm.cpp/src/vt/cuda/triton_aot_vendored/sm_121a/MANIFEST
      vllm.cpp/scripts/triton-aot-compile.py
      vllm.cpp/triton_kernels/chunk_delta_h.py
      vllm.cpp/src/vt/vulkan/vulkan_spirv.h
      vllm.cpp/third_party/README.md
      vllm.cpp/third_party/blake3/LICENSE_A2
      vllm.cpp/third_party/blake3/LICENSE_CC0
      vllm.cpp/third_party/minja/LICENSE
      vllm.cpp/third_party/nlohmann/json.hpp
      vllm.cpp/third_party/vulkan/vulkan_core.h
    )
    for member in "${required_sys_members[@]}"; do
      [[ -s $sys_root/$member ]] || {
        echo "sys package is missing required member: $member" >&2
        exit 1
      }
    done
    while IFS= read -r member; do
      [[ -s $safe_root/$member ]] || {
        echo "safe package is missing required member: $member" >&2
        exit 1
      }
    done < "$safe_expected"

    denied_sys_members=(
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
    for member in "${denied_sys_members[@]}"; do
      [[ ! -e $sys_root/$member ]] || {
        echo "denied upstream tree or record leaked into sys package: $member" >&2
        exit 1
      }
    done

    forbidden_pattern='(^|/)(target|stuff|\.git|\.github|__pycache__|\.cache|cache|fixtures?|downloads?|_deps|sdk)(/|$)|(^|/)(cutlass)(/|$)|(^|/)(model\.safetensors|tokenizer\.json|tokenizer_config\.json)$|\.(o|obj|a|so|dylib|dll|pyc|safetensors|gguf|pt|pth)$'
    for listing in "$sys_list" "$safe_list"; do
      if grep -Eiq "$forbidden_pattern" "$listing"; then
        echo "forbidden package payload detected in $listing:" >&2
        grep -Ei "$forbidden_pattern" "$listing" >&2
        exit 1
      fi
    done

    scan_authored_paths() {
      local package_root=$1
      local files=()
      if [[ -d $package_root/vllm.cpp ]]; then
        mapfile -d '' files < <(find "$package_root" \
          -path "$package_root/vllm.cpp" -prune -o -type f -print0)
      else
        mapfile -d '' files < <(find "$package_root" -type f -print0)
      fi
      if ((${#files[@]})) && grep -IlF "$repo_root" "${files[@]}" >/dev/null; then
        echo "local repository path leaked into $package_root" >&2
        grep -IlF "$repo_root" "${files[@]}" >&2
        exit 1
      fi
      if grep -E '^(path|git)[[:space:]]*=' "$package_root/Cargo.toml.orig"; then
        echo "local Cargo dependency source leaked into $package_root" >&2
        exit 1
      fi
    }
    scan_authored_paths "$sys_root"
    scan_authored_paths "$safe_root"

    sys_license_actual="$temp/sys-licenses.actual"
    find "$sys_root" -type f -printf '%P\n' \
      | grep -Ei '(^|/)(license[^/]*|copying[^/]*|notice[^/]*)$|^THIRD_PARTY\.md$|^licenses/' \
      | LC_ALL=C sort > "$sys_license_actual"
    diff -u \
      <(printf '%s\n' \
        LICENSE-APACHE \
        LICENSE-MIT \
        NOTICE \
        THIRD_PARTY.md \
        licenses/FLASH-ATTENTION-BSD-3-CLAUSE.txt \
        licenses/FLASH-LINEAR-ATTENTION-MIT.txt \
        vllm.cpp/LICENSE \
        vllm.cpp/NOTICE \
        vllm.cpp/third_party/blake3/LICENSE_A2 \
        vllm.cpp/third_party/blake3/LICENSE_CC0 \
        vllm.cpp/third_party/minja/LICENSE | LC_ALL=C sort) \
      "$sys_license_actual"
    diff -u \
      <(printf '%s\n' LICENSE-APACHE LICENSE-MIT | LC_ALL=C sort) \
      <(find "$safe_root" -type f -printf '%P\n' \
        | grep -Ei '(^|/)(license[^/]*|copying[^/]*|notice[^/]*)$' \
        | LC_ALL=C sort)

    check_package_links() {
      local package_root=$1
      shift
      local document document_dir link path
      for document in "$@"; do
        document_dir=$(dirname "$document")
        while IFS= read -r link; do
          case $link in
            http://*|https://*|mailto:*|'#'*) continue ;;
          esac
          path=${link%%#*}
          [[ -e $package_root/$document_dir/$path ]] || {
            echo "broken packaged relative link in $document: $link" >&2
            exit 1
          }
        done < <(grep -oE '\]\([^)]+\)' "$package_root/$document" \
          | sed -e 's/^](//' -e 's/)$//' || true)
      done
    }
    check_package_links "$sys_root" README.md THIRD_PARTY.md
    check_package_links "$safe_root" README.md examples/README.md

    sys_package_size=$(stat -c '%s' "$sys_package")
    sys_unpacked_size=$(du -sb "$sys_root" | cut -f1)
    sys_entry_count=$(wc -l < "$sys_list")
    safe_package_size=$(stat -c '%s' "$safe_package")
    safe_unpacked_size=$(du -sb "$safe_root" | cut -f1)
    safe_entry_count=$(wc -l < "$safe_list")
    ((sys_package_size <= 6 * 1024 * 1024))
    ((sys_unpacked_size <= 36 * 1024 * 1024))
    ((sys_entry_count <= 1300))
    ((safe_package_size <= 128 * 1024))
    ((safe_unpacked_size <= 256 * 1024))
    ((safe_entry_count <= 40))

    jq -e --arg version "$version" '
      .packages | length == 1
        and .[0].name == "vllm-cpp-sys"
        and .[0].version == $version
        and .[0].readme == "README.md"
        and .[0].documentation == "https://docs.rs/vllm-cpp-sys"
        and .[0].license == "MIT OR Apache-2.0"
        and .[0].rust_version == "1.85"
        and .[0].features.default == ["bundled"]
    ' <(cargo metadata --manifest-path "$sys_root/Cargo.toml" \
      --locked --offline --no-deps --format-version 1) >/dev/null
    (
      cd "$sys_root"
      CARGO_NET_OFFLINE=true CARGO_TARGET_DIR="$temp_target" \
        cargo test --locked --release --tests --offline
    )

    sys_consumer="$temp/sys-consumer"
    mkdir -p "$sys_consumer/src"
    cat > "$sys_consumer/Cargo.toml" <<EOF
    [package]
    name = "vllm-cpp-sys-package-smoke"
    version = "0.0.0"
    edition = "2021"
    publish = false

    [dependencies]
    vllm-cpp-sys = { path = "$sys_root" }
    EOF
    cat > "$sys_consumer/src/main.rs" <<'EOF'
    use std::ffi::CStr;

    use vllm_cpp_sys as ffi;

    fn main() {
        assert_eq!(ffi::VLLM_ABI_VERSION, 17);
        assert_eq!(unsafe { ffi::vllm_abi_version() }, 17);
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
      cd "$sys_consumer"
      CARGO_NET_OFFLINE=true cargo generate-lockfile --offline
      CARGO_NET_OFFLINE=true CARGO_TARGET_DIR="$temp_target" \
        cargo run --locked --release --offline
    )

    mkdir -p "$safe_root/.cargo"
    cat > "$safe_root/.cargo/config.toml" <<EOF
    [patch.crates-io]
    vllm-cpp-sys = { path = "$sys_root" }
    EOF
    (
      cd "$safe_root"
      rm Cargo.lock
      CARGO_NET_OFFLINE=true cargo generate-lockfile --offline
      env -u VLLM_CPP_TEST_MODEL \
        CARGO_NET_OFFLINE=true CARGO_TARGET_DIR="$temp_target" \
        cargo test --locked --release --offline --features bundled,serde
    )
    jq -e --arg version "$safe_version" '
      .packages | length == 1
        and .[0].name == "vllm-cpp"
        and .[0].version == $version
        and .[0].readme == "README.md"
        and .[0].documentation == "https://docs.rs/vllm-cpp"
        and .[0].license == "MIT OR Apache-2.0"
        and .[0].rust_version == "1.85"
        and .[0].features.default == ["bundled"]
        and .[0].features.serde == []
        and (.[0].dependencies | map(select(.name == "vllm-cpp-sys" and .req == ("=" + $version) and .uses_default_features == false)) | length == 1)
        and (.[0].dependencies | map(select(.name == "hf-hub" and .req == "^0.5.0" and .optional == false and .uses_default_features == false and .features == ["ureq"])) | length == 1)
        and (.[0].dependencies | map(select(.name == "serde_json" and .optional == false and .kind == null)) | length == 1)
        and (.[0].dependencies | map(select(.name == "clap" and .req == "=4.6.1" and .kind == "dev" and .optional == false and .features == ["derive"])) | length == 1)
    ' <(cargo metadata --manifest-path "$safe_root/Cargo.toml" \
      --locked --offline --no-deps --format-version 1) >/dev/null

    safe_consumer="$temp/safe-consumer"
    mkdir -p "$safe_consumer/src"
    cat > "$safe_consumer/Cargo.toml" <<EOF
    [package]
    name = "vllm-cpp-package-smoke"
    version = "0.0.0"
    edition = "2021"
    publish = false

    [dependencies]
    vllm-cpp = { path = "$safe_root", default-features = false, features = ["bundled", "serde"] }

    [patch.crates-io]
    vllm-cpp-sys = { path = "$sys_root" }
    EOF
    cat > "$safe_consumer/src/main.rs" <<'EOF'
    use vllm_cpp::{
        abi_version, expected_abi_version, version, Engine, Error, HuggingFaceModel, SamplingParams,
    };

    fn main() {
        assert_eq!(expected_abi_version(), 17);
        assert_eq!(abi_version(), 17);
        assert!(!version().expect("native version").is_empty());
        let _params = SamplingParams::greedy()
            .max_tokens(1)
            .logits_processor(|_, logits| logits.fill(0.0));
        let resolver = HuggingFaceModel::gguf("owner/model", "model.gguf")
            .revision("revision")
            .cache_dir("/nonexistent/vllm-cpp-rs-safe-package-hf-cache")
            .offline(true);
        assert!(resolver.resolve().is_err());
        assert!(matches!(
            Engine::load("/nonexistent/vllm-cpp-rs-safe-package-smoke"),
            Err(Error::ModelLoad { .. })
        ));
    }
    EOF
    (
      cd "$safe_consumer"
      CARGO_NET_OFFLINE=true cargo generate-lockfile --offline
      metadata=$(CARGO_NET_OFFLINE=true cargo metadata --locked --offline --format-version 1)
      safe_manifest=$(realpath "$safe_root/Cargo.toml")
      sys_manifest=$(realpath "$sys_root/Cargo.toml")
      jq -e --arg safe "$safe_manifest" --arg sys "$sys_manifest" '
        any(.packages[]; .name == "vllm-cpp" and .manifest_path == $safe)
          and any(.packages[]; .name == "vllm-cpp-sys" and .manifest_path == $sys)
      ' <<<"$metadata" >/dev/null
      CARGO_NET_OFFLINE=true CARGO_TARGET_DIR="$temp_target" \
        cargo run --locked --release --offline
    )

    printf 'sys package: %d entries, %d bytes unpacked, %d bytes compressed\n' \
      "$sys_entry_count" "$sys_unpacked_size" "$sys_package_size"
    printf 'safe package: %d entries, %d bytes unpacked, %d bytes compressed\n' \
      "$safe_entry_count" "$safe_unpacked_size" "$safe_package_size"

# Run sys-first crates.io publication checks without uploading.
publish-dry-run:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{ quote(root) }}
    export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-{{ quote(root + "/target/publish-dry-run") }}}
    # package-test performs the full extracted/offline verification. Workspace
    # dry-run preserves Cargo's sys-first order without requiring sys on crates.io.
    cargo publish --workspace --locked --dry-run --allow-dirty --no-verify

# Resolve the pinned Qwen3-0.6B test fixture into the standard Hugging Face cache.
setup-test-model:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{ quote(root) }}
    cargo run --quiet --locked -p vllm-cpp --example setup_test_model

# Run the full safe/request/model suites with ASan, UBSan, and leak detection.
sanitizers model=env_var_or_default("VLLM_CPP_TEST_MODEL", ""):
    #!/usr/bin/env bash
    set -euo pipefail
    model={{ quote(model) }}
    if [[ -z $model ]]; then
      echo 'set VLLM_CPP_TEST_MODEL or pass model=<prepared-model-directory>' >&2
      exit 1
    fi
    if [[ ! -d $model ]]; then
      echo "model fixture is not a directory: $model" >&2
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
      echo 'set VLLM_CPP_TEST_MODEL or pass model=<prepared-model-directory>' >&2
      exit 1
    fi
    if [[ ! -d $model ]]; then
      echo "model fixture is not a directory: $model" >&2
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
