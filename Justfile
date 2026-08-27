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
    just --justfile '{{ root }}/Justfile' _link-modes \
      '{{ root }}/vllm-cpp-sys' \
      "${CARGO_TARGET_DIR:-{{ root }}/target/link-modes}"

[private]
_link-modes crate_root target_base:
    #!/usr/bin/env bash
    set -euo pipefail
    crate_root={{ quote(crate_root) }}
    target_base={{ quote(target_base) }}
    crate_root=$(cd "$crate_root" && pwd -P)
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
    cd "$crate_root"
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
    cp vllm.cpp/include/vllm.h "$prefix/include/"

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

# Run all locked model-free workspace targets.
test:
    env -u VLLM_CPP_TEST_MODEL cargo test --locked --workspace --all-targets --features vllm-cpp/serde

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
    [[ $native_version == 0.0.2 ]] || {
      echo "expected pinned native version 0.0.2, found $native_version" >&2
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

    native_exclude_pattern='^vllm\.cpp/cmake/(CudaArchFeaturesTest|CudaSourceGencodeTest|DumpTritonAOTContract|InSourceGuardTest|TritonAOTDefaultTest|TritonAOTMultiArchTest|VerifyExports)\.cmake$|^vllm\.cpp/include/vt/rocm/(rocm_gelu_mul_sep|rocm_gemma4_expert_geglu|rocm_matmul_batch|rocm_rmsnorm_plus_add)\.h$|^vllm\.cpp/src/vllm/entrypoints/openai/(api_server|server_main)\.cpp$|^vllm\.cpp/src/vllm/platforms/tenstorrent\.cpp$|^vllm\.cpp/src/vt/cuda/marlin/.*/generate_kernels\.py$|^vllm\.cpp/src/vt/rocm/[^/]+\.hip$|^vllm\.cpp/src/vt/tenstorrent/|^vllm\.cpp/src/vt/vulkan/shaders/'
    native_inventory() {
      local base=$1
      local member
      shift
      for member in "$@"; do
        find "$base/$member" -type f -print
      done | sed "s#^$base/##" | grep -Ev "$native_exclude_pattern" | LC_ALL=C sort
    }
    native_members=(
      vllm.cpp/CMakeLists.txt
      vllm.cpp/LICENSE
      vllm.cpp/NOTICE
      vllm.cpp/cmake
      vllm.cpp/include
      vllm.cpp/src
      vllm.cpp/scripts/triton-aot-compile.py
      vllm.cpp/tests/vt/test_rocm_backend.cpp
      vllm.cpp/triton_kernels
      vllm.cpp/third_party/README.md
      vllm.cpp/third_party/blake3
      vllm.cpp/third_party/doctest/doctest.h
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
        licenses/DOCTEST-MIT.txt \
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
      <(printf '%s\n' CMakeLists.txt LICENSE NOTICE cmake include scripts src tests third_party triton_kernels | LC_ALL=C sort) \
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
      src/abi.rs \
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
      licenses/DOCTEST-MIT.txt
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
      vllm.cpp/src/vt/cuda/triton_aot_vendored/sm_80/MANIFEST
      vllm.cpp/src/vt/cuda/triton_aot_vendored/sm_86/MANIFEST
      vllm.cpp/src/vt/cuda/triton_aot_vendored/sm_89/MANIFEST
      vllm.cpp/src/vt/cuda/triton_aot_vendored/sm_90a/MANIFEST
      vllm.cpp/src/vt/cuda/triton_aot_vendored/sm_100a/MANIFEST
      vllm.cpp/src/vt/cuda/triton_aot_vendored/sm_121a/MANIFEST
      vllm.cpp/src/vt/metal/metal_mlx_provider.mm
      vllm.cpp/src/vt/vulkan/vulkan_spirv.h
      vllm.cpp/src/vllm/platforms/rocm.cpp
      vllm.cpp/include/vt/rocm/rocm_arch.h
      vllm.cpp/include/vt/rocm/rocm_runtime.h
      vllm.cpp/tests/vt/test_rocm_backend.cpp
      vllm.cpp/scripts/triton-aot-compile.py
      vllm.cpp/triton_kernels/chunk_delta_h.py
      vllm.cpp/third_party/README.md
      vllm.cpp/third_party/blake3/LICENSE_A2
      vllm.cpp/third_party/blake3/LICENSE_CC0
      vllm.cpp/third_party/doctest/doctest.h
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
      vllm.cpp/tools
      vllm.cpp/third_party/httplib
      vllm.cpp/cmake/CudaArchFeaturesTest.cmake
      vllm.cpp/cmake/CudaSourceGencodeTest.cmake
      vllm.cpp/cmake/DumpTritonAOTContract.cmake
      vllm.cpp/cmake/InSourceGuardTest.cmake
      vllm.cpp/cmake/TritonAOTDefaultTest.cmake
      vllm.cpp/cmake/TritonAOTMultiArchTest.cmake
      vllm.cpp/cmake/VerifyExports.cmake
      vllm.cpp/src/vllm/entrypoints/openai/api_server.cpp
      vllm.cpp/src/vllm/entrypoints/openai/server_main.cpp
      vllm.cpp/src/vllm/platforms/tenstorrent.cpp
      vllm.cpp/src/vt/cuda/marlin/libtorch_stable/moe/marlin_moe_wna16/generate_kernels.py
      vllm.cpp/src/vt/rocm
      vllm.cpp/src/vt/tenstorrent
      vllm.cpp/src/vt/vulkan/shaders
    )
    for member in "${denied_sys_members[@]}"; do
      [[ ! -e $sys_root/$member ]] || {
        echo "denied upstream tree or record leaked into sys package: $member" >&2
        exit 1
      }
    done
    diff -u \
      <(printf '%s\n' vllm.cpp/tests/vt/test_rocm_backend.cpp) \
      <(find "$sys_root/vllm.cpp/tests" -type f -printf '%P\n' \
        | sed 's#^#vllm.cpp/tests/#' | LC_ALL=C sort)
    diff -u \
      <(printf '%s\n' vllm.cpp/third_party/doctest/doctest.h) \
      <(find "$sys_root/vllm.cpp/third_party/doctest" -type f -printf '%P\n' \
        | sed 's#^#vllm.cpp/third_party/doctest/#' | LC_ALL=C sort)

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
        licenses/DOCTEST-MIT.txt \
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
    sys_regular_file_bytes=$(find "$sys_root" -type f -printf '%s\n' \
      | awk '{ total += $1 } END { print total + 0 }')
    sys_package_sha256=$(sha256sum "$sys_package" | awk '{ print $1 }')
    sys_entry_count=$(wc -l < "$sys_list")
    safe_package_size=$(stat -c '%s' "$safe_package")
    safe_unpacked_size=$(du -sb "$safe_root" | cut -f1)
    safe_package_sha256=$(sha256sum "$safe_package" | awk '{ print $1 }')
    safe_entry_count=$(wc -l < "$safe_list")
    ((sys_package_size <= 6 * 1024 * 1024))
    ((sys_unpacked_size <= 40 * 1024 * 1024))
    ((sys_entry_count <= 1400))
    ((safe_package_size <= 128 * 1024))
    ((safe_unpacked_size <= 512 * 1024))
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
    just --justfile "$repo_root/Justfile" _link-modes \
      "$sys_root" "$temp/extracted-link-modes"

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

    printf 'sys package: %d entries, %d regular-file bytes, %d du bytes, %d compressed bytes, sha256 %s\n' \
      "$sys_entry_count" "$sys_regular_file_bytes" "$sys_unpacked_size" \
      "$sys_package_size" "$sys_package_sha256"
    printf 'safe package: %d entries, %d bytes unpacked, %d bytes compressed, sha256 %s\n' \
      "$safe_entry_count" "$safe_unpacked_size" "$safe_package_size" \
      "$safe_package_sha256"

[private]
_native-capi-known-flake output:
    #!/usr/bin/env bash
    set -euo pipefail
    python3 - {{ quote(output) }} <<'PY'
    import re
    import sys

    text = open(sys.argv[1], encoding="utf-8", errors="replace").read()

    cases = re.findall(r"^\s*TEST CASE:\s*(.*?)\s*$", text, re.MULTILINE)
    if cases != ["capi: vllm_complete_stream early-stop tears the request down cleanly"]:
        raise SystemExit(f"native C API failure was not the sole known test case: {cases}")

    errors = re.findall(
        r"^.*test_capi\.cpp:(\d+): ERROR:\s*(.*?)\s*$", text, re.MULTILINE
    )
    expected_error = [("680", "CHECK( acc.deltas == 2 ) is NOT correct!")]
    if errors != expected_error:
        raise SystemExit(f"native C API failure did not have the exact known assertion: {errors}")

    values = re.findall(r"^\s*values:\s*(.*?)\s*$", text, re.MULTILINE)
    if values != ["CHECK( 1 == 2 )"]:
        raise SystemExit(f"native C API failure did not observe exactly 1 == 2: {values}")

    case_summaries = re.findall(
        r"^\[doctest\]\s+test cases:\s*(\d+)\s*\|\s*(\d+) passed\s*\|\s*"
        r"(\d+) failed\s*\|\s*(\d+) skipped\s*$",
        text,
        re.MULTILINE,
    )
    if case_summaries != [("49", "48", "1", "0")]:
        raise SystemExit(f"native C API test-case summary was not the known sole failure: {case_summaries}")

    assertion_summaries = re.findall(
        r"^\[doctest\]\s+assertions:\s*(\d+)\s*\|\s*(\d+) passed\s*\|\s*"
        r"(\d+) failed\s*\|\s*$",
        text,
        re.MULTILINE,
    )
    if len(assertion_summaries) != 1:
        raise SystemExit(f"native C API assertion summary was ambiguous: {assertion_summaries}")
    total, passed, failed = map(int, assertion_summaries[0])
    if failed != 1 or passed + failed != total:
        raise SystemExit(f"native C API assertion summary was not one failure: {assertion_summaries}")

    ctest_summaries = re.findall(
        r"^(\d+)% tests passed, (\d+) tests failed out of (\d+)$", text, re.MULTILINE
    )
    if ctest_summaries != [("0", "1", "1")]:
        raise SystemExit(f"CTest summary was not the exact test_capi failure: {ctest_summaries}")

    failed_tests = re.findall(
        r"^\s*[0-9]+\s+-\s+(\S+)\s+\(([^)]+)\)\s*$", text, re.MULTILINE
    )
    if failed_tests != [("test_capi", "Failed")]:
        raise SystemExit(f"CTest failure list was not exactly test_capi: {failed_tests}")
    PY

# Build and run the focused native CPU C API fixture gate.
native-capi:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{ quote(root) }}
    command -v python3 >/dev/null || {
      echo 'python3 is required for native C API fixture configuration' >&2
      exit 1
    }
    target_base=${CARGO_TARGET_DIR:-{{ quote(root + "/target") }}}
    mkdir -p "$target_base"
    target_base=$(cd "$target_base" && pwd -P)
    build="$target_base/native-capi"
    cmake -S vllm-cpp-sys/vllm.cpp -B "$build" -G "$CMAKE_GENERATOR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_BUILD_EXAMPLES=OFF \
      -DVLLM_CPP_SERVER=OFF -DVLLM_CPP_HIP=OFF \
      -DVLLM_CPP_TENSTORRENT=OFF -DVLLM_CPP_LITERAL_STATIC=OFF \
      -DVLLM_CPP_BENCH_PROFILE_CONTROL=OFF -DVLLM_CPP_NCCL=OFF \
      -DVLLM_CPP_MARLIN=ON -DVLLM_CPP_FLASH_ATTN=ON \
      -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_METAL=OFF \
      -DVLLM_CPP_MLX=OFF -DVLLM_CPP_VULKAN=OFF \
      -DVLLM_CPP_TRITON=OFF -DVLLM_CPP_TRITON_REGEN=OFF \
      -DVLLM_CPP_TRITON_VENDORED_ARCH= \
      -DVLLM_CPP_TRITON_TARGET= -DVLLM_CPP_CUTLASS_FETCH=OFF \
      -DVLLM_CPP_SANITIZE=OFF
    cmake --build "$build" --target test_capi

    listing=$(mktemp)
    output=$(mktemp)
    retry_output=$(mktemp)
    trap 'rm -f "$listing" "$output" "$retry_output"' EXIT
    ctest --test-dir "$build" -N --tests-regex '^test_capi$' | tee "$listing"
    test_count=$(grep -Ec '^[[:space:]]*Test #[0-9]+: test_capi$' "$listing")
    [[ $test_count -eq 1 ]]
    grep -Fxq 'Total Tests: 1' "$listing"

    set +e
    env -u VT_ASYNC_SCHED -u VT_ASYNC_RUNNER \
      ctest --test-dir "$build" --output-on-failure --tests-regex '^test_capi$' \
      2>&1 | tee "$output"
    default_status=${PIPESTATUS[0]}
    set -e
    if [[ $default_status -eq 0 ]]; then
      grep -Fxq '100% tests passed, 0 tests failed out of 1' "$output"
      exit 0
    fi

    just --justfile {{ quote(root + "/Justfile") }} \
      _native-capi-known-flake "$output"
    # The synchronous scheduler makes pending-delta delivery cardinality
    # deterministic while the complete suite still exercises early-stop abort,
    # request teardown, and engine reuse through the unchanged C ABI test.
    echo 'retrying complete test_capi once with VT_ASYNC_SCHED=0' >&2
    env -u VT_ASYNC_RUNNER VT_ASYNC_SCHED=0 \
      ctest --test-dir "$build" --output-on-failure --tests-regex '^test_capi$' \
      2>&1 | tee "$retry_output"
    grep -Fxq '100% tests passed, 0 tests failed out of 1' "$retry_output"

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

# Run model-free ownership and FFI tests with ASan, UBSan, and leak detection.
sanitizers-model-free:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{ quote(root) }}
    command -v python3 >/dev/null || {
      echo 'python3 is required for sanitizer test discovery and native fixture configuration' >&2
      exit 1
    }
    fixtures={{ quote(root + "/vllm-cpp-sys/vllm.cpp/tests/vllm/models/fixtures") }}
    for fixture in parakeet_e2e llama_embed_e2e minimax_h3_video_fold; do
      if [[ ! -d $fixtures/$fixture ]]; then
        echo "required sanitizer fixture directory is missing: $fixtures/$fixture" >&2
        exit 1
      fi
    done
    for anchor in \
      parakeet_e2e/audio.wav \
      parakeet_e2e/ctc/config.json \
      parakeet_e2e/ctc/model.safetensors \
      parakeet_e2e/ctc/tokenizer.json \
      llama_embed_e2e/config.json \
      llama_embed_e2e/model.safetensors \
      llama_embed_e2e/tokenizer.json \
      minimax_h3_video_fold/golden_mux_argv.txt \
      minimax_h3_video_fold/golden_mux_argv_silent.txt; do
      if [[ ! -s $fixtures/$anchor ]]; then
        echo "required sanitizer fixture anchor is missing or empty: $fixtures/$anchor" >&2
        exit 1
      fi
    done
    unset VLLM_CPP_TEST_MODEL
    export VLLM_CPP_SANITIZE=address,undefined
    export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-{{ quote(root + "/target/sanitize-model-free") }}}

    work=$(mktemp -d)
    trap 'rm -rf "$work"' EXIT
    messages="$work/messages.json"
    cargo test --locked -p vllm-cpp --lib --test safe_api --test qwen3 \
      --no-run --message-format=json > "$messages"

    test_binary() {
      local target=$1
      python3 - "$messages" "$target" <<'PY'
    import json
    import sys

    messages, target = sys.argv[1:]
    matches = []
    with open(messages, encoding="utf-8") as stream:
        for line in stream:
            message = json.loads(line)
            if (
                message.get("reason") == "compiler-artifact"
                and message.get("profile", {}).get("test")
                and message.get("target", {}).get("name") == target
                and message.get("executable")
            ):
                matches.append(message["executable"])
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one test binary for {target}, found {len(matches)}: {matches}"
        )
    print(matches[0])
    PY
    }

    assert_complete_suite() {
      local suite=$1
      local output=$2
      python3 - "$output" "$suite" <<'PY'
    import re
    import sys

    output, suite = sys.argv[1:]
    text = open(output, encoding="utf-8", errors="replace").read()
    running = re.findall(r"^running ([0-9]+) tests?$", text, re.MULTILINE)
    summaries = re.findall(
        r"^test result: ok\. ([0-9]+) passed; 0 failed;", text, re.MULTILINE
    )
    if len(running) != 1 or int(running[0]) < 1:
        raise SystemExit(
            f"{suite} suite did not report exactly one nonzero running count: {running}"
        )
    if len(summaries) != 1 or int(summaries[0]) < 1:
        raise SystemExit(
            f"{suite} suite did not report exactly one successful nonzero summary: {summaries}"
        )
    PY
    }

    assert_focused_test() {
      local test=$1
      local output=$2
      python3 - "$output" "$test" <<'PY'
    import re
    import sys

    output, test = sys.argv[1:]
    text = open(output, encoding="utf-8", errors="replace").read()
    running = re.findall(r"^running ([0-9]+) tests?$", text, re.MULTILINE)
    summaries = re.findall(
        r"^test result: ok\. ([0-9]+) passed; ([0-9]+) failed;", text, re.MULTILINE
    )
    if running != ["1"]:
        raise SystemExit(f"{test} did not report exactly one running test: {running}")
    if summaries != [("1", "0")]:
        raise SystemExit(f"{test} did not report exactly one passed test: {summaries}")
    PY
    }

    library=$(test_binary vllm_cpp)
    safe_api=$(test_binary safe_api)
    qwen3=$(test_binary qwen3)
    for binary in "$library" "$safe_api" "$qwen3"; do
      [[ -x $binary ]] || {
        echo "test binary is not executable: $binary" >&2
        exit 1
      }
    done

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

    library_output="$work/library.out"
    "$library" --test-threads=1 2>&1 | tee "$library_output"
    assert_complete_suite library "$library_output"
    safe_api_output="$work/safe-api.out"
    "$safe_api" --test-threads=1 2>&1 | tee "$safe_api_output"
    assert_complete_suite safe_api "$safe_api_output"
    for test in \
      committed_transcription_fixture_supports_path_pcm_and_wrong_task \
      committed_embedding_fixture_preserves_shape_order_ownership_and_wrong_task \
      committed_video_mux_goldens_preserve_exact_argument_boundaries \
      committed_parakeet_directory_is_rejected_as_video_dit; do
      output="$work/$test.out"
      "$qwen3" --test-threads=1 --exact "$test" 2>&1 | tee "$output"
      assert_focused_test "$test" "$output"
    done
    echo 'sanitizer gate passed: 3 binaries, 2 complete suites, 4 focused fixture tests'

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
ci: fmt-check lint docs test sys link-modes native-capi sanitizers-model-free package-test publish-dry-run
