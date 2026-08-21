#[path = "../src/build_support.rs"]
mod build_support;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use build_support::{
    cmake_cache_path, find_installed_library_dir, require_library_file, shared_library_name,
};

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(label: &str) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is before the Unix epoch")
            .as_nanos();
        let path = env::temp_dir().join(format!(
            "vllm-cpp-sys-build-support-{}-{label}-{timestamp}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("failed to create temporary directory");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn write_contents(path: &Path, contents: &[u8]) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("failed to create parent directory");
    }
    fs::write(path, contents).expect("failed to write test artifact");
}

fn write_file(path: &Path) {
    write_contents(path, b"archive");
}

#[test]
fn shared_library_name_is_target_aware() {
    assert_eq!(shared_library_name("vllm", "linux").unwrap(), "libvllm.so");
    assert_eq!(
        shared_library_name("vllm", "macos").unwrap(),
        "libvllm.dylib"
    );
    assert_eq!(
        shared_library_name("vllm", "windows").unwrap_err(),
        "shared libraries are unsupported for target OS windows"
    );
}

#[test]
fn selects_lib_when_lib64_lacks_the_requested_artifact() {
    let temp = TempDir::new("select-lib");
    fs::create_dir(temp.path().join("lib64")).expect("failed to create lib64");
    write_file(&temp.path().join("lib/libvllm.a"));

    let selected = find_installed_library_dir(temp.path(), "libvllm.a")
        .expect("expected helper to select lib directory");

    assert_eq!(selected, temp.path().join("lib"));
}

#[test]
fn selects_lib64_when_it_contains_the_requested_shared_object() {
    let temp = TempDir::new("select-lib64");
    fs::create_dir(temp.path().join("lib")).expect("failed to create lib");
    write_file(&temp.path().join("lib64/libvllm.so"));

    let selected = find_installed_library_dir(temp.path(), "libvllm.so")
        .expect("expected helper to select lib64 directory");

    assert_eq!(selected, temp.path().join("lib64"));
}

#[test]
fn reports_missing_candidate_artifacts() {
    let temp = TempDir::new("missing");
    fs::create_dir(temp.path().join("lib")).expect("failed to create lib");
    fs::create_dir(temp.path().join("lib64")).expect("failed to create lib64");

    let error = find_installed_library_dir(temp.path(), "libvllm.a")
        .expect_err("expected helper to report a missing library artifact");

    assert!(error.contains("expected libvllm.a below"));
    assert!(error.contains(&temp.path().display().to_string()));
    assert!(error.contains(&temp.path().join("lib").display().to_string()));
    assert!(error.contains(&temp.path().join("lib64").display().to_string()));
}

#[test]
fn reports_ambiguous_installed_library_dirs() {
    let temp = TempDir::new("ambiguous");
    write_file(&temp.path().join("lib/libvllm.a"));
    write_file(&temp.path().join("lib64/libvllm.a"));

    let error = find_installed_library_dir(temp.path(), "libvllm.a")
        .expect_err("expected helper to reject ambiguous library directories");

    assert!(error.contains("found ambiguous libvllm.a artifacts below"));
    assert!(error.contains(&temp.path().join("lib").display().to_string()));
    assert!(error.contains(&temp.path().join("lib64").display().to_string()));
    assert!(error.contains("VLLM_CPP_LIB_DIR"));
}

#[test]
fn require_library_file_reports_the_expected_path() {
    let temp = TempDir::new("require");
    let expected = temp.path().join("lib/libvllm.a");
    write_file(&expected);

    let actual = require_library_file(&temp.path().join("lib"), "libvllm.a", "system static vllm")
        .expect("expected helper to return the exact library path");
    assert_eq!(actual, expected);

    let error = require_library_file(
        &temp.path().join("lib"),
        "libblake3_vendored.a",
        "system static blake3_vendored",
    )
    .expect_err("expected helper to report a missing library");
    assert!(
        error.contains("expected system static blake3_vendored artifact libblake3_vendored.a at")
    );
    assert!(error.contains(
        &temp
            .path()
            .join("lib/libblake3_vendored.a")
            .display()
            .to_string()
    ));
}

#[test]
fn cmake_cache_path_requires_one_nonempty_absolute_value() {
    let temp = TempDir::new("cmake-cache-errors");
    let cache = temp.path().join("CMakeCache.txt");

    write_contents(&cache, b"OTHER:FILEPATH=/tmp/libother.so\n");
    let error = cmake_cache_path(&cache, "CUDA_CUDART").expect_err("key should be absent");
    assert!(error.contains("CUDA_CUDART is absent"));

    write_contents(&cache, b"CUDA_CUDART:FILEPATH=\n");
    let error = cmake_cache_path(&cache, "CUDA_CUDART").expect_err("value should be empty");
    assert!(error.contains("CUDA_CUDART is empty"));

    write_contents(&cache, b"CUDA_CUDART:FILEPATH=CUDA_CUDART-NOTFOUND\n");
    let error = cmake_cache_path(&cache, "CUDA_CUDART").expect_err("value should be unresolved");
    assert!(error.contains("CUDA_CUDART is unresolved"));

    write_contents(&cache, b"CUDA_CUDART:FILEPATH=libcudart.so\n");
    let error = cmake_cache_path(&cache, "CUDA_CUDART").expect_err("value should be relative");
    assert!(error.contains("must be an absolute path"));

    write_contents(
        &cache,
        b"CUDA_CUDART:FILEPATH=/cuda/lib/libcudart.so\nCUDA_CUDART:STRING=/other/libcudart.so\n",
    );
    let error = cmake_cache_path(&cache, "CUDA_CUDART").expect_err("key should be duplicated");
    assert!(error.contains("occurs more than once"));
}

#[test]
fn cmake_cache_path_returns_the_exact_absolute_value() {
    let temp = TempDir::new("cmake-cache-valid");
    let cache = temp.path().join("CMakeCache.txt");
    write_contents(
        &cache,
        b"CUDA_cublasLt_LIBRARY:FILEPATH=/nix/store/toolkit/lib/libcublasLt.so\n",
    );

    assert_eq!(
        cmake_cache_path(&cache, "CUDA_cublasLt_LIBRARY").unwrap(),
        PathBuf::from("/nix/store/toolkit/lib/libcublasLt.so")
    );
}
