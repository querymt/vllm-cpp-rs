use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const RERUN_ENV: &[&str] = &[
    "CMAKE_BUILD_PARALLEL_LEVEL",
    "CMAKE_GENERATOR",
    "CMAKE_GENERATOR_PLATFORM",
    "CMAKE_GENERATOR_TOOLSET",
    "CMAKE_PREFIX_PATH",
    "CC",
    "CFLAGS",
    "CXX",
    "CXXFLAGS",
];

fn main() {
    for path in [
        "vllm.cpp/CMakeLists.txt",
        "vllm.cpp/cmake",
        "vllm.cpp/include",
        "vllm.cpp/src",
        "vllm.cpp/third_party/blake3",
        "vllm.cpp/third_party/minja",
        "vllm.cpp/third_party/nlohmann",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
    for name in RERUN_ENV {
        println!("cargo:rerun-if-env-changed={name}");
    }

    if !cfg!(feature = "bundled") {
        panic!("vllm-cpp-sys supports only the default `bundled` feature in this bootstrap");
    }

    let source = Path::new("vllm.cpp");
    if !source.join("CMakeLists.txt").is_file() {
        panic!(
            "vllm.cpp source is missing; initialize it with `git submodule update --init --recursive`"
        );
    }

    let mut config = cmake::Config::new(source);
    config
        .profile("Release")
        .define("VLLM_CPP_BUILD_TESTS", "OFF")
        .define("VLLM_CPP_BUILD_EXAMPLES", "OFF")
        .define("VLLM_CPP_SERVER", "OFF")
        .define("VLLM_CPP_CUDA", "OFF")
        .define("VLLM_CPP_METAL", "OFF")
        .define("VLLM_CPP_VULKAN", "OFF")
        .define("VLLM_CPP_MLX", "OFF")
        .define("VLLM_CPP_TRITON", "OFF")
        .define("VLLM_CPP_TRITON_REGEN", "OFF")
        .define("VLLM_CPP_CUTLASS_FETCH", "OFF")
        .define("VLLM_CPP_SANITIZE", "OFF");

    let install = config.build();
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("cargo always sets OUT_DIR"));
    let installed_lib_dir = find_installed_library_dir(&install);
    let vllm = find_unique_file(&installed_lib_dir, static_library_name("vllm"))
        .unwrap_or_else(|error| panic!("failed to locate installed vllm archive: {error}"));
    let blake3 = find_unique_file(
        &out_dir.join("build"),
        static_library_name("blake3_vendored"),
    )
    .unwrap_or_else(|error| panic!("failed to locate blake3_vendored build archive: {error}"));

    println!(
        "cargo:rustc-link-search=native={}",
        vllm.parent().expect("library has a parent").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        blake3.parent().expect("library has a parent").display()
    );
    println!("cargo:rustc-link-lib=static:+whole-archive=vllm");
    println!("cargo:rustc-link-lib=static=blake3_vendored");

    let target = env::var("CARGO_CFG_TARGET_OS").expect("cargo sets target OS");
    match target.as_str() {
        "linux" => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-lib=dylib=pthread");
            println!("cargo:rustc-link-lib=dylib=dl");
        }
        unsupported => {
            panic!("bundled linking is not implemented for {unsupported} in this bootstrap")
        }
    }
}

fn static_library_name(stem: &str) -> String {
    if cfg!(target_env = "msvc") {
        format!("{stem}.lib")
    } else {
        format!("lib{stem}.a")
    }
}

fn find_installed_library_dir(install: &Path) -> PathBuf {
    for name in ["lib64", "lib"] {
        let candidate = install.join(name);
        if candidate.is_dir() {
            return candidate;
        }
    }
    panic!(
        "cmake did not install a lib or lib64 directory below {}",
        install.display()
    );
}

fn find_unique_file(root: &Path, name: String) -> Result<PathBuf, String> {
    let mut matches = Vec::new();
    collect_named_files(root, &name, &mut matches).map_err(|error| error.to_string())?;
    match matches.as_slice() {
        [path] => Ok(path.clone()),
        [] => Err(format!("{name} was not found below {}", root.display())),
        _ => Err(format!(
            "found multiple {name} files below {}: {}",
            root.display(),
            matches
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

fn collect_named_files(root: &Path, name: &str, matches: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_named_files(&path, name, matches)?;
        } else if file_type.is_file() && entry.file_name() == name {
            matches.push(path);
        }
    }
    Ok(())
}
