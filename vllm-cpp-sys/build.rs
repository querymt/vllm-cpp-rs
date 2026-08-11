#[path = "src/build_support.rs"]
mod build_support;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use build_support::{find_installed_library_dir, require_library_file, shared_library_name};

const RERUN_ENV: &[&str] = &[
    "CMAKE_BUILD_PARALLEL_LEVEL",
    "CMAKE_GENERATOR",
    "CMAKE_GENERATOR_PLATFORM",
    "CMAKE_GENERATOR_TOOLSET",
    "CMAKE_PREFIX_PATH",
    "VLLM_CPP_ROOT",
    "VLLM_CPP_LIB_DIR",
    "VLLM_CPP_BLAKE3_LIB_DIR",
    "VLLM_CPP_SANITIZE",
    "CC",
    "CFLAGS",
    "CXX",
    "CXXFLAGS",
];

fn main() {
    for path in [
        "vllm.cpp/CMakeLists.txt",
        "vllm.cpp/cmake",
        "vllm.cpp/include/vllm.h",
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

    if env::var_os("DOCS_RS").is_some() {
        return;
    }

    let bundled = cfg!(feature = "bundled");
    let system = cfg!(feature = "system");
    match (bundled, system) {
        (true, true) => panic!("features `bundled` and `system` are mutually exclusive"),
        (false, false) => panic!("enable exactly one of the `bundled` or `system` features"),
        _ => {}
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("cargo sets target OS");
    if target_os != "linux" {
        panic!("linking is implemented only for Linux, not {target_os}");
    }

    let sanitizer = sanitizer_config(bundled);
    let system_root = system.then(validate_system_root);
    if bundled {
        build_bundled(sanitizer.as_deref());
    } else {
        link_system(system_root.as_deref().expect("system root was validated"));
    }
    link_sanitizer_runtimes(sanitizer.as_deref());
}

fn build_bundled(sanitizer: Option<&str>) {
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
        .define("VLLM_CPP_SANITIZE", sanitizer.unwrap_or("OFF"));

    let dynamic_link = cfg!(feature = "dynamic-link");
    let vllm_artifact = if dynamic_link {
        shared_library_name("vllm")
    } else {
        static_library_name("vllm")
    };

    let install = config.build();
    let installed_lib_dir = find_installed_library_dir(&install, &vllm_artifact)
        .unwrap_or_else(|error| panic!("failed to select bundled library directory: {error}"));
    let vllm = require_library_file(
        &installed_lib_dir,
        &vllm_artifact,
        if dynamic_link {
            "bundled dynamic vllm"
        } else {
            "bundled static vllm"
        },
    )
    .unwrap_or_else(|error| panic!("{error}"));
    println!("cargo:rerun-if-changed={}", vllm.display());
    println!(
        "cargo:rustc-link-search=native={}",
        installed_lib_dir.display()
    );

    if dynamic_link {
        println!("cargo:rustc-link-lib=dylib=vllm");
        return;
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("cargo always sets OUT_DIR"));
    let blake3 = find_unique_file(
        &out_dir.join("build"),
        static_library_name("blake3_vendored"),
    )
    .unwrap_or_else(|error| panic!("failed to locate blake3_vendored build archive: {error}"));
    println!("cargo:rerun-if-changed={}", blake3.display());
    println!(
        "cargo:rustc-link-search=native={}",
        blake3.parent().expect("library has a parent").display()
    );
    println!("cargo:rustc-link-lib=static:+whole-archive=vllm");
    println!("cargo:rustc-link-lib=static=blake3_vendored");
    link_platform_dependencies();
}

fn validate_system_root() -> PathBuf {
    let root = required_path("VLLM_CPP_ROOT");
    let header = root.join("include/vllm.h");
    if !header.is_file() {
        panic!(
            "VLLM_CPP_ROOT must contain include/vllm.h; missing {}",
            header.display()
        );
    }
    println!("cargo:rerun-if-changed={}", header.display());
    root
}

fn link_system(root: &Path) {
    let dynamic_link = cfg!(feature = "dynamic-link");
    let vllm_artifact = if dynamic_link {
        shared_library_name("vllm")
    } else {
        static_library_name("vllm")
    };
    let lib_dir = system_library_dir(root, &vllm_artifact);
    let vllm = require_library_file(
        &lib_dir,
        &vllm_artifact,
        if dynamic_link {
            "system dynamic vllm"
        } else {
            "system static vllm"
        },
    )
    .unwrap_or_else(|error| panic!("{error}"));
    println!("cargo:rerun-if-changed={}", vllm.display());

    if dynamic_link {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=dylib=vllm");
        return;
    }

    let blake3_lib_dir = env::var_os("VLLM_CPP_BLAKE3_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| lib_dir.clone());
    if !blake3_lib_dir.is_dir() {
        panic!(
            "VLLM_CPP_BLAKE3_LIB_DIR does not exist: {}",
            blake3_lib_dir.display()
        );
    }
    let blake3 = require_library_file(
        &blake3_lib_dir,
        &static_library_name("blake3_vendored"),
        "system static blake3_vendored; provision it separately and set VLLM_CPP_BLAKE3_LIB_DIR, or use `system,dynamic-link`",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    println!("cargo:rerun-if-changed={}", blake3.display());
    println!(
        "cargo:rustc-link-search=native={}",
        blake3_lib_dir.display()
    );
    if blake3_lib_dir != lib_dir {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }
    println!("cargo:rustc-link-lib=static:+whole-archive=vllm");
    println!("cargo:rustc-link-lib=static=blake3_vendored");
    link_platform_dependencies();
}

fn system_library_dir(root: &Path, expected_artifact: &str) -> PathBuf {
    if let Some(lib_dir) = env::var_os("VLLM_CPP_LIB_DIR").map(PathBuf::from) {
        if !lib_dir.is_dir() {
            panic!("VLLM_CPP_LIB_DIR does not exist: {}", lib_dir.display());
        }
        require_library_file(&lib_dir, expected_artifact, "VLLM_CPP_LIB_DIR override")
            .unwrap_or_else(|error| panic!("{error}"));
        return lib_dir;
    }

    find_installed_library_dir(root, expected_artifact).unwrap_or_else(|error| {
        panic!(
            "failed to locate {expected_artifact} under {}: {error}",
            root.display()
        )
    })
}

fn sanitizer_config(bundled: bool) -> Option<String> {
    let value = env::var_os("VLLM_CPP_SANITIZE")?;
    let value = value
        .into_string()
        .unwrap_or_else(|_| panic!("VLLM_CPP_SANITIZE must be valid UTF-8"));
    if value == "OFF" {
        return None;
    }
    if !bundled {
        panic!("VLLM_CPP_SANITIZE is supported only for bundled builds");
    }
    match value.as_str() {
        "address" | "undefined" | "address,undefined" | "thread" => Some(value),
        _ => panic!(
            "unsupported VLLM_CPP_SANITIZE value `{value}`; expected OFF, address, undefined, address,undefined, or thread"
        ),
    }
}

fn link_sanitizer_runtimes(sanitizer: Option<&str>) {
    let Some(sanitizer) = sanitizer else {
        return;
    };
    if sanitizer.split(',').any(|name| name == "address") {
        println!("cargo:rustc-link-lib=dylib=asan");
    }
    if sanitizer.split(',').any(|name| name == "undefined") {
        println!("cargo:rustc-link-lib=dylib=ubsan");
    }
    if sanitizer.split(',').any(|name| name == "thread") {
        println!("cargo:rustc-link-lib=dylib=tsan");
    }
}

fn link_platform_dependencies() {
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=dl");
}

fn required_path(name: &str) -> PathBuf {
    env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} must be set when the `system` feature is enabled"))
}

fn static_library_name(stem: &str) -> String {
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        format!("{stem}.lib")
    } else {
        format!("lib{stem}.a")
    }
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
