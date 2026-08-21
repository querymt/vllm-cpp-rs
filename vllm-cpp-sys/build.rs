#[path = "src/build_config.rs"]
mod build_config;
#[path = "src/build_support.rs"]
mod build_support;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use build_config::{
    BuildPlan, CudaComponent, Environment, Features, FsProbe, Inputs, LinkRequirement, Target,
};
use build_support::{
    cmake_cache_path, find_installed_library_dir, require_library_file, shared_library_name,
};

const RERUN_ENV: &[&str] = &[
    "CMAKE_BUILD_PARALLEL_LEVEL",
    "CMAKE_GENERATOR",
    "CMAKE_GENERATOR_PLATFORM",
    "CMAKE_GENERATOR_TOOLSET",
    "CMAKE_PREFIX_PATH",
    "VLLM_CPP_ROOT",
    "VLLM_CPP_LIB_DIR",
    "VLLM_CPP_BLAKE3_LIB_DIR",
    "VLLM_CPP_CUDA_ARCHITECTURES",
    "VLLM_CPP_CUTLASS_DIR",
    "VLLM_CPP_SANITIZE",
    "CUDA_PATH",
    "CUDA_HOME",
    "CUDAToolkit_ROOT",
    "CC",
    "CFLAGS",
    "CXX",
    "CXXFLAGS",
];

fn main() {
    for path in [
        "build.rs",
        "src/build_config.rs",
        "src/build_support.rs",
        "vllm.cpp/CMakeLists.txt",
        "vllm.cpp/cmake",
        "vllm.cpp/include/vllm.h",
        "vllm.cpp/src",
        "vllm.cpp/triton_kernels",
        "vllm.cpp/scripts/triton-aot-compile.py",
        "vllm.cpp/third_party/blake3",
        "vllm.cpp/third_party/minja",
        "vllm.cpp/third_party/nlohmann",
        "vllm.cpp/third_party/vulkan",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
    for name in RERUN_ENV {
        println!("cargo:rerun-if-env-changed={name}");
    }

    if env::var_os("DOCS_RS").is_some() {
        return;
    }

    let inputs = build_inputs();
    let plan = build_config::plan(&inputs, &FsProbe).unwrap_or_else(|error| panic!("{error}"));
    let sanitizer = sanitizer_config(inputs.features.bundled);
    let system_root = inputs.features.system.then(validate_system_root);
    let cmake_cache = if inputs.features.bundled {
        Some(build_bundled(&plan))
    } else {
        link_system(system_root.as_deref().expect("system root was validated"));
        None
    };
    emit_link_requirements(&plan, cmake_cache.as_deref());
    link_sanitizer_runtimes(sanitizer.as_deref());
}

fn build_inputs() -> Inputs {
    Inputs {
        features: Features {
            bundled: cfg!(feature = "bundled"),
            system: cfg!(feature = "system"),
            dynamic_link: cfg!(feature = "dynamic-link"),
            cuda: cfg!(feature = "cuda"),
            cuda_cutlass: cfg!(feature = "cuda-cutlass"),
            triton_aot: cfg!(feature = "triton-aot"),
            vulkan: cfg!(feature = "vulkan"),
        },
        target: Target {
            triple: required_env("TARGET"),
            os: required_env("CARGO_CFG_TARGET_OS"),
            arch: required_env("CARGO_CFG_TARGET_ARCH"),
        },
        environment: Environment {
            cuda_architectures: env::var("VLLM_CPP_CUDA_ARCHITECTURES").ok(),
            cutlass_dir: env::var_os("VLLM_CPP_CUTLASS_DIR").map(PathBuf::from),
            sanitizer: env::var("VLLM_CPP_SANITIZE").ok(),
        },
    }
}

fn build_bundled(plan: &BuildPlan) -> PathBuf {
    let source = Path::new("vllm.cpp");
    if !source.join("CMakeLists.txt").is_file() {
        config_error(
            "vllm.cpp source is missing; initialize it with `git submodule update --init --recursive`",
        );
    }

    let out_dir = PathBuf::from(required_env("OUT_DIR"));
    let mut config = cmake::Config::new(source);
    config.profile("Release");
    for (name, value) in &plan.cmake_defines {
        config.define(name, value);
    }
    if !cfg!(feature = "cuda-cutlass") {
        config.define(
            "VLLM_CPP_CUTLASS_DIR",
            out_dir.join("disabled-cutlass-feature"),
        );
    }

    let dynamic_link = cfg!(feature = "dynamic-link");
    let vllm_artifact = if dynamic_link {
        shared_library_name("vllm")
    } else {
        static_library_name("vllm")
    };

    let install = config.build();
    let installed_lib_dir =
        find_installed_library_dir(&install, &vllm_artifact).unwrap_or_else(|error| {
            config_error(format!(
                "failed to select bundled library directory: {error}"
            ))
        });
    let vllm = require_library_file(
        &installed_lib_dir,
        &vllm_artifact,
        if dynamic_link {
            "bundled dynamic vllm"
        } else {
            "bundled static vllm"
        },
    )
    .unwrap_or_else(|error| config_error(error));
    println!("cargo:rerun-if-changed={}", vllm.display());
    println!(
        "cargo:rustc-link-search=native={}",
        installed_lib_dir.display()
    );

    if dynamic_link {
        println!("cargo:rustc-link-lib=dylib=vllm");
    } else {
        let blake3 = find_unique_file(
            &out_dir.join("build"),
            static_library_name("blake3_vendored"),
        )
        .unwrap_or_else(|error| {
            config_error(format!(
                "failed to locate blake3_vendored build archive: {error}"
            ))
        });
        println!("cargo:rerun-if-changed={}", blake3.display());
        println!(
            "cargo:rustc-link-search=native={}",
            blake3.parent().expect("library has a parent").display()
        );
        println!("cargo:rustc-link-lib=static:+whole-archive=vllm");
        println!("cargo:rustc-link-lib=static=blake3_vendored");
    }

    let cache = out_dir.join("build/CMakeCache.txt");
    if !cache.is_file() {
        config_error(format!(
            "CMake did not produce the expected cache at {}",
            cache.display()
        ));
    }
    println!("cargo:rerun-if-changed={}", cache.display());
    cache
}

fn validate_system_root() -> PathBuf {
    let root = required_path("VLLM_CPP_ROOT");
    let header = root.join("include/vllm.h");
    if !header.is_file() {
        config_error(format!(
            "VLLM_CPP_ROOT must contain include/vllm.h; missing {}",
            header.display()
        ));
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
    .unwrap_or_else(|error| config_error(error));
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
        config_error(format!(
            "VLLM_CPP_BLAKE3_LIB_DIR does not exist: {}",
            blake3_lib_dir.display()
        ));
    }
    let blake3 = require_library_file(
        &blake3_lib_dir,
        &static_library_name("blake3_vendored"),
        "system static blake3_vendored; provision it separately and set VLLM_CPP_BLAKE3_LIB_DIR, or use `system,dynamic-link`",
    )
    .unwrap_or_else(|error| config_error(error));
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
}

fn system_library_dir(root: &Path, expected_artifact: &str) -> PathBuf {
    if let Some(lib_dir) = env::var_os("VLLM_CPP_LIB_DIR").map(PathBuf::from) {
        if !lib_dir.is_dir() {
            config_error(format!(
                "VLLM_CPP_LIB_DIR does not exist: {}",
                lib_dir.display()
            ));
        }
        require_library_file(&lib_dir, expected_artifact, "VLLM_CPP_LIB_DIR override")
            .unwrap_or_else(|error| config_error(error));
        return lib_dir;
    }

    find_installed_library_dir(root, expected_artifact).unwrap_or_else(|error| {
        config_error(format!(
            "failed to locate {expected_artifact} under {}: {error}",
            root.display()
        ))
    })
}

fn emit_link_requirements(plan: &BuildPlan, cmake_cache: Option<&Path>) {
    for requirement in &plan.link_requirements {
        match requirement {
            LinkRequirement::Library(name) => {
                println!("cargo:rustc-link-lib=dylib={name}");
            }
            LinkRequirement::CudaToolkit(component) => {
                let cache = cmake_cache.unwrap_or_else(|| {
                    config_error(
                        "internal error: a CUDA link requirement has no bundled CMake cache",
                    )
                });
                link_cuda_component(cache, *component);
            }
        }
    }
}

fn link_cuda_component(cache: &Path, component: CudaComponent) {
    let library = cmake_cache_path(cache, component.cmake_cache_key()).unwrap_or_else(|error| {
        config_error(format!(
            "failed to resolve CUDA component {} from CMake's CUDAToolkit result: {error}",
            component.cargo_library()
        ))
    });
    if !library.is_file() {
        config_error(format!(
            "CMake selected CUDA component {} at {}, but that file does not exist",
            component.cargo_library(),
            library.display()
        ));
    }
    println!("cargo:rerun-if-changed={}", library.display());
    println!(
        "cargo:rustc-link-search=native={}",
        library.parent().expect("library has a parent").display()
    );
    println!("cargo:rustc-link-lib=dylib={}", component.cargo_library());
}

fn sanitizer_config(bundled: bool) -> Option<String> {
    let value = env::var_os("VLLM_CPP_SANITIZE")?;
    let value = value
        .into_string()
        .unwrap_or_else(|_| config_error("VLLM_CPP_SANITIZE must be valid UTF-8"));
    if value == "OFF" {
        return None;
    }
    if !bundled {
        config_error("VLLM_CPP_SANITIZE is supported only for bundled builds");
    }
    match value.as_str() {
        "address" | "undefined" | "address,undefined" | "thread" => Some(value),
        _ => config_error(format!(
            "unsupported VLLM_CPP_SANITIZE value `{value}`; expected OFF, address, undefined, address,undefined, or thread"
        )),
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

fn required_env(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| config_error(format!("Cargo did not set required {name}")))
}

fn required_path(name: &str) -> PathBuf {
    env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| config_error(format!("{name} must be set for `system` mode")))
}

fn config_error(message: impl AsRef<str>) -> ! {
    panic!("{} {}", build_config::ERROR_PREFIX, message.as_ref())
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
