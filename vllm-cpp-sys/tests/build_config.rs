#![allow(dead_code)]

#[path = "../src/build_config.rs"]
mod build_config;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use build_config::{
    plan, CudaComponent, Environment, Features, Inputs, LinkRequirement, PathProbe, Target,
    ERROR_PREFIX,
};

#[derive(Default)]
struct MockProbe {
    canonical: BTreeMap<PathBuf, PathBuf>,
    files: BTreeMap<PathBuf, String>,
    dirs: BTreeSet<PathBuf>,
}

impl MockProbe {
    fn cutlass(version: (u32, u32, u32)) -> Self {
        let mut probe = Self::default();
        probe
            .canonical
            .insert(PathBuf::from("cutlass"), PathBuf::from("/cutlass"));
        probe.dirs.insert(PathBuf::from("cutlass"));
        probe.dirs.insert(PathBuf::from("/cutlass"));
        probe
            .dirs
            .insert(PathBuf::from("/cutlass/tools/util/include"));
        probe.files.insert(
            PathBuf::from("/cutlass/include/cutlass/cutlass.h"),
            String::new(),
        );
        probe.files.insert(
            PathBuf::from("/cutlass/include/cutlass/version.h"),
            format!(
                "#define CUTLASS_MAJOR {}\n#define CUTLASS_MINOR {}\n#define CUTLASS_PATCH {}\n",
                version.0, version.1, version.2
            ),
        );
        probe
    }
}

impl PathProbe for MockProbe {
    fn is_file(&self, path: &Path) -> bool {
        self.files.contains_key(path)
    }

    fn is_dir(&self, path: &Path) -> bool {
        self.dirs.contains(path)
    }

    fn canonicalize(&self, path: &Path) -> Result<PathBuf, String> {
        self.canonical
            .get(path)
            .cloned()
            .or_else(|| path.is_absolute().then(|| path.to_path_buf()))
            .ok_or_else(|| "not found".to_owned())
    }

    fn read_to_string(&self, path: &Path) -> Result<String, String> {
        self.files
            .get(path)
            .cloned()
            .ok_or_else(|| "not found".to_owned())
    }
}

fn linux() -> Inputs {
    Inputs {
        features: Features {
            bundled: true,
            ..Features::default()
        },
        target: Target {
            triple: "x86_64-unknown-linux-gnu".to_owned(),
            os: "linux".to_owned(),
            arch: "x86_64".to_owned(),
        },
        environment: Environment::default(),
    }
}

fn error(inputs: &Inputs, probe: &MockProbe) -> String {
    plan(inputs, probe).unwrap_err().to_string()
}

#[test]
fn cpu_and_vulkan_plans_are_deterministic() {
    let cpu = plan(&linux(), &MockProbe::default()).unwrap();
    assert!(cpu
        .cmake_defines
        .contains(&("VLLM_CPP_CUDA", "OFF".to_owned())));
    assert!(cpu
        .cmake_defines
        .contains(&("VLLM_CPP_VULKAN", "OFF".to_owned())));
    assert!(!cpu
        .cmake_defines
        .iter()
        .any(|(key, _)| *key == "VLLM_CPP_CUDA_ARCHITECTURES"));
    assert_eq!(
        cpu.link_requirements,
        vec![
            LinkRequirement::Library("stdc++"),
            LinkRequirement::Library("pthread"),
            LinkRequirement::Library("dl"),
        ]
    );

    let mut vulkan = linux();
    vulkan.features.vulkan = true;
    let first = plan(&vulkan, &MockProbe::default()).unwrap();
    let second = plan(&vulkan, &MockProbe::default()).unwrap();
    assert_eq!(first, second);
    assert!(first
        .cmake_defines
        .contains(&("VLLM_CPP_VULKAN", "ON".to_owned())));
}

#[test]
fn accepts_exact_cuda_architecture_spellings() {
    for architecture in [
        "80",
        "86",
        "87",
        "89",
        "90a",
        "100a",
        "103a",
        "110",
        "120a",
        "121a",
        "120a;121a",
    ] {
        let mut inputs = linux();
        inputs.features.cuda = true;
        inputs.environment.cuda_architectures = Some(architecture.to_owned());
        let plan = plan(&inputs, &MockProbe::default()).unwrap();
        assert!(plan
            .cmake_defines
            .contains(&("VLLM_CPP_CUDA_ARCHITECTURES", architecture.to_owned())));
    }

    for invalid in ["", "75", "90", "121", "121a;120a", "80;86", " 80"] {
        let mut inputs = linux();
        inputs.features.cuda = true;
        inputs.environment.cuda_architectures = Some(invalid.to_owned());
        assert!(error(&inputs, &MockProbe::default()).contains("unsupported"));
    }
}

#[test]
fn cuda_requires_arch_and_supported_target() {
    let mut inputs = linux();
    inputs.features.cuda = true;
    assert!(error(&inputs, &MockProbe::default()).contains("requires an exact"));

    inputs.environment.cuda_architectures = Some("80".to_owned());
    inputs.target = Target {
        triple: "x86_64-pc-windows-msvc".to_owned(),
        os: "windows".to_owned(),
        arch: "x86_64".to_owned(),
    };
    assert!(error(&inputs, &MockProbe::default()).contains("only for Linux"));
}

#[test]
fn rejects_source_mode_and_backend_conflicts() {
    let mut both = linux();
    both.features.system = true;
    assert!(error(&both, &MockProbe::default()).contains("mutually exclusive"));

    let mut neither = linux();
    neither.features.bundled = false;
    assert!(error(&neither, &MockProbe::default()).contains("exactly one"));

    let mut system_cuda = linux();
    system_cuda.features.bundled = false;
    system_cuda.features.system = true;
    system_cuda.features.cuda = true;
    assert!(error(&system_cuda, &MockProbe::default()).contains("bundled-only"));

    let mut cuda_vulkan = linux();
    cuda_vulkan.features.cuda = true;
    cuda_vulkan.features.vulkan = true;
    assert!(error(&cuda_vulkan, &MockProbe::default()).contains("cannot be combined"));
}

#[test]
fn rejects_broken_feature_implications() {
    let mut cutlass = linux();
    cutlass.features.cuda_cutlass = true;
    assert!(error(&cutlass, &MockProbe::default()).contains("requires the `cuda`"));

    let mut triton = linux();
    triton.features.triton_aot = true;
    assert!(error(&triton, &MockProbe::default()).contains("requires the `cuda`"));
}

#[test]
fn validates_cutlass_tree_version_architecture_and_canonical_root() {
    let mut inputs = linux();
    inputs.features.cuda = true;
    inputs.features.cuda_cutlass = true;
    inputs.environment.cuda_architectures = Some("80".to_owned());
    assert!(error(&inputs, &MockProbe::default()).contains("VLLM_CPP_CUTLASS_DIR"));

    inputs.environment.cutlass_dir = Some(PathBuf::from("cutlass"));
    assert!(error(&inputs, &MockProbe::default()).contains("not an existing directory"));
    assert!(error(&inputs, &MockProbe::cutlass((4, 4, 2))).contains(">=4.5.0"));

    let plan = plan(&inputs, &MockProbe::cutlass((4, 5, 0))).unwrap();
    assert!(plan
        .cmake_defines
        .contains(&("VLLM_CPP_CUTLASS_DIR", "/cutlass".to_owned())));

    for unsupported in ["103a", "110"] {
        inputs.environment.cuda_architectures = Some(unsupported.to_owned());
        assert!(error(&inputs, &MockProbe::cutlass((4, 6, 1))).contains("no enabled"));
    }
}

#[test]
fn cutlass_requires_all_paths_and_parseable_version() {
    let mut inputs = linux();
    inputs.features.cuda = true;
    inputs.features.cuda_cutlass = true;
    inputs.environment.cuda_architectures = Some("121a".to_owned());
    inputs.environment.cutlass_dir = Some(PathBuf::from("cutlass"));

    let mut probe = MockProbe::cutlass((4, 5, 0));
    probe
        .files
        .remove(Path::new("/cutlass/include/cutlass/cutlass.h"));
    assert!(error(&inputs, &probe).contains("cutlass.h"));

    let mut probe = MockProbe::cutlass((4, 5, 0));
    probe.dirs.remove(Path::new("/cutlass/tools/util/include"));
    assert!(error(&inputs, &probe).contains("tools/util/include"));

    let mut probe = MockProbe::cutlass((4, 5, 0));
    probe.files.insert(
        PathBuf::from("/cutlass/include/cutlass/version.h"),
        "not a version".to_owned(),
    );
    assert!(error(&inputs, &probe).contains("could not parse"));
}

#[test]
fn static_cuda_links_exact_components_while_dynamic_relies_on_dt_needed() {
    let mut inputs = linux();
    inputs.features.cuda = true;
    inputs.features.triton_aot = true;
    inputs.environment.cuda_architectures = Some("80".to_owned());

    let static_plan = plan(&inputs, &MockProbe::default()).unwrap();
    assert_eq!(
        static_plan.link_requirements,
        vec![
            LinkRequirement::Library("stdc++"),
            LinkRequirement::Library("pthread"),
            LinkRequirement::Library("dl"),
            LinkRequirement::CudaToolkit(CudaComponent::Runtime),
            LinkRequirement::CudaToolkit(CudaComponent::BlasLt),
            LinkRequirement::CudaToolkit(CudaComponent::Driver),
        ]
    );

    inputs.features.dynamic_link = true;
    let dynamic = plan(&inputs, &MockProbe::default()).unwrap();
    assert!(dynamic.link_requirements.is_empty());
}

#[test]
fn cuda_cache_keys_match_find_cudatoolkit() {
    assert_eq!(CudaComponent::Runtime.cmake_cache_key(), "CUDA_CUDART");
    assert_eq!(
        CudaComponent::BlasLt.cmake_cache_key(),
        "CUDA_cublasLt_LIBRARY"
    );
    assert_eq!(
        CudaComponent::Driver.cmake_cache_key(),
        "CUDA_cuda_driver_LIBRARY"
    );
    assert_eq!(CudaComponent::Runtime.cargo_library(), "cudart");
    assert_eq!(CudaComponent::BlasLt.cargo_library(), "cublasLt");
    assert_eq!(CudaComponent::Driver.cargo_library(), "cuda");
}

#[test]
fn triton_accepts_only_vendored_single_architectures() {
    for architecture in ["80", "86", "89", "90a", "100a", "121a"] {
        let mut inputs = linux();
        inputs.features.cuda = true;
        inputs.features.triton_aot = true;
        inputs.environment.cuda_architectures = Some(architecture.to_owned());
        let plan = plan(&inputs, &MockProbe::default()).unwrap();
        assert!(plan
            .cmake_defines
            .contains(&("VLLM_CPP_TRITON", "ON".to_owned())));
        assert!(plan
            .cmake_defines
            .contains(&("VLLM_CPP_TRITON_REGEN", "OFF".to_owned())));
    }

    for invalid in ["87", "103a", "110", "120a", "120a;121a"] {
        let mut inputs = linux();
        inputs.features.cuda = true;
        inputs.features.triton_aot = true;
        inputs.environment.cuda_architectures = Some(invalid.to_owned());
        assert!(error(&inputs, &MockProbe::default()).contains("vendored single"));
    }
}

#[test]
fn rejects_cuda_sanitizers_and_unsupported_vulkan_targets() {
    let mut inputs = linux();
    inputs.features.cuda = true;
    inputs.environment.cuda_architectures = Some("80".to_owned());
    inputs.environment.sanitizer = Some("address,undefined".to_owned());
    assert!(error(&inputs, &MockProbe::default()).contains("compute-sanitizer"));

    let mut vulkan = linux();
    vulkan.features.vulkan = true;
    vulkan.target.arch = "riscv64".to_owned();
    vulkan.target.triple = "riscv64gc-unknown-linux-gnu".to_owned();
    assert!(error(&vulkan, &MockProbe::default()).contains("Linux x86_64/aarch64"));
}

#[test]
fn every_error_has_the_actionable_prefix() {
    let mut inputs = linux();
    inputs.features.cuda = true;
    assert!(error(&inputs, &MockProbe::default()).starts_with(ERROR_PREFIX));
}
