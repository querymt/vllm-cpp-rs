use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

pub const ERROR_PREFIX: &str = "vllm-cpp-sys build configuration error:";

const CUDA_ARCHITECTURES: &[&str] = &[
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
];
const TRITON_ARCHITECTURES: &[&str] = &["80", "86", "89", "90a", "100a", "121a"];

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Features {
    pub bundled: bool,
    pub system: bool,
    pub dynamic_link: bool,
    pub cuda: bool,
    pub cuda_cutlass: bool,
    pub triton_aot: bool,
    pub vulkan: bool,
}

impl Features {
    fn has_backend(&self) -> bool {
        self.cuda || self.cuda_cutlass || self.triton_aot || self.vulkan
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Target {
    pub triple: String,
    pub os: String,
    pub arch: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Environment {
    pub cuda_architectures: Option<String>,
    pub cutlass_dir: Option<PathBuf>,
    pub sanitizer: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Inputs {
    pub features: Features,
    pub target: Target,
    pub environment: Environment,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LinkRequirement {
    Library(&'static str),
    CudaToolkit(CudaComponent),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaComponent {
    Runtime,
    BlasLt,
    Driver,
}

impl CudaComponent {
    pub fn cmake_cache_key(self) -> &'static str {
        match self {
            Self::Runtime => "CUDA_CUDART",
            Self::BlasLt => "CUDA_cublasLt_LIBRARY",
            Self::Driver => "CUDA_cuda_driver_LIBRARY",
        }
    }

    pub fn cargo_library(self) -> &'static str {
        match self {
            Self::Runtime => "cudart",
            Self::BlasLt => "cublasLt",
            Self::Driver => "cuda",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuildPlan {
    pub cmake_defines: Vec<(&'static str, String)>,
    pub link_requirements: Vec<LinkRequirement>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfigError(String);

impl ConfigError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{ERROR_PREFIX} {}", self.0)
    }
}

impl std::error::Error for ConfigError {}

pub trait PathProbe {
    fn is_file(&self, path: &Path) -> bool;
    fn is_dir(&self, path: &Path) -> bool;
    fn canonicalize(&self, path: &Path) -> Result<PathBuf, String>;
    fn read_to_string(&self, path: &Path) -> Result<String, String>;
}

pub struct FsProbe;

impl PathProbe for FsProbe {
    fn is_file(&self, path: &Path) -> bool {
        path.is_file()
    }

    fn is_dir(&self, path: &Path) -> bool {
        path.is_dir()
    }

    fn canonicalize(&self, path: &Path) -> Result<PathBuf, String> {
        fs::canonicalize(path).map_err(|error| error.to_string())
    }

    fn read_to_string(&self, path: &Path) -> Result<String, String> {
        fs::read_to_string(path).map_err(|error| error.to_string())
    }
}

pub fn plan(inputs: &Inputs, probe: &impl PathProbe) -> Result<BuildPlan, ConfigError> {
    validate_source_mode(&inputs.features)?;
    validate_feature_implications(&inputs.features)?;
    validate_targets(inputs)?;

    let cuda_architectures = validate_cuda(inputs)?;
    let cutlass_dir = if inputs.features.cuda_cutlass {
        Some(validate_cutlass(
            inputs,
            probe,
            cuda_architectures.as_deref(),
        )?)
    } else {
        None
    };

    let sanitizer = inputs.environment.sanitizer.as_deref().unwrap_or("OFF");
    if inputs.features.cuda && sanitizer != "OFF" {
        return Err(ConfigError::new(format!(
            "VLLM_CPP_SANITIZE={sanitizer} cannot be combined with `cuda`; use compute-sanitizer for CUDA or set VLLM_CPP_SANITIZE=OFF"
        )));
    }

    let on_off = |enabled: bool| if enabled { "ON" } else { "OFF" }.to_owned();
    let mut cmake_defines = vec![
        ("VLLM_CPP_BUILD_TESTS", "OFF".to_owned()),
        ("VLLM_CPP_BUILD_EXAMPLES", "OFF".to_owned()),
        ("VLLM_CPP_SERVER", "OFF".to_owned()),
        ("VLLM_CPP_CUDA", on_off(inputs.features.cuda)),
        ("VLLM_CPP_METAL", "OFF".to_owned()),
        ("VLLM_CPP_VULKAN", on_off(inputs.features.vulkan)),
        ("VLLM_CPP_MLX", "OFF".to_owned()),
        ("VLLM_CPP_TRITON", on_off(inputs.features.triton_aot)),
        ("VLLM_CPP_TRITON_REGEN", "OFF".to_owned()),
        ("VLLM_CPP_CUTLASS_FETCH", "OFF".to_owned()),
        ("VLLM_CPP_SANITIZE", sanitizer.to_owned()),
    ];
    if let Some(architectures) = cuda_architectures {
        cmake_defines.push(("VLLM_CPP_CUDA_ARCHITECTURES", architectures));
    }
    if let Some(root) = cutlass_dir {
        cmake_defines.push(("VLLM_CPP_CUTLASS_DIR", root.to_string_lossy().into_owned()));
    }

    let mut link_requirements = platform_link_requirements(inputs)?;
    if inputs.features.cuda && !inputs.features.dynamic_link {
        link_requirements.extend([
            LinkRequirement::CudaToolkit(CudaComponent::Runtime),
            LinkRequirement::CudaToolkit(CudaComponent::BlasLt),
        ]);
        if inputs.features.triton_aot {
            link_requirements.push(LinkRequirement::CudaToolkit(CudaComponent::Driver));
        }
    }

    Ok(BuildPlan {
        cmake_defines,
        link_requirements,
    })
}

fn validate_source_mode(features: &Features) -> Result<(), ConfigError> {
    match (features.bundled, features.system) {
        (true, true) => Err(ConfigError::new(
            "features `bundled` and `system` are mutually exclusive; disable default features when selecting `system`",
        )),
        (false, false) => Err(ConfigError::new(
            "enable exactly one of the `bundled` or `system` features",
        )),
        (false, true) if features.has_backend() => Err(ConfigError::new(
            "accelerator features are bundled-only and cannot be combined with `system`; build the system library with its own backend configuration and select only `system`",
        )),
        _ => Ok(()),
    }
}

fn validate_feature_implications(features: &Features) -> Result<(), ConfigError> {
    if features.cuda_cutlass && !features.cuda {
        return Err(ConfigError::new(
            "`cuda-cutlass` requires the `cuda` feature",
        ));
    }
    if features.triton_aot && !features.cuda {
        return Err(ConfigError::new("`triton-aot` requires the `cuda` feature"));
    }
    if features.cuda && features.vulkan {
        return Err(ConfigError::new(
            "`cuda` and `vulkan` cannot be combined in release 0.1; build separate backend artifacts",
        ));
    }
    Ok(())
}

fn validate_targets(inputs: &Inputs) -> Result<(), ConfigError> {
    if inputs.target.os != "linux" {
        return Err(ConfigError::new(format!(
            "linking is implemented only for Linux, not target {}",
            inputs.target.triple
        )));
    }
    if (inputs.features.cuda || inputs.features.vulkan)
        && !matches!(inputs.target.arch.as_str(), "x86_64" | "aarch64")
    {
        let feature = if inputs.features.cuda {
            "cuda"
        } else {
            "vulkan"
        };
        return Err(ConfigError::new(format!(
            "`{feature}` supports only Linux x86_64/aarch64 targets; target {} is unsupported",
            inputs.target.triple
        )));
    }
    Ok(())
}

fn validate_cuda(inputs: &Inputs) -> Result<Option<String>, ConfigError> {
    if !inputs.features.cuda {
        if inputs.environment.cuda_architectures.is_some() {
            return Err(ConfigError::new(
                "VLLM_CPP_CUDA_ARCHITECTURES is set but the `cuda` feature is disabled; remove it or enable `cuda`",
            ));
        }
        return Ok(None);
    }

    let architectures = inputs
        .environment
        .cuda_architectures
        .as_deref()
        .ok_or_else(|| {
            ConfigError::new(format!(
                "the `cuda` feature requires an exact VLLM_CPP_CUDA_ARCHITECTURES value: {}",
                CUDA_ARCHITECTURES.join(", ")
            ))
        })?;
    if !CUDA_ARCHITECTURES.contains(&architectures) {
        return Err(ConfigError::new(format!(
            "unsupported VLLM_CPP_CUDA_ARCHITECTURES={architectures:?}; supported values are {}",
            CUDA_ARCHITECTURES.join(", ")
        )));
    }
    if inputs.features.triton_aot && !TRITON_ARCHITECTURES.contains(&architectures) {
        return Err(ConfigError::new(format!(
            "`triton-aot` requires one vendored single architecture: {}; got {architectures:?}",
            TRITON_ARCHITECTURES.join(", ")
        )));
    }
    Ok(Some(architectures.to_owned()))
}

fn validate_cutlass(
    inputs: &Inputs,
    probe: &impl PathProbe,
    cuda_architectures: Option<&str>,
) -> Result<PathBuf, ConfigError> {
    let root = inputs.environment.cutlass_dir.as_deref().ok_or_else(|| {
        ConfigError::new(
            "`cuda-cutlass` requires VLLM_CPP_CUTLASS_DIR pointing to an existing CUTLASS >=4.5.0 checkout; it is never fetched",
        )
    })?;
    if !probe.is_dir(root) {
        return Err(ConfigError::new(format!(
            "VLLM_CPP_CUTLASS_DIR={} is not an existing directory",
            root.display()
        )));
    }
    let canonical = probe.canonicalize(root).map_err(|error| {
        ConfigError::new(format!(
            "failed to canonicalize VLLM_CPP_CUTLASS_DIR={}: {error}",
            root.display()
        ))
    })?;
    if !canonical.is_absolute() {
        return Err(ConfigError::new(format!(
            "VLLM_CPP_CUTLASS_DIR must resolve to an absolute path; got {}",
            canonical.display()
        )));
    }

    for relative in ["include/cutlass/cutlass.h", "include/cutlass/version.h"] {
        let path = canonical.join(relative);
        if !probe.is_file(&path) {
            return Err(ConfigError::new(format!(
                "VLLM_CPP_CUTLASS_DIR must contain {relative}; missing {}",
                path.display()
            )));
        }
    }
    let tools = canonical.join("tools/util/include");
    if !probe.is_dir(&tools) {
        return Err(ConfigError::new(format!(
            "VLLM_CPP_CUTLASS_DIR must contain tools/util/include; missing {}",
            tools.display()
        )));
    }

    let version_path = canonical.join("include/cutlass/version.h");
    let contents = probe.read_to_string(&version_path).map_err(|error| {
        ConfigError::new(format!(
            "failed to read CUTLASS version from {}: {error}",
            version_path.display()
        ))
    })?;
    let version = parse_cutlass_version(&contents).ok_or_else(|| {
        ConfigError::new(format!(
            "could not parse CUTLASS_MAJOR, CUTLASS_MINOR, and CUTLASS_PATCH from {}",
            version_path.display()
        ))
    })?;
    if version < (4, 5, 0) {
        return Err(ConfigError::new(format!(
            "CUTLASS >=4.5.0 is required; found {}.{}.{} at {}",
            version.0,
            version.1,
            version.2,
            canonical.display()
        )));
    }
    if matches!(cuda_architectures, Some("103a" | "110")) {
        return Err(ConfigError::new(format!(
            "`cuda-cutlass` has no enabled upstream kernel for CUDA architecture {}; remove `cuda-cutlass` for this portable-kernel build",
            cuda_architectures.expect("matched Some")
        )));
    }
    Ok(canonical)
}

fn parse_cutlass_version(contents: &str) -> Option<(u32, u32, u32)> {
    fn value(contents: &str, name: &str) -> Option<u32> {
        contents.lines().find_map(|line| {
            let mut words = line.split_whitespace();
            (words.next()? == "#define" && words.next()? == name)
                .then(|| words.next()?.parse().ok())?
        })
    }

    Some((
        value(contents, "CUTLASS_MAJOR")?,
        value(contents, "CUTLASS_MINOR")?,
        value(contents, "CUTLASS_PATCH")?,
    ))
}

fn platform_link_requirements(inputs: &Inputs) -> Result<Vec<LinkRequirement>, ConfigError> {
    if inputs.features.dynamic_link {
        return Ok(Vec::new());
    }
    if inputs.target.os != "linux" {
        return Err(ConfigError::new(format!(
            "static linking is implemented only for Linux, not {}",
            inputs.target.os
        )));
    }
    Ok(vec![
        LinkRequirement::Library("stdc++"),
        LinkRequirement::Library("pthread"),
        LinkRequirement::Library("dl"),
    ])
}
