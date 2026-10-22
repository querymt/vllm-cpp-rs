use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Parser, ValueEnum};
use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::{Cache, Repo, RepoType};
use vllm_cpp::{
    compose_video_mux_argv, VideoDevice, VideoEngine, VideoGenerationParams, VideoMuxParams,
    VideoPartition, VideoResult,
};

const GGUF_REPOSITORY: &str = "realrebelai/MiniMax-H3_GGUFs";
const FL2VA_DIT: &str = "MiniMax-H3-FL2VA-Q4_K_M.gguf";
const REF2VA_DIT: &str = "MiniMax-H3-REF2VA-Q4_K_M.gguf";
const ENCODER: &str = "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf";
const VAE_REPOSITORY: &str = "Comfy-Org/MiniMax-H3";
const VIDEO_VAE: &str = "vae/minimax_h3_video_vae_fp16.safetensors";
const AUDIO_VAE: &str = "vae/minimax_h3_audio_vae_fp32.safetensors";
const CONFIG_REPOSITORY: &str = "MiniMaxAI/MiniMax-H3";
const VIDEO_VAE_CONFIG: &str = "FL2VA/video_vae/config.json";
const AUDIO_VAE_CONFIG: &str = "FL2VA/audio_vae/config.json";
const TOKENIZER: &str = "FL2VA/tokenizer/tokenizer.json";
const DEFAULT_REVISION: &str = "main";
const HF_TOKEN: &str = "HF_TOKEN";
const WORKDIR_LOCK_FILE: &str = ".video_generate.lock";
const PRIVATE_MUX_DIR_PREFIX: &str = ".video_generate-mux-";

static NEXT_PRIVATE_MUX: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Parser)]
#[command(
    name = "video_generate",
    about = "Generate MiniMax-H3 PPM frames and WAV audio from Hugging Face artifacts",
    long_about = "Resolve a complete MiniMax-H3 checkpoint set through the normal Hugging Face cache, then generate PPM frames and WAV audio. The optional MP4 step is run by this example through ffmpeg; vllm-cpp only composes its argument boundaries."
)]
struct Args {
    /// Text used by the Qwen3-VL encoder to condition generation.
    #[arg(long, value_name = "TEXT")]
    prompt: String,

    /// Served MiniMax-H3 DiT partition.
    #[arg(long, value_enum, default_value = "fl2va")]
    partition: PartitionArg,

    /// Video engine device. CUDA requires a CUDA-enabled build and suitable hardware.
    #[arg(long, value_enum, default_value = "cuda")]
    device: DeviceArg,

    /// Output width in pixels; must be a multiple of 32.
    #[arg(long, default_value_t = 256)]
    width: u32,

    /// Output height in pixels; must be a multiple of 32.
    #[arg(long, default_value_t = 256)]
    height: u32,

    /// Requested frame count. Native code rounds up to the next 17n+5 count.
    #[arg(long, default_value_t = 5)]
    frames: u32,

    /// Number of denoising steps; must be at least 2.
    #[arg(long, default_value_t = 2)]
    steps: u32,

    /// Optional deterministic generation seed.
    #[arg(long)]
    seed: Option<u64>,

    /// Writable directory that receives frame_%06d.ppm and audio.wav.
    #[arg(long, default_value = "minimax-h3-output", value_name = "DIR")]
    workdir: PathBuf,

    /// First FL2VA keyframe in binary PPM (P6) format.
    #[arg(long, value_name = "PPM")]
    first_frame: Option<PathBuf>,

    /// Last FL2VA keyframe in binary PPM (P6) format.
    #[arg(long, value_name = "PPM")]
    last_frame: Option<PathBuf>,

    /// Ref2VA reference image in binary PPM (P6) format.
    #[arg(long, value_name = "PPM")]
    reference_image: Option<PathBuf>,

    /// Ref2VA directory containing frame_%06d.ppm reference frames.
    #[arg(long, value_name = "DIR")]
    reference_video: Option<PathBuf>,

    /// Ref2VA reference audio as uncompressed PCM16 WAV at exactly 32 kHz.
    #[arg(long, value_name = "WAV")]
    reference_audio: Option<PathBuf>,

    /// Dequantize supported model weights to BF16 instead of keeping them quantized.
    #[arg(long)]
    dequant_bf16: bool,

    /// Keep supported FP4 weights resident on the selected device.
    #[arg(long)]
    fp4_resident: bool,

    /// Mux to a new MP4 in an existing directory; existing entries are refused.
    #[arg(long, value_name = "MP4")]
    output_mp4: Option<PathBuf>,

    /// ffmpeg executable used by this example for --output-mp4.
    #[arg(long, default_value = "ffmpeg", value_name = "PATH")]
    ffmpeg: PathBuf,

    /// Repository containing the DiT file.
    #[arg(long, default_value = GGUF_REPOSITORY, value_name = "REPO")]
    gguf_repo: String,

    /// Branch, tag, or commit for --gguf-repo.
    #[arg(long, default_value = DEFAULT_REVISION, value_name = "REVISION")]
    gguf_revision: String,

    /// DiT path inside --gguf-repo; overrides must serve --partition.
    #[arg(long, value_name = "PATH")]
    dit_file: Option<String>,

    /// Repository containing the prompt encoder file.
    #[arg(long, default_value = GGUF_REPOSITORY, value_name = "REPO")]
    encoder_repo: String,

    /// Branch, tag, or commit for --encoder-repo.
    #[arg(long, default_value = DEFAULT_REVISION, value_name = "REVISION")]
    encoder_revision: String,

    /// Encoder path inside --encoder-repo.
    #[arg(long, default_value = ENCODER, value_name = "PATH")]
    encoder_file: String,

    /// Repository containing both VAE weight files.
    #[arg(long, default_value = VAE_REPOSITORY, value_name = "REPO")]
    vae_repo: String,

    /// Branch, tag, or commit for --vae-repo.
    #[arg(long, default_value = DEFAULT_REVISION, value_name = "REVISION")]
    vae_revision: String,

    /// Video VAE weight path inside --vae-repo.
    #[arg(long, default_value = VIDEO_VAE, value_name = "PATH")]
    video_vae_file: String,

    /// Audio VAE weight path inside --vae-repo.
    #[arg(long, default_value = AUDIO_VAE, value_name = "PATH")]
    audio_vae_file: String,

    /// Repository containing both VAE configs and the tokenizer.
    #[arg(long, default_value = CONFIG_REPOSITORY, value_name = "REPO")]
    config_repo: String,

    /// Branch, tag, or commit for --config-repo.
    #[arg(long, default_value = DEFAULT_REVISION, value_name = "REVISION")]
    config_revision: String,

    /// Video VAE config path inside --config-repo.
    #[arg(long, default_value = VIDEO_VAE_CONFIG, value_name = "PATH")]
    video_vae_config_file: String,

    /// Audio VAE config path inside --config-repo.
    #[arg(long, default_value = AUDIO_VAE_CONFIG, value_name = "PATH")]
    audio_vae_config_file: String,

    /// Tokenizer path inside --config-repo.
    #[arg(long, default_value = TOKENIZER, value_name = "PATH")]
    tokenizer_file: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PartitionArg {
    Fl2va,
    Ref2va,
}

impl PartitionArg {
    const fn default_dit(self) -> &'static str {
        match self {
            Self::Fl2va => FL2VA_DIT,
            Self::Ref2va => REF2VA_DIT,
        }
    }
}

impl From<PartitionArg> for VideoPartition {
    fn from(value: PartitionArg) -> Self {
        match value {
            PartitionArg::Fl2va => Self::Fl2va,
            PartitionArg::Ref2va => Self::Ref2va,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum DeviceArg {
    Cpu,
    Cuda,
}

impl From<DeviceArg> for VideoDevice {
    fn from(value: DeviceArg) -> Self {
        match value {
            DeviceArg::Cpu => Self::Cpu,
            DeviceArg::Cuda => Self::Cuda,
        }
    }
}

#[derive(Debug)]
struct AppError(String);

impl fmt::Display for AppError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for AppError {}

type AppResult<T> = Result<T, AppError>;

#[derive(Debug)]
struct WorkdirLock {
    path: PathBuf,
    file: Option<File>,
}

impl WorkdirLock {
    fn acquire(workdir: &Path) -> AppResult<Self> {
        let path = workdir.join(WORKDIR_LOCK_FILE);
        let file = match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(app_error(format!(
                    "--workdir `{}` is already in use because lock `{}` exists; after an abnormal termination, remove a stale lock manually only after confirming no process owns the workdir",
                    workdir.display(),
                    path.display()
                )));
            }
            Err(error) => {
                return Err(app_error(format!(
                    "could not atomically lock --workdir `{}` by creating `{}`: {error}",
                    workdir.display(),
                    path.display()
                )));
            }
        };

        let mut lock = Self {
            path,
            file: Some(file),
        };
        if let Err(error) = writeln!(
            lock.file.as_mut().expect("newly acquired lock has a file"),
            "video_generate process {} owns this workdir for resolve, load, generate, and mux",
            std::process::id()
        ) {
            return Err(app_error(format!(
                "could not write ownership context to workdir lock `{}`: {error}",
                lock.path.display()
            )));
        }
        Ok(lock)
    }

    fn remove_owned_file(&mut self) -> io::Result<()> {
        if self.file.is_none() {
            return Ok(());
        }

        #[cfg(unix)]
        let result = {
            use std::os::unix::fs::MetadataExt;

            (|| {
                let owned = self.file.as_ref().expect("checked lock file").metadata()?;
                let current = match fs::symlink_metadata(&self.path) {
                    Ok(metadata) => metadata,
                    Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
                    Err(error) => return Err(error),
                };
                if (owned.dev(), owned.ino()) != (current.dev(), current.ino()) {
                    return Err(io::Error::other(
                        "lock path now names a different filesystem object; refusing to remove it",
                    ));
                }
                fs::remove_file(&self.path)
            })()
        };

        #[cfg(not(unix))]
        let result = Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "this platform cannot prove lock-path identity; retaining the lock",
        ));

        self.file.take();
        result
    }
}

impl Drop for WorkdirLock {
    fn drop(&mut self) {
        if let Err(error) = self.remove_owned_file() {
            let _ = writeln!(
                io::stderr().lock(),
                "warning: could not remove owned workdir lock `{}`: {error}",
                self.path.display()
            );
        }
    }
}

#[derive(Debug)]
struct PrivateMuxTarget {
    directory: PathBuf,
    path: PathBuf,
    owned_output: Option<File>,
    active: bool,
    #[cfg(unix)]
    directory_file: File,
}

impl PrivateMuxTarget {
    fn create(parent: &Path) -> AppResult<Self> {
        loop {
            let sequence = NEXT_PRIVATE_MUX.fetch_add(1, Ordering::Relaxed);
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos());
            let directory = parent.join(format!(
                "{PRIVATE_MUX_DIR_PREFIX}{}-{timestamp:032x}-{sequence:016x}",
                std::process::id()
            ));
            match create_private_mux_directory(&directory) {
                Ok(()) => {
                    #[cfg(unix)]
                    let directory_file = match File::open(&directory) {
                        Ok(file) => file,
                        Err(error) => {
                            if let Err(cleanup_error) = fs::remove_dir(&directory) {
                                let _ = writeln!(
                                    io::stderr().lock(),
                                    "warning: could not remove newly created private MP4 directory `{}` after failing to open it: {cleanup_error}",
                                    directory.display()
                                );
                            }
                            return Err(app_error(format!(
                                "could not retain ownership of private MP4 directory `{}`: {error}",
                                directory.display()
                            )));
                        }
                    };
                    return Ok(Self {
                        path: directory.join("mux.mp4"),
                        directory,
                        owned_output: None,
                        active: true,
                        #[cfg(unix)]
                        directory_file,
                    });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => {
                    return Err(app_error(format!(
                        "could not create a private MP4 directory in `{}`: {error}",
                        parent.display()
                    )));
                }
            }
        }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn capture_output(&mut self) -> io::Result<bool> {
        if self.owned_output.is_some() {
            return Ok(true);
        }
        let metadata = match fs::symlink_metadata(&self.path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error),
        };
        if !metadata.file_type().is_file() {
            return Err(io::Error::other(
                "ffmpeg output is not a regular private temporary file",
            ));
        }

        let file = File::open(&self.path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;

            let opened = file.metadata()?;
            let current = fs::symlink_metadata(&self.path)?;
            if !current.file_type().is_file()
                || (opened.dev(), opened.ino()) != (current.dev(), current.ino())
            {
                return Err(io::Error::other(
                    "private MP4 path changed while recording its ownership",
                ));
            }
        }
        self.owned_output = Some(file);
        Ok(true)
    }

    #[cfg(unix)]
    fn verify_directory_ownership(&self) -> io::Result<()> {
        use std::os::unix::fs::MetadataExt;

        let owned = self.directory_file.metadata()?;
        let current = fs::symlink_metadata(&self.directory)?;
        if !current.file_type().is_dir()
            || (owned.dev(), owned.ino()) != (current.dev(), current.ino())
        {
            return Err(io::Error::other(
                "private MP4 directory path no longer names the directory created by this run",
            ));
        }
        Ok(())
    }

    #[cfg(not(unix))]
    fn verify_directory_ownership(&self) -> io::Result<()> {
        Ok(())
    }

    fn remove_owned_output(&self) -> io::Result<()> {
        let current = match fs::symlink_metadata(&self.path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
        };
        let Some(owned_output) = self.owned_output.as_ref() else {
            return Err(io::Error::other(
                "private MP4 path exists but its ownership was not established; refusing to remove it",
            ));
        };

        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;

            let owned = owned_output.metadata()?;
            if !current.file_type().is_file()
                || (owned.dev(), owned.ino()) != (current.dev(), current.ino())
            {
                return Err(io::Error::other(
                    "private MP4 path now names a different filesystem object; refusing to remove it",
                ));
            }
        }
        fs::remove_file(&self.path)
    }

    fn cleanup(&mut self) -> io::Result<()> {
        if !self.active {
            return Ok(());
        }
        self.verify_directory_ownership()?;
        self.remove_owned_output()?;
        self.verify_directory_ownership()?;
        fs::remove_dir(&self.directory)?;
        self.owned_output.take();
        self.active = false;
        Ok(())
    }
}

impl Drop for PrivateMuxTarget {
    fn drop(&mut self) {
        if let Err(error) = self.cleanup() {
            let _ = writeln!(
                io::stderr().lock(),
                "warning: could not clean private MP4 temporary path `{}`: {error}",
                self.path.display()
            );
        }
    }
}

#[cfg(unix)]
fn create_private_mux_directory(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;

    let mut builder = fs::DirBuilder::new();
    builder.mode(0o700).create(path)
}

#[cfg(not(unix))]
fn create_private_mux_directory(path: &Path) -> io::Result<()> {
    fs::create_dir(path)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct HubArtifact {
    label: &'static str,
    repo: String,
    revision: String,
    path: String,
}

impl HubArtifact {
    fn new(
        label: &'static str,
        repo: impl Into<String>,
        revision: impl Into<String>,
        path: impl Into<String>,
    ) -> Self {
        Self {
            label,
            repo: repo.into(),
            revision: revision.into(),
            path: path.into(),
        }
    }
}

#[derive(Debug)]
struct ResolvedArtifacts {
    dit: PathBuf,
    encoder: PathBuf,
    video_vae: PathBuf,
    audio_vae: PathBuf,
    video_vae_config: PathBuf,
    audio_vae_config: PathBuf,
    tokenizer: PathBuf,
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("video_generate: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> AppResult<()> {
    validate_args(&args)?;
    let generation = generation_params(&args);
    generation
        .validate()
        .map_err(|error| app_error(format!("invalid generation parameters: {error}")))?;
    validate_device_build_compatibility(
        args.device,
        cfg!(feature = "bundled"),
        cfg!(feature = "cuda"),
    )?;
    let workdir_lock = prepare_workdir(&args.workdir)?;

    let sources = artifact_sources(&args);
    let artifacts = resolve_artifacts(&sources)?;

    eprintln!("Loading the MiniMax-H3 checkpoint set...");
    let mut engine = VideoEngine::builder(&artifacts.dit)
        .encoder_path(&artifacts.encoder)
        .tokenizer_path(&artifacts.tokenizer)
        .video_vae_path(&artifacts.video_vae)
        .video_vae_config_path(&artifacts.video_vae_config)
        .audio_vae_path(&artifacts.audio_vae)
        .audio_vae_config_path(&artifacts.audio_vae_config)
        .partition(args.partition.into())
        .device(args.device.into())
        .dequant_bf16(args.dequant_bf16)
        .fp4_resident(args.fp4_resident)
        .load()
        .map_err(|error| app_error(format!("could not load MiniMax-H3: {error}")))?;

    eprintln!("Generating PPM frames and WAV audio...");
    let result = engine
        .generate(&generation)
        .map_err(|error| app_error(format!("generation failed: {error}")))?;
    report_artifacts(&result);

    if let Some(output) = &args.output_mp4 {
        mux_mp4(
            result.frame_dir(),
            result.audio_path(),
            result.fps(),
            output,
            &args.ffmpeg,
        )?;
    }
    drop(workdir_lock);
    Ok(())
}

fn generation_params(args: &Args) -> VideoGenerationParams {
    let mut params = VideoGenerationParams::new(&args.prompt, &args.workdir)
        .dimensions(args.width, args.height)
        .num_frames(args.frames)
        .steps(args.steps);
    if let Some(seed) = args.seed {
        params = params.seed(seed);
    }
    if let Some(path) = &args.first_frame {
        params = params.first_frame(path);
    }
    if let Some(path) = &args.last_frame {
        params = params.last_frame(path);
    }
    if let Some(path) = &args.reference_image {
        params = params.reference_image(path);
    }
    if let Some(path) = &args.reference_video {
        params = params.reference_video(path);
    }
    if let Some(path) = &args.reference_audio {
        params = params.reference_audio(path);
    }
    params
}

fn validate_args(args: &Args) -> AppResult<()> {
    if args.prompt.trim().is_empty() {
        return Err(app_error("--prompt must not be empty"));
    }
    if args.workdir.as_os_str().is_empty() {
        return Err(app_error("--workdir must not be empty"));
    }
    validate_cpu_dit_compatibility(args)?;
    if args.width % 32 != 0 || args.height % 32 != 0 {
        return Err(app_error("--width and --height must be multiples of 32"));
    }
    let width = u64::from(args.width);
    let height = u64::from(args.height);
    if width > 4 * height || height > 4 * width {
        return Err(app_error(
            "--width and --height must have an aspect ratio in [1:4, 4:1]",
        ));
    }
    if args.frames <= 1 {
        return Err(app_error("--frames must be greater than one"));
    }
    if args.steps < 2 {
        return Err(app_error("--steps must be at least 2"));
    }
    if args
        .output_mp4
        .as_ref()
        .is_some_and(|path| path.as_os_str().is_empty())
    {
        return Err(app_error("--output-mp4 must not be empty"));
    }

    let has_keyframes = args.first_frame.is_some() || args.last_frame.is_some();
    let has_references = args.reference_image.is_some()
        || args.reference_video.is_some()
        || args.reference_audio.is_some();
    if has_keyframes && has_references {
        return Err(app_error(
            "FL2VA keyframes cannot be combined with Ref2VA references",
        ));
    }
    if args.reference_image.is_some() && args.reference_video.is_some() {
        return Err(app_error(
            "--reference-image and --reference-video are mutually exclusive",
        ));
    }
    let effective_dit_name = portable_hub_path_file_name(effective_dit(args));
    match args.partition {
        PartitionArg::Fl2va if effective_dit_name == REF2VA_DIT => {
            return Err(app_error(format!(
                "DiT `{REF2VA_DIT}` is the known Ref2VA model and cannot serve --partition fl2va"
            )));
        }
        PartitionArg::Ref2va if effective_dit_name == FL2VA_DIT => {
            return Err(app_error(format!(
                "DiT `{FL2VA_DIT}` is the known FL2VA model and cannot serve --partition ref2va"
            )));
        }
        PartitionArg::Fl2va if has_references => {
            return Err(app_error(
                "Ref2VA references require --partition ref2va and its matching DiT",
            ));
        }
        PartitionArg::Ref2va if has_keyframes => {
            return Err(app_error(
                "keyframes require --partition fl2va and its matching DiT",
            ));
        }
        PartitionArg::Ref2va if !has_references => {
            return Err(app_error(
                "--partition ref2va requires a reference image, video, or audio",
            ));
        }
        _ => {}
    }

    require_file(args.first_frame.as_deref(), "--first-frame")?;
    require_file(args.last_frame.as_deref(), "--last-frame")?;
    require_file(args.reference_image.as_deref(), "--reference-image")?;
    require_file(args.reference_audio.as_deref(), "--reference-audio")?;
    require_directory(args.reference_video.as_deref(), "--reference-video")?;
    if let Some(path) = &args.reference_audio {
        validate_reference_wav(path)?;
    }
    if let Some(path) = &args.output_mp4 {
        validate_mp4_output(path, &args.workdir)?;
    }
    Ok(())
}

fn require_file(path: Option<&Path>, flag: &str) -> AppResult<()> {
    require_path_kind(path, flag, "file", std::fs::Metadata::is_file)
}

fn require_directory(path: Option<&Path>, flag: &str) -> AppResult<()> {
    require_path_kind(path, flag, "directory", std::fs::Metadata::is_dir)
}

fn validate_reference_wav(path: &Path) -> AppResult<()> {
    const MIN_WAV_BYTES: u64 = 44;
    const MODEL_SAMPLE_RATE: u32 = 32_000;

    let mut file = File::open(path).map_err(|error| {
        app_error(format!(
            "could not open --reference-audio `{}`: {error}",
            path.display()
        ))
    })?;
    let file_len = file
        .metadata()
        .map_err(|error| {
            app_error(format!(
                "could not inspect --reference-audio `{}`: {error}",
                path.display()
            ))
        })?
        .len();
    if file_len < MIN_WAV_BYTES {
        return Err(invalid_reference_wav(
            path,
            "too short to be a RIFF/WAVE file",
        ));
    }

    let mut riff_header = [0_u8; 12];
    file.read_exact(&mut riff_header).map_err(|error| {
        invalid_reference_wav(path, format!("could not read RIFF header: {error}"))
    })?;
    if &riff_header[..4] != b"RIFF" || &riff_header[8..] != b"WAVE" {
        return Err(invalid_reference_wav(path, "not a RIFF/WAVE file"));
    }

    let mut offset = 12_u64;
    let mut format = None;
    let mut data_len = None;
    while offset
        .checked_add(8)
        .is_some_and(|header_end| header_end <= file_len)
    {
        file.seek(SeekFrom::Start(offset)).map_err(|error| {
            invalid_reference_wav(path, format!("could not seek to WAV chunk: {error}"))
        })?;
        let mut chunk_header = [0_u8; 8];
        file.read_exact(&mut chunk_header).map_err(|error| {
            invalid_reference_wav(path, format!("could not read WAV chunk header: {error}"))
        })?;
        let declared_len = u64::from(u32::from_le_bytes(
            chunk_header[4..8]
                .try_into()
                .expect("four-byte chunk length"),
        ));
        let body = offset.checked_add(8).ok_or_else(|| {
            invalid_reference_wav(
                path,
                "WAV chunk body offset overflows the file address space",
            )
        })?;

        if &chunk_header[..4] == b"fmt "
            && declared_len >= 16
            && body
                .checked_add(16)
                .is_some_and(|format_end| format_end <= file_len)
        {
            file.seek(SeekFrom::Start(body)).map_err(|error| {
                invalid_reference_wav(path, format!("could not seek to fmt chunk: {error}"))
            })?;
            let mut fields = [0_u8; 16];
            file.read_exact(&mut fields).map_err(|error| {
                invalid_reference_wav(path, format!("could not read fmt chunk: {error}"))
            })?;
            let format_tag = u16::from_le_bytes([fields[0], fields[1]]);
            if format_tag != 1 {
                return Err(invalid_reference_wav(
                    path,
                    "only uncompressed PCM (format tag 1) is supported",
                ));
            }
            format = Some((
                u16::from_le_bytes([fields[2], fields[3]]),
                u32::from_le_bytes([fields[4], fields[5], fields[6], fields[7]]),
                u16::from_le_bytes([fields[14], fields[15]]),
            ));
        } else if &chunk_header[..4] == b"data" {
            data_len = Some(declared_len.min(file_len.saturating_sub(body)));
            break;
        }

        offset = body
            .checked_add(declared_len)
            .and_then(|end| end.checked_add(declared_len & 1))
            .ok_or_else(|| {
                invalid_reference_wav(path, "WAV chunk offset overflows the file address space")
            })?;
    }

    let Some((channels, sample_rate, bits_per_sample)) = format else {
        return Err(invalid_reference_wav(
            path,
            "no complete fmt chunk before data",
        ));
    };
    let Some(data_len) = data_len else {
        return Err(invalid_reference_wav(path, "no data chunk after fmt"));
    };
    if bits_per_sample != 16 {
        return Err(invalid_reference_wav(
            path,
            format!("expected 16-bit PCM, found {bits_per_sample}-bit samples"),
        ));
    }
    if channels == 0 {
        return Err(invalid_reference_wav(
            path,
            "the fmt chunk declares zero channels",
        ));
    }
    if sample_rate != MODEL_SAMPLE_RATE {
        return Err(invalid_reference_wav(
            path,
            format!(
                "expected exactly {MODEL_SAMPLE_RATE} Hz, found {sample_rate} Hz; resample the file first"
            ),
        ));
    }
    let source_frame_bytes = u64::from(channels) * 2;
    if data_len < source_frame_bytes {
        return Err(invalid_reference_wav(
            path,
            "the data chunk contains no complete source frame",
        ));
    }
    Ok(())
}

fn invalid_reference_wav(path: &Path, reason: impl fmt::Display) -> AppError {
    app_error(format!(
        "invalid --reference-audio `{}`: {reason}",
        path.display()
    ))
}

fn validate_mp4_output(output: &Path, workdir: &Path) -> AppResult<()> {
    match fs::symlink_metadata(output) {
        Ok(_) => {
            return Err(app_error(format!(
                "--output-mp4 `{}` already exists; refusing to replace any existing entry",
                output.display()
            )));
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(app_error(format!(
                "could not inspect --output-mp4 `{}` without following its final component: {error}",
                output.display()
            )));
        }
    }

    let file_name = output.file_name().ok_or_else(|| {
        app_error(format!(
            "--output-mp4 `{}` has no output file name",
            output.display()
        ))
    })?;
    let parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let resolved_parent = fs::canonicalize(parent).map_err(|error| {
        app_error(format!(
            "could not safely resolve --output-mp4 parent `{}`: {error}",
            parent.display()
        ))
    })?;
    if !fs::metadata(&resolved_parent)
        .map_err(|error| {
            app_error(format!(
                "could not inspect resolved --output-mp4 parent `{}`: {error}",
                resolved_parent.display()
            ))
        })?
        .is_dir()
    {
        return Err(app_error(format!(
            "--output-mp4 parent `{}` is not a directory",
            parent.display()
        )));
    }

    let resolved_workdir = match fs::canonicalize(workdir) {
        Ok(path) => Some(path),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(app_error(format!(
                "could not resolve --workdir `{}` while checking --output-mp4 aliases: {error}",
                workdir.display()
            )));
        }
    };
    if resolved_workdir.as_deref() == Some(resolved_parent.as_path())
        && is_reserved_workdir_output(file_name)
    {
        return Err(app_error(format!(
            "--output-mp4 `{}` aliases reserved workdir output `{}`",
            output.display(),
            file_name.to_string_lossy()
        )));
    }
    Ok(())
}

fn is_reserved_workdir_output(file_name: &std::ffi::OsStr) -> bool {
    file_name == WORKDIR_LOCK_FILE
        || file_name == "audio.wav"
        || file_name.to_str().is_some_and(is_native_frame_output_name)
}

fn prepare_workdir(path: &Path) -> AppResult<WorkdirLock> {
    match fs::metadata(path) {
        Ok(metadata) if !metadata.is_dir() => {
            return Err(app_error(format!(
                "--workdir `{}` is not a directory",
                path.display()
            )));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir_all(path).map_err(|error| {
                app_error(format!(
                    "could not create --workdir directory `{}`: {error}",
                    path.display()
                ))
            })?;
        }
        Err(error) => {
            return Err(app_error(format!(
                "could not inspect or prepare --workdir `{}`: {error}",
                path.display()
            )));
        }
    }

    let metadata = fs::metadata(path).map_err(|error| {
        app_error(format!(
            "could not inspect prepared --workdir `{}`: {error}",
            path.display()
        ))
    })?;
    if !metadata.is_dir() {
        return Err(app_error(format!(
            "--workdir `{}` is not a directory",
            path.display()
        )));
    }

    let lock = WorkdirLock::acquire(path)?;
    reject_existing_native_outputs(path)?;
    Ok(lock)
}

fn reject_existing_native_outputs(path: &Path) -> AppResult<()> {
    let entries = fs::read_dir(path).map_err(|error| {
        app_error(format!(
            "could not read --workdir `{}` while checking native output destinations: {error}",
            path.display()
        ))
    })?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            app_error(format!(
                "could not inspect an entry in --workdir `{}` while checking native output destinations: {error}",
                path.display()
            ))
        })?;
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };
        if file_name == WORKDIR_LOCK_FILE {
            continue;
        }
        if file_name == "audio.wav" || is_native_frame_output_name(file_name) {
            return Err(app_error(format!(
                "--workdir `{}` already contains native output destination `{file_name}`",
                path.display()
            )));
        }
    }
    Ok(())
}

fn is_native_frame_output_name(file_name: &str) -> bool {
    let Some(index) = file_name
        .strip_prefix("frame_")
        .and_then(|name| name.strip_suffix(".ppm"))
    else {
        return false;
    };
    // Native `%06lld` uses six as a minimum width and emits wider indices unchanged.
    index.len() >= 6 && index.bytes().all(|byte| byte.is_ascii_digit())
}

fn require_path_kind(
    path: Option<&Path>,
    flag: &str,
    expected: &str,
    matches: impl FnOnce(&std::fs::Metadata) -> bool,
) -> AppResult<()> {
    let Some(path) = path else {
        return Ok(());
    };
    let metadata = std::fs::metadata(path).map_err(|error| {
        app_error(format!(
            "could not inspect {flag} `{}`: {error}",
            path.display()
        ))
    })?;
    if !matches(&metadata) {
        return Err(app_error(format!(
            "{flag} `{}` is not a {expected}",
            path.display()
        )));
    }
    Ok(())
}

fn effective_dit(args: &Args) -> &str {
    args.dit_file
        .as_deref()
        .unwrap_or_else(|| args.partition.default_dit())
}

fn validate_device_build_compatibility(
    device: DeviceArg,
    bundled: bool,
    cuda: bool,
) -> AppResult<()> {
    // System libraries configure backends independently of Cargo features.
    if device == DeviceArg::Cuda && bundled && !cuda {
        return Err(app_error(
            "--device cuda is unavailable: this bundled build has CUDA disabled; rebuild with --features cuda and a compatible CUDA toolkit/GPU. Selecting --device cpu does not support the default GGUF DiT",
        ));
    }
    Ok(())
}

fn validate_cpu_dit_compatibility(args: &Args) -> AppResult<()> {
    let dit = effective_dit(args);
    let is_gguf = portable_hub_path_file_name(dit)
        .to_ascii_lowercase()
        .ends_with(".gguf");
    if args.device == DeviceArg::Cpu && is_gguf {
        return Err(app_error(format!(
            "--device cpu cannot use GGUF DiT `{dit}` with the pinned vllm.cpp: the CPU MiniMax-H3 host forward requires F32 weights, while GGUF provides block-quantized weights or BF16 with --dequant-bf16; lower quantization does not fix this routing. Use --device cuda or a future native quantized CPU-forward implementation"
        )));
    }
    Ok(())
}

fn portable_hub_path_file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn artifact_sources(args: &Args) -> [HubArtifact; 7] {
    [
        HubArtifact::new(
            "DiT",
            &args.gguf_repo,
            &args.gguf_revision,
            effective_dit(args),
        ),
        HubArtifact::new(
            "prompt encoder",
            &args.encoder_repo,
            &args.encoder_revision,
            &args.encoder_file,
        ),
        HubArtifact::new(
            "video VAE",
            &args.vae_repo,
            &args.vae_revision,
            &args.video_vae_file,
        ),
        HubArtifact::new(
            "audio VAE",
            &args.vae_repo,
            &args.vae_revision,
            &args.audio_vae_file,
        ),
        HubArtifact::new(
            "video VAE config",
            &args.config_repo,
            &args.config_revision,
            &args.video_vae_config_file,
        ),
        HubArtifact::new(
            "audio VAE config",
            &args.config_repo,
            &args.config_revision,
            &args.audio_vae_config_file,
        ),
        HubArtifact::new(
            "tokenizer",
            &args.config_repo,
            &args.config_revision,
            &args.tokenizer_file,
        ),
    ]
}

fn cached_commit_snapshot_or_else<E>(
    cache: &Cache,
    repo: &str,
    commit: &str,
    artifact_path: &str,
    fallback: impl FnOnce() -> Result<PathBuf, E>,
) -> Result<PathBuf, E> {
    let snapshot_path = cache
        .repo(Repo::new(repo.to_owned(), RepoType::Model))
        .pointer_path(commit)
        .join(artifact_path);
    if snapshot_path.is_file() {
        Ok(snapshot_path)
    } else {
        fallback()
    }
}

fn validate_artifact_sources(sources: &[HubArtifact; 7]) -> AppResult<()> {
    for source in sources {
        validate_repo_id(&source.repo, source.label)?;
        validate_hub_path(&source.revision, &format!("{} revision", source.label))?;
        validate_hub_path(&source.path, &format!("{} artifact path", source.label))?;
    }
    Ok(())
}

fn source_revision_groups(sources: &[HubArtifact; 7]) -> BTreeMap<(String, String), usize> {
    let mut groups = BTreeMap::new();
    for (index, source) in sources.iter().enumerate() {
        groups
            .entry((source.repo.clone(), source.revision.clone()))
            .or_insert(index);
    }
    groups
}

fn resolve_source_revisions<E>(
    sources: &[HubArtifact; 7],
    mut resolve: impl FnMut(&HubArtifact) -> Result<String, E>,
) -> Result<BTreeMap<(String, String), String>, E> {
    let groups = source_revision_groups(sources);
    let mut commits = BTreeMap::new();
    for (index, source) in sources.iter().enumerate() {
        let key = (source.repo.clone(), source.revision.clone());
        if groups.get(&key) == Some(&index) {
            commits.insert(key, resolve(source)?);
        }
    }
    Ok(commits)
}

fn resolve_artifacts(sources: &[HubArtifact; 7]) -> AppResult<ResolvedArtifacts> {
    validate_artifact_sources(sources)?;

    let cache = Cache::from_env();
    let mut builder = ApiBuilder::from_cache(cache.clone()).with_progress(true);
    if let Some(token) = selected_hf_token(env::var(HF_TOKEN).ok()) {
        builder = builder.with_token(Some(token));
    }
    let api = builder
        .build()
        .map_err(|error| app_error(format!("could not create Hugging Face API: {error}")))?;

    let commits = resolve_source_revisions(sources, |source| resolve_commit(&api, source))?;

    let mut paths = Vec::with_capacity(sources.len());
    for source in sources {
        let key = (source.repo.clone(), source.revision.clone());
        let sha = commits
            .get(&key)
            .expect("every validated artifact source has a revision group")
            .clone();
        eprintln!(
            "Resolving {} `{}` from `{}` at `{}`...",
            source.label, source.path, source.repo, source.revision
        );
        let path =
            cached_commit_snapshot_or_else(&cache, &source.repo, &sha, &source.path, || {
                api.repo(Repo::with_revision(
                    source.repo.clone(),
                    RepoType::Model,
                    sha.clone(),
                ))
                .get(&source.path)
                .map_err(|error| {
                    app_error(format!(
                        "could not resolve {} `{}` from `{}` at `{}`: {error}",
                        source.label, source.path, source.repo, source.revision
                    ))
                })
            })?;
        if !path.is_file() {
            return Err(app_error(format!(
                "resolved {} path `{}` is not a file",
                source.label,
                path.display()
            )));
        }
        paths.push(path);
    }

    for ((repo, revision), sha) in commits {
        cache
            .repo(Repo::with_revision(
                repo.clone(),
                RepoType::Model,
                revision.clone(),
            ))
            .create_ref(&sha)
            .map_err(|error| {
                app_error(format!(
                    "could not update Hugging Face cache ref `{revision}` for `{repo}`: {error}"
                ))
            })?;
    }

    let [dit, encoder, video_vae, audio_vae, video_vae_config, audio_vae_config, tokenizer] =
        paths.try_into().map_err(|_| {
            app_error("internal error: Hugging Face artifact count changed unexpectedly")
        })?;
    Ok(ResolvedArtifacts {
        dit,
        encoder,
        video_vae,
        audio_vae,
        video_vae_config,
        audio_vae_config,
        tokenizer,
    })
}

fn selected_hf_token(environment_token: Option<String>) -> Option<String> {
    environment_token.filter(|token| !token.trim().is_empty())
}

fn resolve_commit(api: &Api, source: &HubArtifact) -> AppResult<String> {
    let requested = Repo::with_revision(
        source.repo.clone(),
        RepoType::Model,
        source.revision.clone(),
    );
    let info = api.repo(requested).info().map_err(|error| {
        app_error(format!(
            "could not read metadata for `{}` at `{}`: {error}",
            source.repo, source.revision
        ))
    })?;
    let sha = info.sha.trim();
    validate_single_component(sha, "repository metadata commit SHA")?;
    Ok(sha.to_owned())
}

fn validate_repo_id(repo: &str, label: &str) -> AppResult<()> {
    validate_hub_path(repo, &format!("{label} repository ID"))?;
    if repo.split('/').count() > 2 {
        return Err(app_error(format!(
            "{label} repository ID must be `name` or `namespace/name`"
        )));
    }
    Ok(())
}

fn validate_single_component(value: &str, label: &str) -> AppResult<()> {
    validate_hub_path(value, label)?;
    if Path::new(value).components().count() != 1 {
        return Err(app_error(format!(
            "{label} must be a single path component"
        )));
    }
    Ok(())
}

fn validate_hub_path(value: &str, label: &str) -> AppResult<()> {
    if value.trim().is_empty() {
        return Err(app_error(format!("{label} must not be empty")));
    }
    if value.contains(['\\', '\0', ':'])
        || value.starts_with('~')
        || value.split('/').any(str::is_empty)
    {
        return Err(app_error(format!(
            "{label} must use portable repository-relative path syntax"
        )));
    }
    let path = Path::new(value);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(app_error(format!(
            "{label} must not be absolute or contain `.` or `..` components"
        )));
    }
    for component in path.components() {
        let Component::Normal(component) = component else {
            unreachable!("non-normal components were rejected above");
        };
        let component = component.to_string_lossy();
        if component.ends_with([' ', '.'])
            || component.chars().any(|character| {
                character.is_control() || matches!(character, '<' | '>' | '"' | '|' | '?' | '*')
            })
        {
            return Err(app_error(format!(
                "{label} contains characters that are not portable path syntax"
            )));
        }
    }
    Ok(())
}

fn report_artifacts(result: &VideoResult) {
    println!(
        "wrote {} PPM frames to {} ({}x{} at {} fps)",
        result.frame_count(),
        result.frame_dir().display(),
        result.width(),
        result.height(),
        result.fps()
    );
    println!(
        "wrote WAV audio to {} ({} Hz)",
        result.audio_path().display(),
        result.sample_rate()
    );
}

fn mux_mp4(
    frame_dir: &Path,
    audio_path: &Path,
    fps: u32,
    output: &Path,
    ffmpeg: &Path,
) -> AppResult<()> {
    let file_name = output.file_name().ok_or_else(|| {
        app_error(format!(
            "--output-mp4 `{}` has no output file name",
            output.display()
        ))
    })?;
    let requested_parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let canonical_parent = fs::canonicalize(requested_parent).map_err(|error| {
        app_error(format!(
            "could not safely resolve --output-mp4 parent `{}` for muxing: {error}",
            requested_parent.display()
        ))
    })?;
    if !fs::metadata(&canonical_parent)
        .map_err(|error| {
            app_error(format!(
                "could not inspect resolved --output-mp4 parent `{}` for muxing: {error}",
                canonical_parent.display()
            ))
        })?
        .is_dir()
    {
        return Err(app_error(format!(
            "--output-mp4 parent `{}` is not a directory",
            requested_parent.display()
        )));
    }
    let destination = canonical_parent.join(file_name);
    let mut private_output = PrivateMuxTarget::create(&canonical_parent)?;

    let frame_pattern = frame_dir.join("frame_%06d.ppm");
    let argv = compose_video_mux_argv(
        &VideoMuxParams::new(frame_pattern, private_output.path())
            .audio_path(audio_path)
            .fps(fps),
    )
    .map_err(|error| app_error(format!("could not compose ffmpeg arguments: {error}")))?;
    let mut arguments = argv.into_args();
    if arguments.is_empty() {
        return Err(app_error("ffmpeg argument composer returned an empty argv"));
    }
    let overwrite_policy = arguments
        .get_mut(1)
        .filter(|argument| argument.as_os_str() == "-y")
        .ok_or_else(|| {
            app_error("ffmpeg argument composer did not return the expected overwrite policy")
        })?;
    *overwrite_policy = "-n".into();

    eprintln!(
        "Running ffmpeg from this example to create a private MP4 for `{}`...",
        output.display()
    );
    let status = Command::new(ffmpeg)
        .args(&arguments[1..])
        .status()
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                app_error(format!(
                    "ffmpeg executable `{}` was not found; install ffmpeg or pass --ffmpeg PATH (the PPM/WAV artifacts remain in `{}`)",
                    ffmpeg.display(),
                    frame_dir.display()
                ))
            } else {
                app_error(format!(
                    "could not start ffmpeg executable `{}`: {error} (the PPM/WAV artifacts remain in `{}`)",
                    ffmpeg.display(),
                    frame_dir.display()
                ))
            }
        })?;
    let captured_output = private_output.capture_output();
    if !status.success() {
        if let Err(error) = captured_output {
            eprintln!(
                "warning: could not establish ownership of ffmpeg's private output `{}` for cleanup: {error}",
                private_output.path().display()
            );
        }
        let outcome = status.code().map_or_else(
            || "after a signal".to_owned(),
            |code| format!("with status {code}"),
        );
        return Err(app_error(format!(
            "ffmpeg exited {outcome}; the PPM/WAV artifacts remain in `{}`",
            frame_dir.display()
        )));
    }
    if !captured_output.map_err(|error| {
        app_error(format!(
            "could not establish ownership of ffmpeg's private MP4 output `{}`: {error}",
            private_output.path().display()
        ))
    })? {
        return Err(app_error(format!(
            "ffmpeg succeeded without creating its private MP4 output `{}`",
            private_output.path().display()
        )));
    }

    match fs::hard_link(private_output.path(), &destination) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            return Err(app_error(format!(
                "--output-mp4 `{}` appeared before atomic publication; refusing to replace the existing entry",
                output.display()
            )));
        }
        Err(error) => {
            return Err(app_error(format!(
                "could not atomically publish --output-mp4 `{}` with a same-filesystem hard link from its private mux: {error}; the destination filesystem must support hard links",
                output.display()
            )));
        }
    }
    private_output.cleanup().map_err(|error| {
        app_error(format!(
            "published --output-mp4 `{}` but could not remove its private temporary link `{}`: {error}",
            output.display(),
            private_output.path().display()
        ))
    })?;

    println!("wrote MP4 to {}", output.display());
    Ok(())
}

fn app_error(message: impl Into<String>) -> AppError {
    AppError(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct TempDir(PathBuf);

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &Path) -> Self {
            let previous = env::var_os(key);
            env::set_var(key, value);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(value) = &self.previous {
                env::set_var(self.key, value);
            } else {
                env::remove_var(self.key);
            }
        }
    }

    impl TempDir {
        fn new() -> Self {
            loop {
                let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
                let path = env::temp_dir().join(format!(
                    "vllm-cpp-video-generate-{}-{id}",
                    std::process::id()
                ));
                match fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(error) => panic!("could not create temporary directory: {error}"),
                }
            }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn parse(extra: &[&str]) -> Args {
        let mut arguments = vec!["video_generate", "--prompt", "a test prompt"];
        arguments.extend_from_slice(extra);
        Args::try_parse_from(arguments).expect("valid arguments")
    }

    fn pcm_format(format_tag: u16, channels: u16, sample_rate: u32, bits: u16) -> Vec<u8> {
        let mut fields = Vec::with_capacity(16);
        fields.extend_from_slice(&format_tag.to_le_bytes());
        fields.extend_from_slice(&channels.to_le_bytes());
        fields.extend_from_slice(&sample_rate.to_le_bytes());
        fields.extend_from_slice(&0_u32.to_le_bytes());
        fields.extend_from_slice(&0_u16.to_le_bytes());
        fields.extend_from_slice(&bits.to_le_bytes());
        fields
    }

    fn wav_fixture(chunks: Vec<([u8; 4], Vec<u8>)>) -> Vec<u8> {
        let mut wav = b"RIFF\0\0\0\0WAVE".to_vec();
        for (id, body) in chunks {
            wav.extend_from_slice(&id);
            wav.extend_from_slice(
                &u32::try_from(body.len())
                    .expect("small fixture")
                    .to_le_bytes(),
            );
            wav.extend_from_slice(&body);
            if body.len() % 2 != 0 {
                wav.push(0);
            }
        }
        let riff_len = u32::try_from(wav.len() - 8).expect("small fixture");
        wav[4..8].copy_from_slice(&riff_len.to_le_bytes());
        wav
    }

    fn write_wav(path: &Path, chunks: Vec<([u8; 4], Vec<u8>)>) {
        fs::write(path, wav_fixture(chunks)).expect("write WAV fixture");
    }

    #[test]
    fn smoke_defaults_use_verified_hugging_face_artifacts() {
        let args = parse(&[]);
        assert_eq!(args.partition, PartitionArg::Fl2va);
        assert_eq!(args.device, DeviceArg::Cuda);
        assert_eq!(
            (args.width, args.height, args.frames, args.steps),
            (256, 256, 5, 2)
        );

        let sources = artifact_sources(&args);
        assert_eq!(sources[0].repo, GGUF_REPOSITORY);
        assert_eq!(sources[0].revision, DEFAULT_REVISION);
        assert_eq!(sources[0].path, FL2VA_DIT);
        assert_eq!(args.encoder_repo, GGUF_REPOSITORY);
        assert_eq!(args.encoder_revision, DEFAULT_REVISION);
        assert_eq!(args.encoder_file, ENCODER);
        assert_eq!(sources[1].repo, GGUF_REPOSITORY);
        assert_eq!(sources[1].revision, DEFAULT_REVISION);
        assert_eq!(sources[1].path, ENCODER);
        assert_eq!(sources[2].repo, VAE_REPOSITORY);
        assert_eq!(sources[2].path, VIDEO_VAE);
        assert_eq!(sources[3].path, AUDIO_VAE);
        assert_eq!(sources[4].repo, CONFIG_REPOSITORY);
        assert_eq!(sources[4].path, VIDEO_VAE_CONFIG);
        assert_eq!(sources[5].path, AUDIO_VAE_CONFIG);
        assert_eq!(sources[6].path, TOKENIZER);
        validate_args(&args).expect("valid smoke defaults");
        generation_params(&args)
            .validate()
            .expect("safe API accepts smoke defaults");
    }

    #[test]
    fn encoder_source_is_independent_and_identical_groups_coalesce() {
        let defaults = artifact_sources(&parse(&[]));
        let default_groups = source_revision_groups(&defaults);
        assert_eq!(default_groups.len(), 3);
        assert_eq!(
            default_groups.get(&(GGUF_REPOSITORY.to_owned(), DEFAULT_REVISION.to_owned())),
            Some(&0)
        );
        let mut default_resolutions = 0;
        let default_commits = resolve_source_revisions(&defaults, |source| {
            default_resolutions += 1;
            Ok::<_, std::convert::Infallible>(format!("{}@{}", source.repo, source.revision))
        })
        .expect("offline resolver cannot fail");
        assert_eq!(default_resolutions, 3);
        assert_eq!(
            default_commits.get(&(GGUF_REPOSITORY.to_owned(), DEFAULT_REVISION.to_owned())),
            Some(&format!("{GGUF_REPOSITORY}@{DEFAULT_REVISION}"))
        );

        let dit_only = parse(&[
            "--gguf-repo",
            "dit/repository",
            "--gguf-revision",
            "dit-revision",
        ]);
        assert_eq!(dit_only.encoder_repo, GGUF_REPOSITORY);
        assert_eq!(dit_only.encoder_revision, DEFAULT_REVISION);

        let args = parse(&[
            "--gguf-repo",
            "dit/repository",
            "--gguf-revision",
            "dit-revision",
            "--encoder-repo",
            "encoder/repository",
            "--encoder-revision",
            "encoder-revision",
            "--encoder-file",
            "nested/encoder-Q2_K.gguf",
        ]);
        let sources = artifact_sources(&args);
        assert_eq!(sources[0].repo, "dit/repository");
        assert_eq!(sources[0].revision, "dit-revision");
        assert_eq!(sources[1].repo, "encoder/repository");
        assert_eq!(sources[1].revision, "encoder-revision");
        assert_eq!(sources[1].path, "nested/encoder-Q2_K.gguf");
        assert_eq!(source_revision_groups(&sources).len(), 4);
        validate_artifact_sources(&sources).expect("independent sources are valid offline");

        let mut resolved = Vec::new();
        let commits = resolve_source_revisions(&sources, |source| {
            resolved.push((source.repo.clone(), source.revision.clone()));
            Ok::<_, std::convert::Infallible>(format!("{}@{}", source.repo, source.revision))
        })
        .expect("offline resolver cannot fail");
        assert_eq!(resolved.len(), 4);
        assert_eq!(
            commits.get(&("dit/repository".to_owned(), "dit-revision".to_owned())),
            Some(&"dit/repository@dit-revision".to_owned())
        );
        assert_eq!(
            commits.get(&(
                "encoder/repository".to_owned(),
                "encoder-revision".to_owned()
            )),
            Some(&"encoder/repository@encoder-revision".to_owned())
        );
    }

    #[test]
    fn help_lists_independent_encoder_source_options() {
        let help = Args::try_parse_from(["video_generate", "--help"])
            .expect_err("--help exits through Clap")
            .to_string();
        assert!(help.contains("--encoder-repo <REPO>"));
        assert!(help.contains("--encoder-revision <REVISION>"));
        assert!(help.contains("--encoder-file <PATH>"));
    }

    #[test]
    fn rejects_fewer_than_two_denoising_steps() {
        let args = parse(&["--steps", "1"]);
        assert!(validate_args(&args)
            .unwrap_err()
            .to_string()
            .contains("at least 2"));
    }

    #[test]
    fn device_build_guard_rejects_only_known_bundled_cuda_incompatibility() {
        // These are the valid source-mode/CUDA combinations; system + cuda is
        // rejected by the sys build contract before the example can compile.
        for (bundled, cuda, rejects_cuda) in [
            (true, false, true),
            (true, true, false),
            (false, false, false),
        ] {
            assert_eq!(
                validate_device_build_compatibility(DeviceArg::Cuda, bundled, cuda).is_err(),
                rejects_cuda,
                "bundled={bundled}, cuda={cuda}"
            );
            validate_device_build_compatibility(DeviceArg::Cpu, bundled, cuda)
                .expect("build guard does not reject CPU selection");
        }
    }

    #[cfg(all(feature = "bundled", not(feature = "cuda")))]
    #[test]
    fn rejects_default_cuda_before_workdir_or_hub_side_effects() {
        let _env_lock = ENV_LOCK.lock().expect("lock environment");
        let temp = TempDir::new();
        let isolated_hf_home = temp.0.join("hf-home");
        let _hf_home = EnvVarGuard::set("HF_HOME", &isolated_hf_home);
        let workdir = temp.0.join("must-not-be-created");
        let mut args = parse(&[]);
        args.workdir = workdir.clone();
        args.gguf_repo = "must/not/be/resolved".to_owned();
        validate_args(&args).expect("default CLI is semantically valid");
        generation_params(&args)
            .validate()
            .expect("default generation parameters are valid");

        let error = run(args).expect_err("CUDA build guard must fail before setup");
        assert!(error
            .to_string()
            .contains("bundled build has CUDA disabled"));
        assert!(!workdir.exists());
        assert!(!isolated_hf_home.exists());
    }

    #[cfg(all(feature = "bundled", not(feature = "cuda")))]
    #[test]
    fn rejects_default_cuda_before_locking_existing_workdir() {
        let _env_lock = ENV_LOCK.lock().expect("lock environment");
        let temp = TempDir::new();
        let isolated_hf_home = temp.0.join("hf-home");
        let _hf_home = EnvVarGuard::set("HF_HOME", &isolated_hf_home);
        let workdir = temp.0.join("existing-workdir");
        fs::create_dir(&workdir).expect("create existing workdir");
        let lock_path = workdir.join(WORKDIR_LOCK_FILE);

        for locked in [false, true] {
            if locked {
                fs::write(&lock_path, b"owned elsewhere").expect("write competing lock");
            }
            let mut args = parse(&[]);
            args.workdir = workdir.clone();
            args.gguf_repo = "must/not/be/resolved".to_owned();
            let error = run(args).expect_err("CUDA build guard precedes lock acquisition");
            assert!(error
                .to_string()
                .contains("bundled build has CUDA disabled"));
            assert_eq!(
                fs::read_dir(&workdir).expect("inspect workdir").count(),
                usize::from(locked)
            );
            if locked {
                assert_eq!(
                    fs::read(&lock_path).expect("read competing lock"),
                    b"owned elsewhere"
                );
            }
        }
        assert!(!isolated_hf_home.exists());
    }

    #[test]
    fn rejects_cpu_gguf_before_workdir_or_hub_side_effects() {
        let _env_lock = ENV_LOCK.lock().expect("lock environment");
        let temp = TempDir::new();
        let isolated_hf_home = temp.0.join("hf-home");
        let _hf_home = EnvVarGuard::set("HF_HOME", &isolated_hf_home);

        for (case, dit) in [
            ("default", None),
            ("lower-q", Some("nested/minimax_h3_fl2va_pruned-Q2_K.gguf")),
        ] {
            let workdir = temp.0.join(format!("{case}-must-not-be-created"));
            let mut args = parse(&["--device", "cpu"]);
            args.workdir = workdir.clone();
            args.gguf_repo = "must/not/be/resolved".to_owned();
            args.dit_file = dit.map(str::to_owned);

            let error = run(args).expect_err("CPU GGUF must fail before setup");
            let message = error.to_string();
            assert!(message.contains("CPU MiniMax-H3 host forward requires F32"));
            assert!(message.contains("lower quantization does not fix this routing"));
            assert!(!workdir.exists());
        }
        assert!(!isolated_hf_home.exists());
    }

    #[test]
    fn cpu_dit_guard_accepts_safetensors_and_leaves_cuda_gguf_supported() {
        validate_args(&parse(&[])).expect("CUDA GGUF remains structurally accepted");
        validate_args(&parse(&[
            "--device",
            "cpu",
            "--dit-file",
            "nested/minimax_h3_fl2va_pruned_nvfp4.safetensors",
        ]))
        .expect("CPU Safetensors remains structurally accepted");

        let mixed_case = parse(&[
            "--device",
            "cpu",
            "--dit-file",
            "nested/minimax_h3_fl2va_pruned-Q2_K.GgUf",
        ]);
        assert!(validate_args(&mixed_case).is_err());
    }

    #[test]
    fn ref2va_selects_matching_default_dit_and_requires_a_reference() {
        let without_reference = parse(&["--partition", "ref2va"]);
        assert!(validate_args(&without_reference)
            .unwrap_err()
            .to_string()
            .contains("requires a reference"));

        let mut with_reference = without_reference;
        with_reference.reference_audio = Some(PathBuf::from("reference.wav"));
        assert_eq!(artifact_sources(&with_reference)[0].path, REF2VA_DIT);
    }

    #[test]
    fn rejects_known_opposite_dit_for_effective_partition() {
        let nested_ref2va_dit = format!("nested/{REF2VA_DIT}");
        for dit in [REF2VA_DIT, nested_ref2va_dit.as_str()] {
            let fl2va_with_ref2va_dit = parse(&["--dit-file", dit]);
            assert!(validate_args(&fl2va_with_ref2va_dit)
                .unwrap_err()
                .to_string()
                .contains("cannot serve --partition fl2va"));
        }

        let nested_fl2va_dit = format!("nested/{FL2VA_DIT}");
        for dit in [FL2VA_DIT, nested_fl2va_dit.as_str()] {
            let mut ref2va_with_fl2va_dit = parse(&["--partition", "ref2va", "--dit-file", dit]);
            ref2va_with_fl2va_dit.reference_audio = Some(PathBuf::from("reference.wav"));
            assert!(validate_args(&ref2va_with_fl2va_dit)
                .unwrap_err()
                .to_string()
                .contains("cannot serve --partition ref2va"));
        }

        validate_args(&parse(&["--dit-file", "nested/renamed.gguf"]))
            .expect("arbitrarily renamed DiT remains the user's responsibility");
    }

    #[test]
    fn rejects_partition_mismatches_before_hub_resolution() {
        let mut fl2va_with_reference = parse(&[]);
        fl2va_with_reference.reference_image = Some(PathBuf::from("reference.ppm"));
        assert!(validate_args(&fl2va_with_reference)
            .unwrap_err()
            .to_string()
            .contains("--partition ref2va"));

        let mut ref2va_with_keyframe = parse(&["--partition", "ref2va"]);
        ref2va_with_keyframe.first_frame = Some(PathBuf::from("first.ppm"));
        ref2va_with_keyframe.reference_audio = Some(PathBuf::from("reference.wav"));
        assert!(validate_args(&ref2va_with_keyframe)
            .unwrap_err()
            .to_string()
            .contains("cannot be combined"));
    }

    #[test]
    fn prepares_new_and_existing_workdirs_without_removing_content() {
        let temp = TempDir::new();
        let new_workdir = temp.0.join("new").join("nested");
        drop(prepare_workdir(&new_workdir).expect("create writable workdir"));
        assert!(new_workdir.is_dir());

        let nonmatching_names = [
            "marker",
            "audio.wav.bak",
            "frame_00000.ppm",
            "frame_00000x.ppm",
            "frame_000000.ppm.bak",
            "xframe_000000.ppm",
        ];
        for name in nonmatching_names {
            fs::write(temp.0.join(name), b"keep").expect("write nonmatching marker");
        }
        drop(prepare_workdir(&temp.0).expect("accept writable existing directory"));
        for name in nonmatching_names {
            assert_eq!(
                fs::read(temp.0.join(name)).expect("read nonmatching marker"),
                b"keep"
            );
        }
    }

    #[test]
    fn workdir_lock_excludes_concurrent_acquisition_and_cleans_up() {
        let temp = TempDir::new();
        let lock_path = temp.0.join(WORKDIR_LOCK_FILE);

        let mut first = prepare_workdir(&temp.0).expect("first lock acquisition succeeds");
        assert!(fs::read_to_string(&lock_path)
            .expect("read lock context")
            .contains(&std::process::id().to_string()));

        let error = prepare_workdir(&temp.0).expect_err("second acquisition must fail");
        assert!(error.to_string().contains("already in use"));
        assert!(lock_path.is_file());

        first
            .remove_owned_file()
            .expect("explicit owned-lock cleanup succeeds");
        assert!(!lock_path.exists());

        let reacquired = prepare_workdir(&temp.0).expect("acquire after normal cleanup");
        drop(reacquired);
        assert!(!lock_path.exists());
    }

    #[test]
    fn preserves_a_preexisting_workdir_lock() {
        let temp = TempDir::new();
        let lock_path = temp.0.join(WORKDIR_LOCK_FILE);
        fs::write(&lock_path, b"owned elsewhere").expect("write pre-existing lock");

        let error = prepare_workdir(&temp.0).expect_err("pre-existing lock must exclude run");
        assert!(error.to_string().contains("already in use"));
        assert_eq!(
            fs::read(&lock_path).expect("read preserved lock"),
            b"owned elsewhere"
        );
    }

    #[cfg(unix)]
    #[test]
    fn dropping_a_removed_lock_does_not_remove_its_replacement() {
        let temp = TempDir::new();
        let lock_path = temp.0.join(WORKDIR_LOCK_FILE);
        let original = prepare_workdir(&temp.0).expect("acquire original lock");

        fs::remove_file(&lock_path).expect("externally remove original lock path");
        fs::write(&lock_path, b"replacement owner").expect("install replacement lock");
        drop(original);

        assert_eq!(
            fs::read(&lock_path).expect("replacement lock survives original guard drop"),
            b"replacement owner"
        );
    }

    #[cfg(unix)]
    #[test]
    fn preserves_a_preexisting_symlink_lock_without_following_it() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new();
        let sentinel = temp.0.join("sentinel");
        let lock_path = temp.0.join(WORKDIR_LOCK_FILE);
        fs::write(&sentinel, b"keep").expect("write symlink target");
        symlink(&sentinel, &lock_path).expect("create pre-existing lock symlink");

        let error = prepare_workdir(&temp.0).expect_err("symlink lock must exclude run");
        assert!(error.to_string().contains("already in use"));
        assert!(fs::symlink_metadata(&lock_path)
            .expect("inspect preserved symlink")
            .file_type()
            .is_symlink());
        assert_eq!(fs::read(&sentinel).expect("read sentinel"), b"keep");
    }

    #[test]
    fn early_failures_do_not_reach_hub_resolution() {
        let temp = TempDir::new();
        let invalid_workdir = temp.0.join("must-not-be-created");
        let mut invalid_args = parse(&["--steps", "1"]);
        invalid_args.workdir = invalid_workdir.clone();
        invalid_args.gguf_repo = "invalid/repository/shape".to_owned();
        let error = run(invalid_args).expect_err("argument validation fails first");
        assert!(error.to_string().contains("at least 2"));
        assert!(!invalid_workdir.exists());

        let lock_path = temp.0.join(WORKDIR_LOCK_FILE);
        fs::write(&lock_path, b"owned elsewhere").expect("write competing lock");
        // Use a structurally valid CPU DiT selection to reach workdir validation
        // even when this example is built without CUDA. No model is loaded.
        let mut locked_args = parse(&["--device", "cpu", "--dit-file", "model.safetensors"]);
        locked_args.workdir = temp.0.clone();
        locked_args.gguf_repo = "invalid/repository/shape".to_owned();
        let error = run(locked_args).expect_err("workdir lock fails before Hub resolution");
        assert!(error.to_string().contains("already in use"));
        assert_eq!(
            fs::read(&lock_path).expect("read competing lock"),
            b"owned elsewhere"
        );
    }

    #[test]
    fn rejects_existing_audio_output_files_and_directories() {
        let temp = TempDir::new();
        for entry_kind in ["file", "directory"] {
            let workdir = temp.0.join(entry_kind);
            fs::create_dir(&workdir).expect("create workdir fixture");
            let output = workdir.join("audio.wav");
            if entry_kind == "file" {
                fs::write(&output, b"keep").expect("write audio collision");
            } else {
                fs::create_dir(&output).expect("create audio directory collision");
            }

            let error = prepare_workdir(&workdir).expect_err("reject existing audio output");
            assert!(error.to_string().contains("audio.wav"));
            assert!(!workdir.join(WORKDIR_LOCK_FILE).exists());
            if entry_kind == "file" {
                assert_eq!(fs::read(&output).expect("read audio collision"), b"keep");
            } else {
                assert!(output.is_dir());
            }
        }
    }

    #[test]
    fn rejects_existing_frame_output_files_and_directories() {
        let temp = TempDir::new();
        assert!(is_native_frame_output_name("frame_000000.ppm"));
        assert!(is_native_frame_output_name("frame_1000000.ppm"));

        for entry_kind in ["file", "directory"] {
            let workdir = temp.0.join(entry_kind);
            fs::create_dir(&workdir).expect("create workdir fixture");
            let output = workdir.join("frame_000000.ppm");
            if entry_kind == "file" {
                fs::write(&output, b"keep").expect("write frame collision");
            } else {
                fs::create_dir(&output).expect("create frame directory collision");
            }

            let error = prepare_workdir(&workdir).expect_err("reject existing frame output");
            assert!(error.to_string().contains("frame_000000.ppm"));
            assert!(!workdir.join(WORKDIR_LOCK_FILE).exists());
            if entry_kind == "file" {
                assert_eq!(fs::read(&output).expect("read frame collision"), b"keep");
            } else {
                assert!(output.is_dir());
            }
        }
    }

    #[test]
    fn rejects_file_and_invalid_parent_as_workdir() {
        let temp = TempDir::new();
        let file = temp.0.join("not-a-directory");
        fs::write(&file, b"content").expect("write fixture");
        assert!(prepare_workdir(&file)
            .unwrap_err()
            .to_string()
            .contains("is not a directory"));

        let child = file.join("child");
        assert!(prepare_workdir(&child)
            .unwrap_err()
            .to_string()
            .contains("could not inspect or prepare --workdir"));
    }

    #[test]
    fn accepts_native_compatible_reference_wav_layouts() {
        let temp = TempDir::new();
        let fixtures = [
            (
                "mono.wav",
                vec![
                    (*b"fmt ", pcm_format(1, 1, 32_000, 16)),
                    (*b"data", vec![0, 0]),
                ],
            ),
            (
                "stereo.wav",
                vec![
                    (*b"fmt ", pcm_format(1, 2, 32_000, 16)),
                    (*b"data", vec![0, 0, 1, 0]),
                ],
            ),
            (
                "padded-unknown.wav",
                vec![
                    (*b"fmt ", pcm_format(1, 3, 32_000, 16)),
                    (*b"JUNK", vec![1, 2, 3]),
                    (*b"data", vec![0, 0, 1, 0, 2, 0]),
                ],
            ),
        ];
        for (name, chunks) in fixtures {
            let path = temp.0.join(name);
            write_wav(&path, chunks);
            validate_reference_wav(&path).expect("native-compatible WAV");
        }
    }

    #[test]
    fn rejects_incompatible_reference_wav_formats() {
        let temp = TempDir::new();
        for (name, format, expected) in [
            (
                "wrong-rate.wav",
                pcm_format(1, 2, 44_100, 16),
                "exactly 32000 Hz",
            ),
            (
                "compressed.wav",
                pcm_format(3, 2, 32_000, 16),
                "format tag 1",
            ),
            (
                "wrong-depth.wav",
                pcm_format(1, 2, 32_000, 24),
                "16-bit PCM",
            ),
            (
                "zero-channels.wav",
                pcm_format(1, 0, 32_000, 16),
                "zero channels",
            ),
        ] {
            let path = temp.0.join(name);
            write_wav(
                &path,
                vec![(*b"fmt ", format), (*b"data", vec![0, 0, 0, 0])],
            );
            assert!(validate_reference_wav(&path)
                .unwrap_err()
                .to_string()
                .contains(expected));
        }
    }

    #[test]
    fn rejects_missing_truncated_and_empty_reference_wav_chunks() {
        let temp = TempDir::new();
        let mut truncated_fmt_header = wav_fixture(vec![(*b"JUNK", vec![0; 17])]);
        truncated_fmt_header.extend_from_slice(b"fmt \x10\0");
        let mut truncated_fmt_body = wav_fixture(vec![(*b"JUNK", vec![0; 8])]);
        truncated_fmt_body.extend_from_slice(b"fmt \x10\0\0\0\x01\0\x01\0\0\x7d\0\0");
        let mut truncated_data_header = wav_fixture(vec![
            (*b"fmt ", pcm_format(1, 1, 32_000, 16)),
            (*b"JUNK", vec![1]),
        ]);
        truncated_data_header.extend_from_slice(b"data\x02\0\0");
        let mut truncated_data_body = wav_fixture(vec![
            (*b"fmt ", pcm_format(1, 2, 32_000, 16)),
            (*b"data", vec![0, 0]),
        ]);
        truncated_data_body[40..44].copy_from_slice(&4_u32.to_le_bytes());
        let mut oversized_unknown = b"RIFF\0\0\0\0WAVEJUNK".to_vec();
        oversized_unknown.extend_from_slice(&u32::MAX.to_le_bytes());
        oversized_unknown.resize(44, 0);

        let fixtures = [
            ("short.wav", b"RIFF".to_vec(), "too short"),
            ("not-wave.wav", vec![b'X'; 44], "not a RIFF/WAVE file"),
            (
                "missing-fmt.wav",
                wav_fixture(vec![(*b"JUNK", vec![0; 20]), (*b"data", vec![0, 0])]),
                "no complete fmt chunk",
            ),
            (
                "truncated-fmt-header.wav",
                truncated_fmt_header,
                "no complete fmt chunk",
            ),
            (
                "truncated-fmt-body.wav",
                truncated_fmt_body,
                "no complete fmt chunk",
            ),
            (
                "missing-data.wav",
                wav_fixture(vec![
                    (*b"fmt ", pcm_format(1, 1, 32_000, 16)),
                    (*b"JUNK", vec![0, 0]),
                ]),
                "no data chunk",
            ),
            (
                "truncated-data-header.wav",
                truncated_data_header,
                "no data chunk",
            ),
            (
                "data-before-fmt.wav",
                wav_fixture(vec![
                    (*b"JUNK", vec![0; 12]),
                    (*b"data", vec![0, 0]),
                    (*b"fmt ", pcm_format(1, 1, 32_000, 16)),
                ]),
                "no complete fmt chunk",
            ),
            (
                "empty-data.wav",
                wav_fixture(vec![
                    (*b"fmt ", pcm_format(1, 1, 32_000, 16)),
                    (*b"data", Vec::new()),
                ]),
                "no complete source frame",
            ),
            (
                "incomplete-stereo-frame.wav",
                wav_fixture(vec![
                    (*b"fmt ", pcm_format(1, 2, 32_000, 16)),
                    (*b"data", vec![0, 0]),
                ]),
                "no complete source frame",
            ),
            (
                "truncated-data-body.wav",
                truncated_data_body,
                "no complete source frame",
            ),
            (
                "oversized-unknown.wav",
                oversized_unknown,
                "no complete fmt chunk",
            ),
        ];
        for (name, bytes, expected) in fixtures {
            let path = temp.0.join(name);
            fs::write(&path, bytes).expect("write malformed WAV fixture");
            assert!(validate_reference_wav(&path)
                .unwrap_err()
                .to_string()
                .contains(expected));
        }
    }

    #[test]
    fn invalid_reference_wav_fails_before_workdir_or_hub_side_effects() {
        let _env_lock = ENV_LOCK.lock().expect("lock environment");
        let temp = TempDir::new();
        let reference = temp.0.join("invalid.wav");
        let workdir = temp.0.join("must-not-be-created");
        let isolated_hf_home = temp.0.join("hf-home");
        let _hf_home = EnvVarGuard::set("HF_HOME", &isolated_hf_home);
        fs::write(&reference, b"not a wave").expect("write invalid reference");

        let mut args = parse(&["--partition", "ref2va"]);
        args.reference_audio = Some(reference);
        args.workdir = workdir.clone();
        args.gguf_repo = "must/not/be/resolved".to_owned();
        let error = run(args).expect_err("invalid WAV must fail before setup");
        assert!(error.to_string().contains("invalid --reference-audio"));
        assert!(!workdir.exists());
        assert!(!isolated_hf_home.exists());
    }

    #[test]
    fn existing_mp4_fails_before_workdir_or_hub_side_effects() {
        let _env_lock = ENV_LOCK.lock().expect("lock environment");
        let temp = TempDir::new();
        let output = temp.0.join("existing.mp4");
        let workdir = temp.0.join("must-not-be-created");
        let isolated_hf_home = temp.0.join("hf-home");
        let _hf_home = EnvVarGuard::set("HF_HOME", &isolated_hf_home);
        fs::write(&output, b"keep").expect("write existing output");

        let mut args = parse(&[]);
        args.output_mp4 = Some(output.clone());
        args.workdir = workdir.clone();
        args.gguf_repo = "must/not/be/resolved".to_owned();
        let error = run(args).expect_err("existing MP4 must fail before setup");
        assert!(error.to_string().contains("already exists"));
        assert_eq!(fs::read(output).expect("existing output remains"), b"keep");
        assert!(!workdir.exists());
        assert!(!isolated_hf_home.exists());
    }

    #[test]
    fn accepts_a_nonexistent_mp4_in_a_resolvable_parent() {
        let temp = TempDir::new();
        let workdir = temp.0.join("workdir");
        fs::create_dir(&workdir).expect("create workdir");
        let output = temp.0.join("new.mp4");
        validate_mp4_output(&output, &workdir).expect("safe new output");
        assert!(!output.exists());
    }

    #[test]
    fn rejects_every_kind_of_existing_mp4_entry() {
        let temp = TempDir::new();
        let workdir = temp.0.join("workdir");
        fs::create_dir(&workdir).expect("create workdir");

        let file = temp.0.join("file.mp4");
        fs::write(&file, b"keep").expect("write existing output");
        let directory = temp.0.join("directory.mp4");
        fs::create_dir(&directory).expect("create existing output directory");
        for output in [&file, &directory] {
            assert!(validate_mp4_output(output, &workdir)
                .unwrap_err()
                .to_string()
                .contains("already exists"));
        }
        assert_eq!(fs::read(file).expect("existing file remains"), b"keep");
        assert!(directory.is_dir());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_existing_mp4_symlinks_including_dangling_links() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new();
        let workdir = temp.0.join("workdir");
        fs::create_dir(&workdir).expect("create workdir");
        let target = temp.0.join("target");
        fs::write(&target, b"keep").expect("write symlink target");
        let existing_link = temp.0.join("existing.mp4");
        symlink(&target, &existing_link).expect("create output symlink");
        let dangling_link = temp.0.join("dangling.mp4");
        symlink(temp.0.join("missing"), &dangling_link).expect("create dangling output symlink");

        for output in [&existing_link, &dangling_link] {
            assert!(validate_mp4_output(output, &workdir)
                .unwrap_err()
                .to_string()
                .contains("already exists"));
            assert!(fs::symlink_metadata(output)
                .expect("symlink remains")
                .file_type()
                .is_symlink());
        }
        assert_eq!(fs::read(target).expect("target remains"), b"keep");
    }

    #[test]
    fn rejects_direct_reserved_workdir_mp4_aliases() {
        let temp = TempDir::new();
        let workdir = temp.0.join("workdir");
        fs::create_dir(&workdir).expect("create workdir");
        for name in [
            WORKDIR_LOCK_FILE,
            "audio.wav",
            "frame_000000.ppm",
            "frame_1000000.ppm",
        ] {
            let error = validate_mp4_output(&workdir.join(name), &workdir)
                .expect_err("reserved output must be rejected");
            assert!(error.to_string().contains("reserved workdir output"));
        }
    }

    #[cfg(unix)]
    #[test]
    fn rejects_reserved_mp4_aliases_through_a_symlinked_parent() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new();
        let workdir = temp.0.join("workdir");
        fs::create_dir(&workdir).expect("create workdir");
        let alias = temp.0.join("workdir-alias");
        symlink(&workdir, &alias).expect("create workdir alias");
        for name in [WORKDIR_LOCK_FILE, "audio.wav", "frame_000000.ppm"] {
            let error = validate_mp4_output(&alias.join(name), &workdir)
                .expect_err("symlink-parent alias must be rejected");
            assert!(error.to_string().contains("reserved workdir output"));
        }
    }

    #[test]
    fn rejects_unresolvable_mp4_parents() {
        let temp = TempDir::new();
        let workdir = temp.0.join("workdir");
        fs::create_dir(&workdir).expect("create workdir");
        let output = temp.0.join("missing-parent").join("output.mp4");
        assert!(validate_mp4_output(&output, &workdir)
            .unwrap_err()
            .to_string()
            .contains("could not safely resolve"));
        assert!(!output.exists());
    }

    #[test]
    fn reuses_commit_snapshot_from_normal_mutable_revision_cache() {
        let temp = TempDir::new();
        let cache = Cache::new(temp.0.join("hub"));
        let repo_id = "owner/model";
        let commit = "0123456789abcdef";
        let artifact_path = "nested/model.gguf";
        let repo = cache.repo(Repo::new(repo_id.to_owned(), RepoType::Model));
        repo.create_ref(commit).expect("create mutable main ref");
        let snapshot_path = repo.pointer_path(commit).join(artifact_path);
        fs::create_dir_all(snapshot_path.parent().expect("snapshot parent"))
            .expect("create snapshot directory");
        fs::write(&snapshot_path, b"cached artifact").expect("write snapshot artifact");

        let commit_ref = cache
            .path()
            .join(Repo::new(repo_id.to_owned(), RepoType::Model).folder_name())
            .join("refs")
            .join(commit);
        assert!(!commit_ref.exists(), "fixture must not contain refs/<sha>");

        let fallback_called = Cell::new(false);
        let resolved = cached_commit_snapshot_or_else(
            &cache,
            repo_id,
            commit,
            artifact_path,
            || -> AppResult<PathBuf> {
                fallback_called.set(true);
                Err(app_error("fallback must not run for a cached snapshot"))
            },
        )
        .expect("reuse commit snapshot");

        assert_eq!(resolved, snapshot_path);
        assert!(!fallback_called.get());
    }

    #[test]
    fn missing_or_non_file_commit_snapshot_uses_fallback() {
        let temp = TempDir::new();
        let cache = Cache::new(temp.0.join("hub"));
        let repo_id = "owner/model";
        let commit = "0123456789abcdef";
        let fallback_path = temp.0.join("downloaded.gguf");
        fs::write(&fallback_path, b"downloaded artifact").expect("write fallback artifact");

        for artifact_path in ["missing/model.gguf", "directory/model.gguf"] {
            if artifact_path.starts_with("directory/") {
                let directory_path = cache
                    .repo(Repo::new(repo_id.to_owned(), RepoType::Model))
                    .pointer_path(commit)
                    .join(artifact_path);
                fs::create_dir_all(&directory_path).expect("create non-file snapshot entry");
            }

            let fallback_called = Cell::new(false);
            let resolved = cached_commit_snapshot_or_else(
                &cache,
                repo_id,
                commit,
                artifact_path,
                || -> AppResult<PathBuf> {
                    fallback_called.set(true);
                    Ok(fallback_path.clone())
                },
            )
            .expect("use fallback");

            assert_eq!(resolved, fallback_path);
            assert!(fallback_called.get());
        }
    }

    #[test]
    fn selects_only_nonempty_environment_hf_token() {
        assert_eq!(selected_hf_token(None), None);
        assert_eq!(selected_hf_token(Some(" \t".to_owned())), None);
        assert_eq!(
            selected_hf_token(Some(" token ".to_owned())),
            Some(" token ".to_owned())
        );
    }

    #[cfg(unix)]
    fn write_fake_ffmpeg(path: &Path, body: &str) {
        use std::os::unix::fs::PermissionsExt;

        fs::write(path, body).expect("write fake ffmpeg");
        let mut permissions = fs::metadata(path)
            .expect("fake ffmpeg metadata")
            .permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(path, permissions).expect("make fake ffmpeg executable");
    }

    fn private_mux_paths(parent: &Path) -> Vec<PathBuf> {
        fs::read_dir(parent)
            .expect("read mux parent")
            .map(|entry| entry.expect("read mux-parent entry").path())
            .filter(|path| {
                path.file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with(PRIVATE_MUX_DIR_PREFIX))
            })
            .collect()
    }

    #[cfg(unix)]
    #[test]
    fn mux_mp4_uses_private_output_preserves_arguments_and_cleans_up() {
        let temp = TempDir::new();
        let frame_dir = temp.0.join("frames with spaces");
        let audio = frame_dir.join("audio input.wav");
        let output = temp.0.join("output video.mp4");
        let executable = temp.0.join("fake-ffmpeg.sh");
        let capture = temp.0.join("fake-ffmpeg.sh.args");

        write_fake_ffmpeg(
            &executable,
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$0.args\"\n[ \"$1\" = '-n' ] || exit 97\nfor output do :; done\n[ ! -e \"$output\" ] || exit 98\nprintf private-mp4 > \"$output\"\n",
        );
        validate_mp4_output(&output, &frame_dir).expect("output passes preflight");
        mux_mp4(&frame_dir, &audio, 24, &output, &executable).expect("fake ffmpeg success");

        let captured_text = fs::read_to_string(&capture).expect("read captured arguments");
        let captured = captured_text.lines().collect::<Vec<_>>();
        let private_path = PathBuf::from(captured.last().expect("captured private output"));
        let mut expected = compose_video_mux_argv(
            &VideoMuxParams::new(frame_dir.join("frame_%06d.ppm"), &private_path)
                .audio_path(&audio)
                .fps(24),
        )
        .expect("compose expected arguments")
        .into_args()
        .into_iter()
        .skip(1)
        .collect::<Vec<_>>();
        assert_eq!(expected.first().and_then(|arg| arg.to_str()), Some("-y"));
        expected[0] = "-n".into();
        assert_eq!(
            captured,
            expected
                .iter()
                .map(|argument| argument.to_str().expect("UTF-8 test argument"))
                .collect::<Vec<_>>()
        );
        assert!(!captured
            .iter()
            .any(|argument| *argument == output.to_str().unwrap()));
        assert_eq!(
            private_path.file_name().and_then(|name| name.to_str()),
            Some("mux.mp4")
        );
        assert!(private_path
            .parent()
            .and_then(Path::file_name)
            .is_some_and(|name| name.to_string_lossy().starts_with(PRIVATE_MUX_DIR_PREFIX)));
        assert_eq!(
            fs::read(&output).expect("read atomically published output"),
            b"private-mp4"
        );
        assert!(private_mux_paths(&temp.0).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn mux_mp4_refuses_destination_created_after_preflight() {
        let temp = TempDir::new();
        let frame_dir = temp.0.join("frames");
        let audio = frame_dir.join("audio.wav");
        let executable = temp.0.join("fake-racing-ffmpeg.sh");
        let mut output = executable.as_os_str().to_os_string();
        output.push(".raced-output");
        let output = PathBuf::from(output);

        write_fake_ffmpeg(
            &executable,
            "#!/bin/sh\nfor output do :; done\nprintf private-mp4 > \"$output\"\nprintf raced-destination > \"$0.raced-output\"\n",
        );
        validate_mp4_output(&output, &frame_dir).expect("output passes preflight");
        let error = mux_mp4(&frame_dir, &audio, 24, &output, &executable)
            .expect_err("atomic publication must refuse raced destination");

        assert!(error
            .to_string()
            .contains("appeared before atomic publication"));
        assert_eq!(
            fs::read(&output).expect("read preserved raced destination"),
            b"raced-destination"
        );
        assert!(private_mux_paths(&temp.0).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn mux_mp4_failure_cleans_private_output_and_leaves_no_destination() {
        let temp = TempDir::new();
        let frame_dir = temp.0.join("frames");
        let audio = frame_dir.join("audio.wav");
        let output = temp.0.join("must-not-exist.mp4");
        let executable = temp.0.join("fake-failing-ffmpeg.sh");

        write_fake_ffmpeg(
            &executable,
            "#!/bin/sh\nfor output do :; done\nprintf partial > \"$output\"\nexit 23\n",
        );
        let error = mux_mp4(&frame_dir, &audio, 24, &output, &executable)
            .expect_err("ffmpeg failure must propagate");

        assert!(error.to_string().contains("status 23"));
        assert!(!output.exists());
        assert!(private_mux_paths(&temp.0).is_empty());
    }

    #[test]
    fn rejects_unsafe_hub_inputs_without_network_access() {
        for (flag, value) in [
            ("--encoder-repo", "../model"),
            ("--encoder-revision", "/revision"),
            ("--encoder-file", "owner//encoder.gguf"),
        ] {
            let args = parse(&[flag, value]);
            let error = validate_artifact_sources(&artifact_sources(&args))
                .expect_err("unsafe encoder source must fail before network access");
            assert!(error.to_string().contains("prompt encoder"));
        }

        for (value, label) in [
            ("", "empty"),
            ("../model", "parent"),
            ("/model", "absolute"),
            ("owner//model", "empty component"),
            ("model\\file", "backslash"),
        ] {
            assert!(
                validate_hub_path(value, label).is_err(),
                "unexpectedly accepted {label}"
            );
        }
        validate_repo_id("owner/model", "test").expect("valid repository");
        validate_hub_path("nested/model.safetensors", "test").expect("valid artifact path");
    }
}
