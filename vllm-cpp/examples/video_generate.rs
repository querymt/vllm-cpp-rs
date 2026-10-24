use std::error::Error;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use vllm_cpp::{VideoDevice, VideoEngine, VideoGenerationParams, VideoPartition};

#[derive(Debug, Parser)]
#[command(
    name = "video_generate",
    about = "Generate MiniMax-H3 PPM frames and WAV audio from local checkpoints",
    long_about = "Demonstrate blocking video generation with local checkpoints and a fresh output directory. No downloads or ffmpeg execution. CPU requires an F32 DiT; quantized CUDA checkpoints require a compatible CUDA build and GPU. Checkpoint, partition and media compatibility remain native checks. Partial output is left in place on failure."
)]
struct Args {
    #[arg(long)]
    prompt: String,
    /// Local DiT weights compatible with the selected partition and device.
    #[arg(long, value_name = "FILE")]
    dit: PathBuf,
    #[arg(long, value_name = "FILE")]
    encoder: PathBuf,
    #[arg(long, value_name = "FILE")]
    tokenizer: PathBuf,
    #[arg(long, value_name = "FILE")]
    video_vae: PathBuf,
    #[arg(long, value_name = "JSON")]
    video_vae_config: PathBuf,
    #[arg(long, value_name = "FILE")]
    audio_vae: PathBuf,
    #[arg(long, value_name = "JSON")]
    audio_vae_config: PathBuf,
    /// New directory in an existing parent; existing entries are never reused.
    #[arg(long, value_name = "DIR")]
    output_dir: PathBuf,
    #[arg(long, value_enum, default_value = "cpu")]
    device: DeviceArg,
    #[arg(long, value_enum, default_value = "fl2va")]
    partition: PartitionArg,
    /// P6 PPM image required by this example's Ref2VA mode.
    #[arg(long, value_name = "PPM")]
    reference_image: Option<PathBuf>,
    #[arg(long, default_value_t = 256)]
    width: u32,
    #[arg(long, default_value_t = 256)]
    height: u32,
    /// Native code rounds up to the next 17n+5 frame count.
    #[arg(long, default_value_t = 5)]
    frames: u32,
    #[arg(long, default_value_t = 2)]
    steps: u32,
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PartitionArg {
    Fl2va,
    Ref2va,
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

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("video_generate: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let generation = prepare(&args)?;
    let mut engine = VideoEngine::builder(&args.dit)
        .encoder_path(&args.encoder)
        .tokenizer_path(&args.tokenizer)
        .video_vae_path(&args.video_vae)
        .video_vae_config_path(&args.video_vae_config)
        .audio_vae_path(&args.audio_vae)
        .audio_vae_config_path(&args.audio_vae_config)
        .partition(args.partition.into())
        .device(args.device.into())
        .load()?;
    let result = engine.generate(&generation)?;
    println!(
        "wrote {} PPM frames to {} ({}x{} at {} fps)",
        result.frame_count(),
        result.frame_dir().display(),
        result.width(),
        result.height(),
        result.fps()
    );
    println!("wrote WAV audio to {}", result.audio_path().display());
    Ok(())
}

fn generation_params(args: &Args) -> VideoGenerationParams {
    let mut params = VideoGenerationParams::new(&args.prompt, &args.output_dir)
        .dimensions(args.width, args.height)
        .num_frames(args.frames)
        .steps(args.steps);
    if let Some(seed) = args.seed {
        params = params.seed(seed);
    }
    if let Some(path) = &args.reference_image {
        params = params.reference_image(path);
    }
    params
}

fn prepare(args: &Args) -> Result<VideoGenerationParams, Box<dyn Error>> {
    let generation = generation_params(args);
    generation.validate()?;
    if args.prompt.trim().is_empty() {
        return Err("--prompt must not be empty".into());
    }
    if (args.partition == PartitionArg::Ref2va) != args.reference_image.is_some() {
        return Err("use --reference-image with --partition ref2va; omit it for fl2va".into());
    }
    // System libraries configure their backends independently of Cargo features.
    if args.device == DeviceArg::Cuda && cfg!(feature = "bundled") && !cfg!(feature = "cuda") {
        return Err(
            "--device cuda requires a CUDA-enabled build; no CPU fallback is performed".into(),
        );
    }
    for (flag, path) in [
        ("--dit", &args.dit),
        ("--encoder", &args.encoder),
        ("--tokenizer", &args.tokenizer),
        ("--video-vae", &args.video_vae),
        ("--video-vae-config", &args.video_vae_config),
        ("--audio-vae", &args.audio_vae),
        ("--audio-vae-config", &args.audio_vae_config),
    ]
    .into_iter()
    .chain(
        args.reference_image
            .as_ref()
            .map(|path| ("--reference-image", path)),
    ) {
        if !path.is_file() {
            return Err(format!("{flag} must name a local file: {}", path.display()).into());
        }
    }
    // create_dir atomically refuses existing entries, including symlinks. Native
    // generation trusts these caller paths; this is not a filesystem sandbox.
    fs::create_dir(&args.output_dir).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "cannot create --output-dir {}: {error}; choose a fresh directory in an existing parent",
                args.output_dir.display()
            ),
        )
    })?;
    Ok(generation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "vllm-local-video-{}-{}",
                std::process::id(),
                NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn args(&self, extra: &[&str]) -> Args {
            let file = self.0.join("checkpoint");
            fs::write(&file, b"local test placeholder, not a model").unwrap();
            let mut args = vec![
                "video_generate".into(),
                "--prompt".into(),
                "a shoreline".into(),
            ];
            for flag in [
                "--dit",
                "--encoder",
                "--tokenizer",
                "--video-vae",
                "--video-vae-config",
                "--audio-vae",
                "--audio-vae-config",
            ] {
                args.extend([
                    std::ffi::OsString::from(flag),
                    file.clone().into_os_string(),
                ]);
            }
            args.extend([
                "--output-dir".into(),
                self.0.join("output").into_os_string(),
            ]);
            args.extend(extra.iter().map(std::ffi::OsString::from));
            Args::try_parse_from(args).unwrap()
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn parses_local_defaults_and_explicit_generation_options() {
        let temp = TempDir::new();
        let defaults = temp.args(&[]);
        assert_eq!(defaults.device, DeviceArg::Cpu);
        assert_eq!(defaults.partition, PartitionArg::Fl2va);
        let args = temp.args(&[
            "--width", "320", "--height", "192", "--frames", "22", "--steps", "4", "--seed", "42",
            "--device", "cuda",
        ]);
        assert_eq!(
            (args.width, args.height, args.frames, args.steps, args.seed),
            (320, 192, 22, 4, Some(42))
        );
        assert_eq!(VideoDevice::from(args.device), VideoDevice::Cuda);
        assert_eq!(
            VideoPartition::from(PartitionArg::Ref2va),
            VideoPartition::Ref2va
        );
        generation_params(&args).validate().unwrap();
        assert!(!args.output_dir.exists());
    }

    #[test]
    fn help_and_missing_arguments_exit_at_the_parser() {
        assert_eq!(
            Args::try_parse_from(["video_generate", "--help"])
                .unwrap_err()
                .kind(),
            clap::error::ErrorKind::DisplayHelp
        );
        assert_eq!(
            Args::try_parse_from(["video_generate"]).unwrap_err().kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
    }

    #[test]
    fn invalid_parameters_and_missing_files_have_no_output_side_effects() {
        let temp = TempDir::new();
        for extra in [
            vec!["--width", "0"],
            vec!["--frames", "1"],
            vec!["--steps", "0"],
        ] {
            let args = temp.args(&extra);
            let output = args.output_dir.clone();
            assert!(run(args).is_err());
            assert!(!output.exists());
        }
        let mut args = temp.args(&[]);
        args.prompt.clear();
        let output = args.output_dir.clone();
        assert!(run(args).unwrap_err().to_string().contains("--prompt"));
        assert!(!output.exists());
        let mut args = temp.args(&[]);
        args.dit = temp.0.join("missing");
        assert!(prepare(&args).unwrap_err().to_string().contains("--dit"));
        assert!(!args.output_dir.exists());
        if cfg!(feature = "bundled") && !cfg!(feature = "cuda") {
            let args = temp.args(&["--device", "cuda"]);
            let output = args.output_dir.clone();
            assert!(run(args)
                .unwrap_err()
                .to_string()
                .contains("CUDA-enabled build"));
            assert!(!output.exists());
        }
    }

    #[test]
    fn fresh_output_is_created_and_existing_entries_are_preserved() {
        let temp = TempDir::new();
        let args = temp.args(&[]);
        prepare(&args).unwrap();
        let marker = args.output_dir.join("keep");
        fs::write(&marker, b"unchanged").unwrap();
        assert!(run(args)
            .unwrap_err()
            .to_string()
            .contains("fresh directory"));
        assert_eq!(fs::read(marker).unwrap(), b"unchanged");
        let mut args = temp.args(&[]);
        args.output_dir = temp.0.join("checkpoint");
        let before = fs::read(&args.output_dir).unwrap();
        assert!(prepare(&args).is_err());
        assert_eq!(fs::read(&args.output_dir).unwrap(), before);
    }

    #[test]
    fn ref2va_requires_a_local_reference_image() {
        let temp = TempDir::new();
        let mut args = temp.args(&[]);
        args.reference_image = Some(args.dit.clone());
        assert!(prepare(&args).is_err());
        assert!(!args.output_dir.exists());
        args.partition = PartitionArg::Ref2va;
        args.reference_image = None;
        assert!(prepare(&args).is_err());
        assert!(!args.output_dir.exists());
        args.reference_image = Some(args.dit.clone());
        prepare(&args).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn existing_output_symlinks_are_not_followed() {
        let temp = TempDir::new();
        let args = temp.args(&[]);
        let missing = temp.0.join("absent");
        std::os::unix::fs::symlink(&missing, &args.output_dir).unwrap();
        assert!(prepare(&args).is_err());
        assert!(args.output_dir.is_symlink());
        assert!(!missing.exists());
    }
}
