use std::process::ExitCode;

use vllm_cpp::HuggingFaceModel;

const REPOSITORY: &str = "Qwen/Qwen3-0.6B";
const REVISION: &str = "c1899de289a04d12100db370d81485cdf75e47ca";

fn main() -> ExitCode {
    match HuggingFaceModel::safetensors(REPOSITORY)
        .revision(REVISION)
        .progress(true)
        .resolve()
    {
        Ok(path) => {
            println!("{}", path.display());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("could not set up test model: {error}");
            ExitCode::FAILURE
        }
    }
}
