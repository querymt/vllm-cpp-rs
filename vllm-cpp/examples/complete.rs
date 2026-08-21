use vllm_cpp::{Engine, SamplingParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args_os()
        .nth(1)
        .ok_or("usage: complete <model-directory>")?;
    let engine = Engine::load(model)?;
    let completion = engine.complete(
        "The capital of France is",
        &SamplingParams::greedy().max_tokens(16),
    )?;
    println!("{}", completion.text);
    Ok(())
}
