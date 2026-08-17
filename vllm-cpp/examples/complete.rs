mod common;

use vllm_cpp::{Engine, SamplingParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = common::resolve_model("complete")?;
    let engine = Engine::load(model)?;
    let completion = engine.complete(
        "The capital of France is",
        &SamplingParams::greedy().max_tokens(16),
    )?;
    println!("{}", completion.text);
    Ok(())
}
