mod common;

use std::io::{self, Write};

use vllm_cpp::{Engine, SamplingParams, StreamControl};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = common::resolve_model("concurrent")?;
    let engine = Engine::load(model)?;
    let params = SamplingParams::greedy().max_tokens(16);

    let mut france = engine.submit("The capital of France is", &params, |event| {
        print!("[france] {}", event.delta);
        io::stdout().flush().expect("flush stdout");
        StreamControl::Continue
    })?;
    let mut germany = engine.submit("The capital of Germany is", &params, |event| {
        print!("[germany] {}", event.delta);
        io::stdout().flush().expect("flush stdout");
        StreamControl::Continue
    })?;

    println!("\nfrance: {:?}", france.wait()?);
    println!("germany: {:?}", germany.wait()?);
    Ok(())
}
