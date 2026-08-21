mod common;

use std::io::{self, Write};

use vllm_cpp::{Engine, SamplingParams, StreamControl};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = common::resolve_model("stream")?;
    let engine = Engine::load(model)?;
    engine.complete_stream(
        "Write one short sentence about Rust:",
        &SamplingParams::greedy().max_tokens(32),
        |event| {
            print!("{}", event.delta);
            io::stdout().flush().expect("flush stdout");
            StreamControl::Continue
        },
    )?;
    println!();
    Ok(())
}
