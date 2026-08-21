use vllm_cpp::{Engine, SamplingParams, StructuredOutput};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args_os()
        .nth(1)
        .ok_or("usage: structured <model-directory>")?;
    let engine = Engine::load(model)?;
    let completion = engine.complete(
        "Choose exactly one color: red or blue. Answer:",
        &SamplingParams::greedy()
            .max_tokens(8)
            .structured_output(StructuredOutput::Choice(vec![
                "red".to_owned(),
                "blue".to_owned(),
            ])),
    )?;
    println!("{}", completion.text);
    Ok(())
}
