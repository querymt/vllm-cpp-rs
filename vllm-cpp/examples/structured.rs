mod common;

use vllm_cpp::{Engine, SamplingParams, StructuredOutput};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = common::resolve_model("structured")?;
    let engine = Engine::load(model)?;
    let schema = r#"{
        "type": "object",
        "properties": {
            "location": { "type": "string" },
            "temperature_celsius": { "type": "number" },
            "condition": { "type": "string" }
        },
        "required": ["location", "temperature_celsius", "condition"],
        "additionalProperties": false
    }"#;
    let completion = engine.complete(
        "Extract the weather report as JSON: Paris is sunny and 22 degrees Celsius.",
        &SamplingParams::greedy()
            .max_tokens(64)
            .structured_output(StructuredOutput::JsonSchema(schema.to_owned())),
    )?;
    println!("{}", completion.text);
    Ok(())
}
