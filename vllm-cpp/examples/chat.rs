use vllm_cpp::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args_os()
        .nth(1)
        .ok_or("usage: chat <model-directory>")?;
    let engine = Engine::load(model)?;
    let response = engine.chat_json(
        r#"{
            "messages": [{"role": "user", "content": "Reply with hello."}],
            "temperature": 0,
            "max_tokens": 16
        }"#,
    )?;
    println!("{response}");
    Ok(())
}
