mod common;

use std::error::Error;
use std::fmt;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use vllm_cpp::{Engine, StreamControl};

use common::ModelSource;

#[derive(Debug, Parser)]
#[command(
    name = "chat",
    about = "Chat interactively with a local or Hugging Face model",
    override_usage = "chat [OPTIONS] <MODEL>\n       chat [OPTIONS] <COMMAND>",
    subcommand_negates_reqs = true
)]
struct Args {
    /// Bare local model directory or GGUF path (an alias for `local <PATH>`).
    #[arg(value_name = "MODEL", required = true)]
    model_path: Option<PathBuf>,

    #[command(subcommand)]
    model: Option<Model>,

    /// Initial user message.
    #[arg(
        short = 'p',
        long,
        global = true,
        conflicts_with = "file",
        value_name = "TEXT"
    )]
    prompt: Option<String>,

    /// Read the initial user message from a UTF-8 file.
    #[arg(
        short = 'f',
        long,
        global = true,
        conflicts_with = "prompt",
        value_name = "PATH"
    )]
    file: Option<PathBuf>,

    /// System message added at the start of the conversation.
    #[arg(long, global = true, value_name = "TEXT")]
    system: Option<String>,

    /// Maximum tokens generated per response, from 1 through 2147483647.
    #[arg(long, global = true, default_value_t = 256, value_parser = parse_max_tokens)]
    max_tokens: u32,

    /// Sampling temperature in [0, 2].
    #[arg(long, global = true, default_value_t = 0.7, value_parser = parse_temperature)]
    temperature: f64,

    /// Nucleus-sampling probability in (0, 1].
    #[arg(long, global = true, default_value_t = 1.0, value_parser = parse_top_p)]
    top_p: f64,

    /// Number of top tokens to consider; use 0 or -1 for all tokens.
    #[arg(long, global = true, default_value_t = 0, allow_hyphen_values = true, value_parser = parse_top_k)]
    top_k: i32,

    /// Minimum token probability relative to the most likely token, in [0, 1].
    #[arg(long, global = true, default_value_t = 0.0, value_parser = parse_min_p)]
    min_p: f64,

    /// Random seed for generation.
    #[arg(long, global = true, allow_hyphen_values = true)]
    seed: Option<i64>,

    /// Wait for each complete response instead of printing streamed deltas.
    #[arg(long, global = true)]
    no_stream: bool,
}

#[derive(Debug, Subcommand)]
enum Model {
    /// Use a local model directory or standalone GGUF file.
    Local {
        #[arg(value_name = "PATH")]
        path: PathBuf,
    },

    /// Download one GGUF file from Hugging Face or reuse its cached copy.
    #[command(name = "hf-gguf")]
    HuggingFaceGguf {
        /// Hugging Face repository, for example `owner/model`.
        #[arg(value_name = "REPO")]
        repo: String,
        /// Root-level GGUF filename in the repository.
        #[arg(value_name = "FILENAME")]
        filename: String,
        /// Branch, tag, or commit; defaults to the mutable `main` revision.
        #[arg(long, value_name = "REVISION")]
        revision: Option<String>,
    },

    /// Download a runtime-complete Safetensors snapshot from Hugging Face.
    #[command(name = "hf-safetensors")]
    HuggingFaceSafetensors {
        /// Hugging Face repository, for example `owner/model`.
        #[arg(value_name = "REPO")]
        repo: String,
        /// Branch, tag, or commit; defaults to the mutable `main` revision.
        #[arg(long, value_name = "REVISION")]
        revision: Option<String>,
    },
}

impl Model {
    fn into_source(self) -> ModelSource {
        match self {
            Self::Local { path } => ModelSource::Local(path),
            Self::HuggingFaceGguf {
                repo,
                filename,
                revision,
            } => ModelSource::Gguf {
                repo,
                filename,
                revision,
            },
            Self::HuggingFaceSafetensors { repo, revision } => {
                ModelSource::Safetensors { repo, revision }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Message {
    role: Role,
    content: String,
}

impl Message {
    fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct GenerationConfig {
    max_tokens: u32,
    temperature: f64,
    top_p: f64,
    top_k: i32,
    min_p: f64,
    seed: Option<i64>,
}

#[derive(Debug)]
enum ChatError {
    Turn(String),
    Fatal(String),
}

impl ChatError {
    fn turn(message: impl Into<String>) -> Self {
        Self::Turn(message.into())
    }

    fn fatal(message: impl Into<String>) -> Self {
        Self::Fatal(message.into())
    }

    const fn is_fatal(&self) -> bool {
        matches!(self, Self::Fatal(_))
    }
}

impl fmt::Display for ChatError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Turn(message) | Self::Fatal(message) => formatter.write_str(message),
        }
    }
}

impl Error for ChatError {}

type ChatResult<T> = Result<T, ChatError>;

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("chat: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> ChatResult<()> {
    let Args {
        model_path,
        model,
        prompt,
        file,
        system,
        max_tokens,
        temperature,
        top_p,
        top_k,
        min_p,
        seed,
        no_stream,
    } = args;

    let initial_prompt = read_initial_prompt(prompt, file)?;
    let source = match (model_path, model) {
        (Some(path), None) => ModelSource::Local(path),
        (None, Some(model)) => model.into_source(),
        _ => return Err(ChatError::fatal("exactly one model source is required")),
    };
    let model = source
        .resolve()
        .map_err(|error| ChatError::fatal(format!("failed to resolve model source: {error}")))?;
    let engine = Engine::load(model)
        .map_err(|error| ChatError::fatal(format!("failed to load model: {error}")))?;
    let config = GenerationConfig {
        max_tokens,
        temperature,
        top_p,
        top_k,
        min_p,
        seed,
    };

    let mut history = Vec::new();
    if let Some(system) = system {
        history.push(Message::new(Role::System, system));
    }

    write_stdout_line("Interactive chat: use /clear to reset the conversation or /quit to exit.")?;
    if let Some(prompt) = initial_prompt {
        handle_turn(
            &mut history,
            prompt,
            |history, prompt| run_turn(&engine, history, &config, !no_stream, prompt),
            report_turn_error,
        )?;
    }
    interactive_loop(&engine, &mut history, &config, !no_stream)
}

fn read_initial_prompt(
    prompt: Option<String>,
    file: Option<PathBuf>,
) -> ChatResult<Option<String>> {
    match (prompt, file) {
        (Some(prompt), None) => Ok(Some(prompt)),
        (None, Some(path)) => fs::read_to_string(&path).map(Some).map_err(|error| {
            ChatError::fatal(format!(
                "failed to read UTF-8 prompt file `{}`: {error}",
                path.display()
            ))
        }),
        (None, None) => Ok(None),
        (Some(_), Some(_)) => Err(ChatError::fatal(
            "--prompt and --file cannot be used together",
        )),
    }
}

fn interactive_loop(
    engine: &Engine,
    history: &mut Vec<Message>,
    config: &GenerationConfig,
    stream: bool,
) -> ChatResult<()> {
    let stdin = io::stdin();
    let mut stdin = stdin.lock();
    let mut line = String::new();

    loop {
        {
            let stdout = io::stdout();
            let mut stdout = stdout.lock();
            write!(stdout, "user> ")
                .and_then(|()| stdout.flush())
                .map_err(|error| {
                    ChatError::fatal(format!("failed to write input prompt: {error}"))
                })?;
        }

        line.clear();
        let bytes = stdin
            .read_line(&mut line)
            .map_err(|error| ChatError::fatal(format!("failed to read standard input: {error}")))?;
        if bytes == 0 {
            write_stdout_line("")?;
            return Ok(());
        }

        let input = line.trim_end_matches(['\r', '\n']);
        match classify_input(input) {
            Input::Quit => return Ok(()),
            Input::Clear => {
                history.retain(|message| message.role == Role::System);
                write_stdout_line("Conversation cleared.")?;
            }
            Input::Empty => {}
            Input::Message(message) => {
                handle_turn(
                    history,
                    message.to_owned(),
                    |history, message| run_turn(engine, history, config, stream, message),
                    report_turn_error,
                )?;
            }
        }
    }
}

fn run_turn(
    engine: &Engine,
    history: &mut Vec<Message>,
    config: &GenerationConfig,
    stream: bool,
    user_message: String,
) -> ChatResult<()> {
    run_history_turn(history, user_message, |history| {
        let request = build_request(history, config, stream)?;
        if stream {
            write_assistant_prefix()?;
            let result = stream_assistant(engine, &request);
            let content = result.as_ref().map_or("", |assistant| assistant.as_str());
            finish_assistant_output(content)?;
            result
        } else {
            engine
                .chat_json(&request)
                .map_err(|error| ChatError::turn(format!("blocking chat request failed: {error}")))
                .and_then(|response| extract_blocking_content(&response))
                .and_then(|assistant| {
                    write_assistant_prefix()?;
                    {
                        let stdout = io::stdout();
                        let mut stdout = stdout.lock();
                        stdout.write_all(assistant.as_bytes()).map_err(|error| {
                            ChatError::fatal(format!("failed to write assistant response: {error}"))
                        })?;
                    }
                    finish_assistant_output(&assistant)?;
                    Ok(assistant)
                })
        }
    })
}

fn run_history_turn<F>(
    history: &mut Vec<Message>,
    user_message: String,
    run_assistant: F,
) -> ChatResult<()>
where
    F: FnOnce(&mut Vec<Message>) -> ChatResult<String>,
{
    let checkpoint = history.len();
    history.push(Message::new(Role::User, user_message));
    match run_assistant(history) {
        Ok(assistant) => {
            history.push(Message::new(Role::Assistant, assistant));
            Ok(())
        }
        Err(error) => {
            rollback_turn(history, checkpoint);
            Err(error)
        }
    }
}

fn handle_turn<F, R>(
    history: &mut Vec<Message>,
    user_message: String,
    run: F,
    mut report_error: R,
) -> ChatResult<()>
where
    F: FnOnce(&mut Vec<Message>, String) -> ChatResult<()>,
    R: FnMut(&ChatError),
{
    match run(history, user_message) {
        Ok(()) => Ok(()),
        Err(error) if error.is_fatal() => Err(error),
        Err(error) => {
            report_error(&error);
            Ok(())
        }
    }
}

fn report_turn_error(error: &ChatError) {
    eprintln!("chat: {error}");
}

fn rollback_turn(history: &mut Vec<Message>, checkpoint: usize) {
    history.truncate(checkpoint);
}

fn stream_assistant(engine: &Engine, request: &str) -> ChatResult<String> {
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    let mut assistant = String::new();
    let mut callback_error = None;

    let outcome = engine.chat_stream_json(request, |event| {
        if callback_error.is_some() {
            return StreamControl::Stop;
        }
        if event.finished {
            return StreamControl::Continue;
        }

        match accumulate_stream_content(&mut assistant, &event.delta) {
            Ok(delta) => {
                if let Err(error) = stdout
                    .write_all(delta.as_bytes())
                    .and_then(|()| stdout.flush())
                {
                    callback_error = Some(ChatError::fatal(format!(
                        "failed to write streaming assistant response: {error}"
                    )));
                    StreamControl::Stop
                } else {
                    StreamControl::Continue
                }
            }
            Err(error) => {
                callback_error = Some(error);
                StreamControl::Stop
            }
        }
    });
    drop(stdout);

    if let Some(error) = callback_error {
        return Err(error);
    }
    let outcome = outcome
        .map_err(|error| ChatError::turn(format!("streaming chat request failed: {error}")))?;
    complete_stream(assistant, outcome.stopped_by_callback)
}

fn accumulate_stream_content(assistant: &mut String, chunk: &str) -> ChatResult<String> {
    let content = extract_stream_content(chunk)?;
    assistant.push_str(&content);
    Ok(content)
}

fn complete_stream(assistant: String, stopped_by_callback: bool) -> ChatResult<String> {
    if stopped_by_callback {
        Err(ChatError::turn(
            "streaming chat response stopped unexpectedly",
        ))
    } else {
        Ok(assistant)
    }
}

fn write_stdout_line(message: &str) -> ChatResult<()> {
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    writeln!(stdout, "{message}")
        .and_then(|()| stdout.flush())
        .map_err(|error| ChatError::fatal(format!("failed to write standard output: {error}")))
}

fn write_assistant_prefix() -> ChatResult<()> {
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    write!(stdout, "assistant> ")
        .and_then(|()| stdout.flush())
        .map_err(|error| ChatError::fatal(format!("failed to write assistant prompt: {error}")))
}

fn finish_assistant_output(content: &str) -> ChatResult<()> {
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    if !content.ends_with('\n') {
        writeln!(stdout).map_err(|error| {
            ChatError::fatal(format!("failed to finish assistant response: {error}"))
        })?;
    }
    stdout
        .flush()
        .map_err(|error| ChatError::fatal(format!("failed to flush assistant response: {error}")))
}

fn build_request(
    messages: &[Message],
    config: &GenerationConfig,
    stream: bool,
) -> ChatResult<String> {
    let messages: Vec<_> = messages
        .iter()
        .map(|message| {
            json!({
                "role": message.role.as_str(),
                "content": message.content,
            })
        })
        .collect();
    let mut request = json!({
        "messages": messages,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "min_p": config.min_p,
        "stream": stream,
    });
    if let Some(seed) = config.seed {
        request["seed"] = json!(seed);
    }
    serde_json::to_string(&request)
        .map_err(|error| ChatError::turn(format!("failed to serialize chat request: {error}")))
}

fn extract_blocking_content(response: &str) -> ChatResult<String> {
    let response: Value = serde_json::from_str(response).map_err(|error| {
        ChatError::turn(format!("invalid blocking chat response JSON: {error}"))
    })?;
    let choices = response
        .get("choices")
        .and_then(Value::as_array)
        .ok_or_else(|| ChatError::turn("blocking chat response is missing array `choices`"))?;
    let choice = choices
        .first()
        .and_then(Value::as_object)
        .ok_or_else(|| ChatError::turn("blocking chat response is missing object `choices[0]`"))?;
    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ChatError::turn("blocking chat response is missing object `choices[0].message`")
        })?;

    match message.get("content") {
        Some(Value::String(content)) => Ok(content.clone()),
        None | Some(Value::Null) => Ok(String::new()),
        Some(_) => Err(ChatError::turn(
            "blocking chat response has non-string `choices[0].message.content`",
        )),
    }
}

fn extract_stream_content(chunk: &str) -> ChatResult<String> {
    let chunk: Value = serde_json::from_str(chunk)
        .map_err(|error| ChatError::turn(format!("invalid streaming chat chunk JSON: {error}")))?;
    let choices = chunk
        .get("choices")
        .and_then(Value::as_array)
        .ok_or_else(|| ChatError::turn("streaming chat chunk is missing array `choices`"))?;
    let mut content = String::new();
    for (index, choice) in choices.iter().enumerate() {
        let delta = choice
            .get("delta")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                ChatError::turn(format!(
                    "streaming chat chunk is missing object `choices[{index}].delta`"
                ))
            })?;
        match delta.get("content") {
            Some(Value::String(part)) => content.push_str(part),
            None | Some(Value::Null) => {}
            Some(_) => {
                return Err(ChatError::turn(format!(
                    "streaming chat chunk has non-string `choices[{index}].delta.content`"
                )))
            }
        }
    }
    Ok(content)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Input<'a> {
    Quit,
    Clear,
    Empty,
    Message(&'a str),
}

fn classify_input(input: &str) -> Input<'_> {
    match input.trim() {
        "/quit" | "/exit" => Input::Quit,
        "/clear" => Input::Clear,
        "" => Input::Empty,
        _ => Input::Message(input),
    }
}

fn parse_max_tokens(value: &str) -> Result<u32, String> {
    let value = value
        .parse::<u32>()
        .map_err(|error| format!("invalid token count: {error}"))?;
    if value == 0 {
        return Err("max tokens must be greater than zero".to_owned());
    }
    if value > i32::MAX as u32 {
        return Err(format!("max tokens must not exceed {}", i32::MAX));
    }
    Ok(value)
}

fn parse_temperature(value: &str) -> Result<f64, String> {
    parse_float_range(value, "temperature", 0.0, 2.0, false)
}

fn parse_top_p(value: &str) -> Result<f64, String> {
    parse_float_range(value, "top-p", 0.0, 1.0, true)
}

fn parse_top_k(value: &str) -> Result<i32, String> {
    let value = value
        .parse::<i32>()
        .map_err(|error| format!("invalid top-k value: {error}"))?;
    if value < -1 {
        return Err("top-k must be -1, 0, or a positive integer".to_owned());
    }
    Ok(value)
}

fn parse_min_p(value: &str) -> Result<f64, String> {
    parse_float_range(value, "min-p", 0.0, 1.0, false)
}

fn parse_float_range(
    value: &str,
    name: &str,
    minimum: f64,
    maximum: f64,
    minimum_is_exclusive: bool,
) -> Result<f64, String> {
    let value = value
        .parse::<f64>()
        .map_err(|error| format!("invalid {name} value: {error}"))?;
    let below_minimum = if minimum_is_exclusive {
        value <= minimum
    } else {
        value < minimum
    };
    if !value.is_finite() || below_minimum || value > maximum {
        let opening = if minimum_is_exclusive { "(" } else { "[" };
        return Err(format!(
            "{name} must be finite and in {opening}{minimum}, {maximum}]"
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use clap::error::ErrorKind;

    use super::*;

    fn source(args: Args) -> ModelSource {
        match (args.model_path, args.model) {
            (Some(path), None) => ModelSource::Local(path),
            (None, Some(model)) => model.into_source(),
            _ => panic!("invalid parsed model source"),
        }
    }

    #[test]
    fn parses_bare_and_local_model_forms_with_global_options() {
        let args = Args::try_parse_from(["chat", "--prompt", "hello", "model.gguf", "--no-stream"])
            .unwrap();
        assert_eq!(source(args), ModelSource::Local("model.gguf".into()));

        let args = Args::try_parse_from([
            "chat",
            "--system",
            "be concise",
            "local",
            "model-directory",
            "--temperature",
            "0.25",
        ])
        .unwrap();
        assert_eq!(args.temperature, 0.25);
        assert_eq!(source(args), ModelSource::Local("model-directory".into()));
    }

    #[test]
    fn rejects_missing_model_source() {
        assert!(Args::try_parse_from(["chat"]).is_err());
    }

    #[test]
    fn parses_hugging_face_forms_and_revisions() {
        let gguf = Args::try_parse_from([
            "chat",
            "hf-gguf",
            "owner/model",
            "model.gguf",
            "--max-tokens",
            "64",
        ])
        .unwrap();
        assert_eq!(gguf.max_tokens, 64);
        assert_eq!(
            source(gguf),
            ModelSource::Gguf {
                repo: "owner/model".to_owned(),
                filename: "model.gguf".to_owned(),
                revision: None,
            }
        );

        let safetensors = Args::try_parse_from([
            "chat",
            "--top-p",
            "0.9",
            "hf-safetensors",
            "owner/model",
            "--revision",
            "release",
            "--min-p",
            "0.05",
        ])
        .unwrap();
        assert_eq!(safetensors.min_p, 0.05);
        assert_eq!(
            source(safetensors),
            ModelSource::Safetensors {
                repo: "owner/model".to_owned(),
                revision: Some("release".to_owned()),
            }
        );
    }

    #[test]
    fn parses_supported_sampling_options() {
        let args = Args::try_parse_from([
            "chat",
            "local",
            "model",
            "--max-tokens",
            "512",
            "--temperature",
            "1.25",
            "--top-p",
            "0.8",
            "--top-k",
            "-1",
            "--min-p",
            "0.1",
            "--seed",
            "-2",
        ])
        .unwrap();
        assert_eq!(args.max_tokens, 512);
        assert_eq!(args.temperature, 1.25);
        assert_eq!(args.top_p, 0.8);
        assert_eq!(args.top_k, -1);
        assert_eq!(args.min_p, 0.1);
        assert_eq!(args.seed, Some(-2));
    }

    #[test]
    fn uses_documented_generation_defaults() {
        let args = Args::try_parse_from(["chat", "local", "model"]).unwrap();
        assert_eq!(args.max_tokens, 256);
        assert_eq!(args.temperature, 0.7);
        assert_eq!(args.top_p, 1.0);
        assert_eq!(args.top_k, 0);
        assert_eq!(args.min_p, 0.0);
        assert_eq!(args.seed, None);
        assert!(!args.no_stream);
    }

    #[test]
    fn rejects_prompt_file_conflict_and_invalid_sampling_values() {
        let error = Args::try_parse_from([
            "chat",
            "local",
            "model",
            "--prompt",
            "hello",
            "--file",
            "prompt.txt",
        ])
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ArgumentConflict);

        for arguments in [
            vec!["chat", "local", "model", "--max-tokens", "0"],
            vec!["chat", "local", "model", "--max-tokens", "2147483648"],
            vec!["chat", "local", "model", "--temperature", "nan"],
            vec!["chat", "local", "model", "--temperature", "2.1"],
            vec!["chat", "local", "model", "--top-p", "0"],
            vec!["chat", "local", "model", "--top-k", "-2"],
            vec!["chat", "local", "model", "--min-p", "1.1"],
        ] {
            assert!(Args::try_parse_from(arguments).is_err());
        }
        assert!(
            Args::try_parse_from(["chat", "local", "model", "--max-tokens", "2147483647",]).is_ok()
        );
    }

    #[test]
    fn rolls_back_failed_turn_and_continues_with_the_next_input() {
        let mut history = vec![Message::new(Role::System, "system")];
        let mut errors = Vec::new();
        let mut attempted_messages = Vec::new();

        handle_turn(
            &mut history,
            "failed question".to_owned(),
            |history, message| {
                attempted_messages.push(message.clone());
                run_history_turn(history, message, |_history| {
                    Err(ChatError::turn("request failed"))
                })
            },
            |error| errors.push(error.to_string()),
        )
        .unwrap();
        assert_eq!(history, vec![Message::new(Role::System, "system")]);
        assert_eq!(attempted_messages, ["failed question"]);
        assert_eq!(errors, ["request failed"]);

        handle_turn(
            &mut history,
            "next question".to_owned(),
            |history, message| {
                run_history_turn(history, message, |_history| Ok("answer".to_owned()))
            },
            |_| panic!("successful turn must not report an error"),
        )
        .unwrap();
        assert_eq!(
            history,
            vec![
                Message::new(Role::System, "system"),
                Message::new(Role::User, "next question"),
                Message::new(Role::Assistant, "answer"),
            ]
        );
    }

    #[test]
    fn propagates_fatal_errors_instead_of_recovering() {
        let mut history = Vec::new();
        let mut reported = false;
        let error = handle_turn(
            &mut history,
            "question".to_owned(),
            |_history, _message| Err(ChatError::fatal("standard output failed")),
            |_| reported = true,
        )
        .unwrap_err();
        assert!(error.is_fatal());
        assert!(!reported);
    }

    #[test]
    fn builds_request_with_history_and_sampling_configuration() {
        let messages = vec![
            Message::new(Role::System, "Be concise."),
            Message::new(Role::User, "Hello"),
            Message::new(Role::Assistant, "Hi"),
        ];
        let request = build_request(
            &messages,
            &GenerationConfig {
                max_tokens: 42,
                temperature: 0.5,
                top_p: 0.9,
                top_k: 10,
                min_p: 0.1,
                seed: Some(7),
            },
            true,
        )
        .unwrap();
        let request: Value = serde_json::from_str(&request).unwrap();
        assert_eq!(request["messages"][0]["role"], "system");
        assert_eq!(request["messages"][1]["content"], "Hello");
        assert_eq!(request["messages"][2]["content"], "Hi");
        assert_eq!(request["max_tokens"], 42);
        assert_eq!(request["temperature"], 0.5);
        assert_eq!(request["top_p"], 0.9);
        assert_eq!(request["top_k"], 10);
        assert_eq!(request["min_p"], 0.1);
        assert_eq!(request["seed"], 7);
        assert_eq!(request["stream"], true);
    }

    #[test]
    fn extracts_native_blocking_response_content() {
        let response = r#"{
            "id":"chatcmpl-test",
            "object":"chat.completion",
            "choices":[{
                "index":0,
                "message":{"role":"assistant","content":"hello"},
                "finish_reason":"stop"
            }]
        }"#;
        assert_eq!(extract_blocking_content(response).unwrap(), "hello");

        let reasoning_only = r#"{
            "id":"chatcmpl-test",
            "object":"chat.completion",
            "choices":[{
                "index":0,
                "message":{
                    "role":"assistant",
                    "content":null,
                    "reasoning":"private reasoning"
                },
                "finish_reason":"stop"
            }]
        }"#;
        assert_eq!(extract_blocking_content(reasoning_only).unwrap(), "");

        let missing_content = r#"{
            "choices":[{
                "message":{
                    "role":"assistant",
                    "tool_calls":[{
                        "id":"chatcmpl-tool-0",
                        "type":"function",
                        "function":{"name":"weather","arguments":"{}"}
                    }]
                },
                "finish_reason":"tool_calls"
            }]
        }"#;
        assert_eq!(extract_blocking_content(missing_content).unwrap(), "");
    }

    #[test]
    fn rejects_malformed_blocking_response_shapes() {
        for response in [
            r#"{}"#,
            r#"{"choices":"not an array"}"#,
            r#"{"choices":[]}"#,
            r#"{"choices":[{"message":"not an object"}]}"#,
            r#"{"choices":[{"message":{"content":42}}]}"#,
            "not json",
        ] {
            assert!(extract_blocking_content(response).is_err());
        }
    }

    #[test]
    fn extracts_native_stream_chunks_and_accepts_empty_content() {
        let chunks = [
            r#"{"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning":"private"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null},{"index":1,"delta":{"content":"!"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        ];
        let mut content = String::new();
        for chunk in chunks {
            assert_eq!(
                accumulate_stream_content(&mut content, chunk).unwrap(),
                extract_stream_content(chunk).unwrap()
            );
        }
        assert_eq!(content, "Hello!");

        let empty_chunks = [
            r#"{"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"reasoning":"private"},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"weather"}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":null},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ];
        let mut empty = String::new();
        for chunk in empty_chunks {
            assert_eq!(accumulate_stream_content(&mut empty, chunk).unwrap(), "");
        }
        assert_eq!(complete_stream(empty, false).unwrap(), "");

        let mut history = vec![Message::new(Role::User, "question")];
        run_history_turn(&mut history, "next".to_owned(), |_history| {
            complete_stream(String::new(), false)
        })
        .unwrap();
        assert_eq!(history.last(), Some(&Message::new(Role::Assistant, "")));
    }

    #[test]
    fn rejects_malformed_stream_content_types_and_shapes() {
        for chunk in [
            r#"{}"#,
            r#"{"choices":"not an array"}"#,
            r#"{"choices":[{}]}"#,
            r#"{"choices":[{"delta":"not an object"}]}"#,
            r#"{"choices":[{"delta":{"content":42}}]}"#,
            "not json",
        ] {
            assert!(extract_stream_content(chunk).is_err());
        }
        assert!(complete_stream(String::new(), true).is_err());
    }

    #[test]
    fn classifies_interactive_commands_without_trimming_messages() {
        assert_eq!(classify_input(" /quit "), Input::Quit);
        assert_eq!(classify_input("/exit"), Input::Quit);
        assert_eq!(classify_input("/clear"), Input::Clear);
        assert_eq!(classify_input("   "), Input::Empty);
        assert_eq!(classify_input(" hello "), Input::Message(" hello "));
    }
}
