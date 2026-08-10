use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use vllm_cpp::{Engine, FinishReason, SamplingParams, StreamControl, StructuredOutput};

const REQUIRED_MODEL_FILES: [&str; 4] = [
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
];

fn model_path() -> Option<PathBuf> {
    let path = std::env::var_os("VLLM_CPP_TEST_MODEL").map(PathBuf::from)?;
    let missing = REQUIRED_MODEL_FILES
        .iter()
        .filter(|file| !path.join(file).is_file())
        .copied()
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "VLLM_CPP_TEST_MODEL fixture is incomplete at {}: missing {}",
        path.display(),
        missing.join(", ")
    );
    Some(path)
}

fn with_engine(test: impl FnOnce(&Engine, &Path)) {
    let Some(path) = model_path() else {
        eprintln!("skipping model test; set VLLM_CPP_TEST_MODEL with `just setup-test-model`");
        return;
    };
    static ENGINE: OnceLock<Mutex<Engine>> = OnceLock::new();
    let engine = ENGINE.get_or_init(|| {
        Mutex::new(
            Engine::builder(&path)
                .num_blocks(64)
                .max_model_len(256)
                .max_num_seqs(2)
                .max_num_batched_tokens(256)
                .load()
                .expect("load Qwen3-0.6B"),
        )
    });
    let engine = engine.lock().expect("model test engine lock");
    test(&engine, &path);
}

#[test]
fn greedy_completion_and_streaming_match() {
    with_engine(|engine, _| {
        let params = SamplingParams::greedy().max_tokens(8);
        let completion = engine
            .complete("The capital of France is", &params)
            .expect("blocking completion");
        assert!(!completion.text.is_empty());
        assert_eq!(completion.completion_tokens, 8);
        assert_eq!(completion.finish_reason, Some(FinishReason::Length));

        let mut streamed = String::new();
        let outcome = engine
            .complete_stream("The capital of France is", &params, |event| {
                streamed.push_str(&event.delta);
                StreamControl::Continue
            })
            .expect("streaming completion");
        assert!(!outcome.stopped_by_callback);
        assert_eq!(streamed, completion.text);
    });
}

#[test]
fn seeded_sampling_is_repeatable() {
    with_engine(|engine, _| {
        let params = SamplingParams::default()
            .temperature(0.8)
            .seed(42)
            .max_tokens(8);
        let first = engine
            .complete("A surprising fact about rust is", &params)
            .expect("first seeded completion");
        let second = engine
            .complete("A surprising fact about rust is", &params)
            .expect("second seeded completion");
        assert_eq!(first, second);
    });
}

#[test]
fn early_stop_and_callback_panic_leave_engine_reusable() {
    with_engine(|engine, _| {
        let params = SamplingParams::greedy().max_tokens(8);
        let outcome = engine
            .complete_stream("Count from one to ten:", &params, |_| StreamControl::Stop)
            .expect("early stop");
        assert!(outcome.stopped_by_callback);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = engine.complete_stream("Say hello", &params, |_| {
                panic!("intentional callback panic")
            });
        }));
        assert!(panic.is_err());

        let completion = engine
            .complete("Say hello", &SamplingParams::greedy().max_tokens(2))
            .expect("engine remains reusable");
        assert!(!completion.text.is_empty());
    });
}

#[test]
fn structured_choice_is_enforced() {
    with_engine(|engine, _| {
        let params =
            SamplingParams::greedy()
                .max_tokens(8)
                .structured_output(StructuredOutput::Choice(vec![
                    "red".to_owned(),
                    "blue".to_owned(),
                ]));
        let completion = engine
            .complete("Choose exactly one color: red or blue. Answer:", &params)
            .expect("structured completion");
        assert!(
            completion.text.trim() == "red" || completion.text.trim() == "blue",
            "unexpected choice: {:?}",
            completion.text
        );
    });
}

#[test]
fn terminal_stop_is_natural_finish_for_completion_and_chat() {
    with_engine(|engine, _| {
        let mut completion_finished = false;
        let completion_outcome = engine
            .complete_stream(
                "Reply with hello.",
                &SamplingParams::greedy().max_tokens(4),
                |event| {
                    completion_finished |= event.finished;
                    if event.finished {
                        StreamControl::Stop
                    } else {
                        StreamControl::Continue
                    }
                },
            )
            .expect("terminal completion stop");
        assert!(completion_finished);
        assert!(!completion_outcome.stopped_by_callback);

        let request = r#"{
            "messages":[{"role":"user","content":"Reply with hello."}],
            "temperature":0,
            "max_tokens":4
        }"#;
        let mut chat_finished = false;
        let chat_outcome = engine
            .chat_stream_json(request, |event| {
                chat_finished |= event.finished;
                if event.finished {
                    StreamControl::Stop
                } else {
                    StreamControl::Continue
                }
            })
            .expect("terminal chat stop");
        assert!(chat_finished);
        assert!(!chat_outcome.stopped_by_callback);
    });
}

#[test]
fn blocking_and_streaming_chat_return_json() {
    with_engine(|engine, _| {
        let request = r#"{
            "messages":[{"role":"user","content":"Reply with hello."}],
            "temperature":0,
            "max_tokens":4
        }"#;
        let response = engine.chat_json(request).expect("blocking chat");
        let json: serde_json::Value = serde_json::from_str(&response).expect("valid response JSON");
        assert_eq!(json["object"], "chat.completion");
        assert!(json["choices"]
            .as_array()
            .is_some_and(|choices| !choices.is_empty()));

        let mut chunks = Vec::new();
        let outcome = engine
            .chat_stream_json(request, |event| {
                if !event.finished {
                    let chunk: serde_json::Value =
                        serde_json::from_str(&event.delta).expect("valid chunk JSON");
                    chunks.push(chunk);
                }
                StreamControl::Continue
            })
            .expect("streaming chat");
        assert!(!outcome.stopped_by_callback);
        assert!(!chunks.is_empty());
    });
}
