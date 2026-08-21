use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Barrier, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use vllm_cpp::{
    Engine, Error, FinishReason, Request, RequestOutcome, SamplingParams, StreamControl,
    StructuredOutput,
};

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
    if std::env::var_os("VLLM_CPP_TEST_ISOLATED_ENGINE").is_some() {
        let engine = load_engine(&path);
        test(&engine, &path);
        return;
    }
    static ENGINE: OnceLock<Mutex<Engine>> = OnceLock::new();
    let engine = ENGINE.get_or_init(|| Mutex::new(load_engine(&path)));
    let engine = engine.lock().expect("model test engine lock");
    test(&engine, &path);
}

fn load_engine(path: &Path) -> Engine {
    Engine::builder(path)
        .num_blocks(64)
        .max_model_len(256)
        .max_num_seqs(2)
        .max_num_batched_tokens(256)
        .load()
        .expect("load Qwen3-0.6B")
}

fn wait_until_done(request: &Request) {
    let deadline = Instant::now() + Duration::from_secs(180);
    while !request.is_done() {
        assert!(
            Instant::now() < deadline,
            "request did not finish before timeout"
        );
        thread::sleep(Duration::from_millis(1));
    }
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
fn concurrent_requests_batch_with_correct_output() {
    with_engine(|engine, _| {
        let params = SamplingParams::greedy().max_tokens(12);
        let expected_first = engine
            .complete("The capital of France is", &params)
            .expect("first reference completion")
            .text;
        let expected_second = engine
            .complete("The capital of Germany is", &params)
            .expect("second reference completion")
            .text;

        let first_text = Arc::new(Mutex::new(String::new()));
        let second_text = Arc::new(Mutex::new(String::new()));
        let mut first = engine
            .submit("The capital of France is", &params, {
                let text = Arc::clone(&first_text);
                move |event| {
                    text.lock()
                        .expect("first callback text")
                        .push_str(&event.delta);
                    StreamControl::Continue
                }
            })
            .expect("submit first request");
        let mut second = engine
            .submit("The capital of Germany is", &params, {
                let text = Arc::clone(&second_text);
                move |event| {
                    text.lock()
                        .expect("second callback text")
                        .push_str(&event.delta);
                    StreamControl::Continue
                }
            })
            .expect("submit second request");

        assert_eq!(first.wait().expect("wait first"), RequestOutcome::Completed);
        assert_eq!(
            second.wait().expect("wait second"),
            RequestOutcome::Completed
        );
        assert_eq!(*first_text.lock().expect("first result"), expected_first);
        assert_eq!(*second_text.lock().expect("second result"), expected_second);
        assert_eq!(first.native_error().expect("first native error"), None);
        assert_eq!(second.native_error().expect("second native error"), None);
    });
}

#[test]
fn engine_clones_submit_and_wait_from_multiple_rust_threads() {
    with_engine(|engine, _| {
        let start = Arc::new(Barrier::new(3));
        let workers = ["The capital of France is", "The capital of Germany is"]
            .into_iter()
            .map(|prompt| {
                let engine = engine.clone();
                let start = Arc::clone(&start);
                thread::spawn(move || {
                    let output = Arc::new(Mutex::new(String::new()));
                    start.wait();
                    let mut request = engine
                        .submit(prompt, &SamplingParams::greedy().max_tokens(8), {
                            let output = Arc::clone(&output);
                            move |event| {
                                output
                                    .lock()
                                    .expect("cross-thread callback output")
                                    .push_str(&event.delta);
                                StreamControl::Continue
                            }
                        })
                        .expect("submit from Rust worker thread");
                    let outcome = request.wait().expect("wait on Rust worker thread");
                    let native_error = request
                        .native_error()
                        .expect("native error on Rust worker thread");
                    let output = output.lock().expect("cross-thread output").clone();
                    (outcome, native_error, output)
                })
            })
            .collect::<Vec<_>>();

        start.wait();
        for worker in workers {
            let (outcome, native_error, output) = worker.join().expect("Rust request worker");
            assert_eq!(outcome, RequestOutcome::Completed);
            assert_eq!(native_error, None);
            assert!(!output.is_empty());
        }
    });
}

#[test]
fn live_request_moves_to_rust_thread_for_cancel_wait_and_drop() {
    with_engine(|engine, _| {
        let callback_release = Arc::new(Barrier::new(2));
        let (callback_started_sender, callback_started_receiver) = mpsc::channel();
        let (callback_drop_sender, callback_drop_receiver) = mpsc::channel();
        struct CallbackDropProbe(mpsc::Sender<thread::ThreadId>);
        impl Drop for CallbackDropProbe {
            fn drop(&mut self) {
                let _ = self.0.send(thread::current().id());
            }
        }

        let mut callback_started_sender = Some(callback_started_sender);
        let request = engine
            .submit(
                "Write a long numbered list:",
                &SamplingParams::greedy().max_tokens(64),
                {
                    let callback_release = Arc::clone(&callback_release);
                    let drop_probe = CallbackDropProbe(callback_drop_sender);
                    move |_| {
                        let _ = &drop_probe;
                        if let Some(sender) = callback_started_sender.take() {
                            sender.send(()).expect("report live callback");
                            callback_release.wait();
                        }
                        StreamControl::Continue
                    }
                },
            )
            .expect("submit request before moving it");

        let worker = thread::spawn(move || {
            let mut request = request;
            callback_started_receiver
                .recv_timeout(Duration::from_secs(180))
                .expect("callback starts while request is live");
            let cancel_result = request.cancel();
            callback_release.wait();
            cancel_result.expect("cancel moved request");
            let outcome = request.wait().expect("wait for moved request");
            let native_error = request.native_error().expect("moved request error");
            let worker_thread = thread::current().id();
            drop(request);
            (worker_thread, outcome, native_error)
        });

        let (worker_thread, outcome, native_error) = worker.join().expect("moved request worker");
        assert!(matches!(
            outcome,
            RequestOutcome::Cancelled | RequestOutcome::Completed
        ));
        assert_eq!(native_error, None);
        assert_eq!(
            callback_drop_receiver
                .recv_timeout(Duration::from_secs(30))
                .expect("callback state drops with moved request"),
            worker_thread
        );
    });
}

#[test]
fn request_outcomes_and_probes_are_precise() {
    with_engine(|engine, _| {
        let params = SamplingParams::greedy().max_tokens(64);
        let mut stopped = engine
            .submit("Count upward forever:", &params, |_| StreamControl::Stop)
            .expect("submit callback-stop request");
        assert_eq!(
            stopped.wait().expect("wait callback-stop request"),
            RequestOutcome::StoppedByCallback
        );
        assert!(stopped.is_done());
        assert!(stopped.is_done());
        assert_eq!(
            stopped.wait().expect("repeat callback-stop wait"),
            RequestOutcome::StoppedByCallback
        );

        let mut terminal_stopped = engine
            .submit(
                "Say hello",
                &SamplingParams::greedy().max_tokens(1),
                |event| {
                    if event.finished {
                        StreamControl::Stop
                    } else {
                        StreamControl::Continue
                    }
                },
            )
            .expect("submit terminal callback-stop request");
        assert_eq!(
            terminal_stopped
                .wait()
                .expect("wait terminal callback-stop request"),
            RequestOutcome::StoppedByCallback
        );

        let barrier = Arc::new(Barrier::new(2));
        let mut cancelled = engine
            .submit("Write a long numbered list:", &params, {
                let barrier = Arc::clone(&barrier);
                let mut first = true;
                move |_| {
                    if first {
                        first = false;
                        barrier.wait();
                        thread::sleep(Duration::from_millis(20));
                    }
                    StreamControl::Continue
                }
            })
            .expect("submit cancellable request");
        barrier.wait();
        cancelled.cancel().expect("first cancel");
        cancelled.cancel().expect("idempotent cancel");
        assert_eq!(
            cancelled.wait().expect("wait cancelled request"),
            RequestOutcome::Cancelled
        );
        assert!(cancelled.is_done());
        assert!(cancelled.is_done());
        assert_eq!(cancelled.native_error().expect("cancel native error"), None);
    });
}

#[test]
fn callback_panic_is_reported_and_engine_is_reusable() {
    with_engine(|engine, _| {
        let params = SamplingParams::greedy().max_tokens(16);
        let mut request = engine
            .submit("Say hello", &params, |_| {
                panic!("intentional async callback panic")
            })
            .expect("submit panic request");
        assert_eq!(request.wait().unwrap_err(), Error::CallbackPanicked);
        assert_eq!(request.wait().unwrap_err(), Error::CallbackPanicked);

        let completion = engine
            .complete("Say hello", &SamplingParams::greedy().max_tokens(2))
            .expect("engine remains reusable after async panic");
        assert!(!completion.text.is_empty());
    });
}

#[test]
fn request_retains_engine_and_live_drop_is_safe() {
    let Some(path) = model_path() else {
        eprintln!("skipping model test; set VLLM_CPP_TEST_MODEL using `just setup-test-model`");
        return;
    };
    let engine = Engine::builder(path)
        .num_blocks(64)
        .max_model_len(256)
        .max_num_seqs(2)
        .max_num_batched_tokens(256)
        .load()
        .expect("load drop-order engine");
    let params = SamplingParams::greedy().max_tokens(64);
    let final_clone = engine.clone();
    let live = engine
        .submit("Write a long numbered list:", &params, |_| {
            thread::sleep(Duration::from_millis(1));
            StreamControl::Continue
        })
        .expect("submit live request");
    drop(engine);
    drop(live);
    let completion = final_clone
        .complete("Say hello", &SamplingParams::greedy().max_tokens(2))
        .expect("final public engine clone remains usable");
    assert!(!completion.text.is_empty());

    let retained = final_clone
        .submit(
            "Count from one:",
            &SamplingParams::greedy().max_tokens(4),
            |_| StreamControl::Continue,
        )
        .expect("submit engine-retaining request");
    drop(final_clone);
    let mut retained = retained;
    assert_eq!(
        retained.wait().expect("request outlives public engines"),
        RequestOutcome::Completed
    );
}

#[test]
fn callback_thread_self_wait_is_rejected_and_self_drop_is_deferred() {
    with_engine(|engine, _| {
        let params = SamplingParams::greedy().max_tokens(16);
        let slot = Arc::new(Mutex::new(None::<Request>));
        let wait_error = Arc::new(Mutex::new(None));
        let callback_finished = Arc::new(Barrier::new(2));
        let callback_started = Arc::new(Barrier::new(2));
        let callback_thread = Arc::new(Mutex::new(None));
        let (drop_sender, drop_receiver) = mpsc::channel();
        struct CallbackDropProbe(mpsc::Sender<thread::ThreadId>);
        impl Drop for CallbackDropProbe {
            fn drop(&mut self) {
                let _ = self.0.send(thread::current().id());
            }
        }
        let request = engine
            .submit("Count from one:", &params, {
                let slot = Arc::clone(&slot);
                let wait_error = Arc::clone(&wait_error);
                let callback_started = Arc::clone(&callback_started);
                let callback_finished = Arc::clone(&callback_finished);
                let callback_thread = Arc::clone(&callback_thread);
                let drop_probe = CallbackDropProbe(drop_sender);
                move |_| {
                    let _ = &drop_probe;
                    *callback_thread.lock().expect("callback thread") =
                        Some(thread::current().id());
                    callback_started.wait();
                    if let Some(mut request) = slot.lock().expect("request slot").take() {
                        *wait_error.lock().expect("wait result") =
                            Some(request.wait().unwrap_err());
                        drop(request);
                        callback_finished.wait();
                    }
                    StreamControl::Stop
                }
            })
            .expect("submit self-lifecycle request");
        *slot.lock().expect("request slot") = Some(request);
        callback_started.wait();
        callback_finished.wait();
        assert_eq!(
            wait_error.lock().expect("wait result").take(),
            Some(Error::RequestCallbackThread { operation: "wait" })
        );
        let callback_thread = callback_thread
            .lock()
            .expect("callback thread")
            .expect("callback thread recorded");
        let cleanup_thread = drop_receiver
            .recv_timeout(Duration::from_secs(30))
            .expect("deferred callback state drop");
        assert_ne!(cleanup_thread, callback_thread);
    });
}

#[test]
fn concurrent_request_lifecycle_stress() {
    with_engine(|engine, _| {
        let params = SamplingParams::greedy().max_tokens(16);
        for round in 0..16 {
            let mut requests = Vec::new();
            for index in 0..4 {
                let request = engine
                    .submit(
                        &format!("Round {round}, item {index}:"),
                        &params,
                        move |event| {
                            if (round + index) % 4 != 0
                                && (round + index) % 5 == 0
                                && !event.finished
                            {
                                StreamControl::Stop
                            } else {
                                StreamControl::Continue
                            }
                        },
                    )
                    .expect("submit stress request");
                requests.push(request);
            }

            for (index, mut request) in requests.drain(..).enumerate() {
                match (round + index) % 4 {
                    0 => {
                        request.cancel().expect("stress cancel");
                        request.cancel().expect("stress repeated cancel");
                        assert!(matches!(
                            request.wait().expect("wait stress cancel"),
                            RequestOutcome::Cancelled | RequestOutcome::Completed
                        ));
                    }
                    1 => {
                        wait_until_done(&request);
                        assert!(request.is_done());
                        request.wait().expect("wait probed stress request");
                    }
                    2 => {
                        request.wait().expect("wait stress request");
                    }
                    _ => drop(request),
                }
            }
        }

        let completion = engine
            .complete("Say hello", &SamplingParams::greedy().max_tokens(2))
            .expect("engine remains reusable after lifecycle stress");
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
