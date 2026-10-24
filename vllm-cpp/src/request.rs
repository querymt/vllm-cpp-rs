use std::any::Any;
use std::cell::Cell;
use std::ffi::CStr;
use std::marker::PhantomData;
use std::os::raw::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::{self, NonNull};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::thread::{self, ThreadId};

use vllm_cpp_sys as ffi;

use crate::callback::{StreamControl, StreamEvent};
use crate::engine::{Engine, EngineInner};
use crate::error::{status_result, Error};
use crate::params::{
    discard_panic_payload, lock_unpoisoned, to_cstring, LogitsProcessorRegistration, SamplingParams,
};

/// How a successfully waited non-blocking request ended.
///
/// This is a Rust-side classification because the native ABI does not expose its
/// cancellation flag. Callback panic/error takes precedence, followed by an
/// explicit callback stop, an observed terminal callback, and cancellation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RequestOutcome {
    /// Native generation delivered its terminal callback.
    Completed,
    /// The Rust callback returned [`StreamControl::Stop`].
    ///
    /// ABI 17 treats this as an explicit stop even for the terminal event.
    StoppedByCallback,
    /// Rust requested cancellation before completion was observable.
    Cancelled,
}

/// An owned non-blocking streaming request.
///
/// A request keeps its parent [`Engine`] alive. Lifecycle methods require mutable
/// access, and the request is intentionally `Send` but not `Sync`.
pub struct Request {
    raw: Option<NonNull<ffi::vllm_request>>,
    callback: Option<Box<AsyncCallbackState>>,
    logits_processor: Option<LogitsProcessorRegistration>,
    engine: Option<Arc<EngineInner>>,
    cancellation_requested: bool,
    _not_sync: PhantomData<Cell<()>>,
}

impl std::fmt::Debug for Request {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Request")
            .field("raw", &self.raw)
            .field("cancellation_requested", &self.cancellation_requested)
            .finish_non_exhaustive()
    }
}

impl Engine {
    /// Submits a non-blocking streaming completion to the shared engine.
    ///
    /// The callback runs on a native delivery thread and receives an owned UTF-8
    /// copy of each delta. A callback panic is contained and later reported by
    /// [`Request::wait`] as [`Error::CallbackPanicked`].
    pub fn submit<F>(
        &self,
        prompt: &str,
        params: &SamplingParams,
        callback: F,
    ) -> Result<Request, Error>
    where
        F: FnMut(StreamEvent) -> StreamControl + Send + 'static,
    {
        cleanup_sender()?;
        let prompt = to_cstring(prompt, "prompt")?;
        let mut params = params.marshal(&self.inner.compatibility)?;
        let mut callback = Box::new(AsyncCallbackState::new(callback));
        let mut output = ptr::null_mut();
        // SAFETY: the engine is retained by the returned Request, native code
        // copies prompt/parameters before returning, callback has a stable boxed
        // address, and callback state remains live until request_free joins.
        let status = unsafe {
            ffi::vllm_request_submit(
                self.inner.raw.as_ptr(),
                prompt.as_ptr(),
                params.raw(),
                Some(async_callback_trampoline),
                ptr::from_mut(&mut *callback).cast(),
                &mut output,
            )
        };
        if status != ffi::vllm_status_VLLM_OK {
            status_result(status)?;
            unreachable!("non-OK native status unexpectedly succeeded");
        }
        let raw = match NonNull::new(output) {
            Some(raw) => raw,
            None => {
                return Err(Error::Runtime {
                    message: "vllm_request_submit succeeded without a request handle".to_owned(),
                });
            }
        };
        Ok(Request {
            raw: Some(raw),
            callback: Some(callback),
            logits_processor: params.take_logits_processor(),
            engine: Some(Arc::clone(&self.inner)),
            cancellation_requested: false,
            _not_sync: PhantomData,
        })
    }
}

impl Request {
    /// Returns whether native callback delivery has finished.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.native_done()
    }

    /// Requests cancellation.
    ///
    /// Cancellation is idempotent. The ABI does not report whether this call
    /// changed native state, so [`wait`](Self::wait) returns
    /// [`RequestOutcome::Cancelled`] when cancellation succeeded after a false
    /// completion probe and no terminal callback was subsequently observed. A
    /// terminal callback wins that race unless it stops or panics.
    pub fn cancel(&mut self) -> Result<(), Error> {
        let was_done = self.is_done();
        // SAFETY: mutable access serializes safe lifecycle calls and raw is live.
        let status = unsafe { ffi::vllm_request_cancel(self.raw().as_ptr()) };
        status_result(status)?;
        self.cancellation_requested |= !was_done;
        Ok(())
    }

    /// Waits for callback delivery to finish and returns its terminal outcome.
    ///
    /// Calling this from this request's own callback returns
    /// [`Error::RequestCallbackThread`] without entering native code.
    pub fn wait(&mut self) -> Result<RequestOutcome, Error> {
        if self.is_native_callback_thread() {
            return Err(Error::RequestCallbackThread { operation: "wait" });
        }
        // SAFETY: mutable access serializes safe lifecycle calls, raw is live,
        // and the delivery-thread case was rejected before the FFI call.
        let status = unsafe { ffi::vllm_request_wait(self.raw().as_ptr()) };
        let native_result = status_result(status);
        if let Some(error) = self.logits_processor_error() {
            return Err(error);
        }
        let callback_result = self.callback().result(self.cancellation_requested);
        match callback_result {
            Err(error) => Err(error),
            Ok(Some(outcome)) => native_result.map(|()| outcome),
            Ok(None) => {
                native_result?;
                Err(Error::Runtime {
                    message: "request completed without a terminal callback or locally observable stop/cancellation"
                        .to_owned(),
                })
            }
        }
    }

    /// Copies the native request diagnostic after completion.
    ///
    /// Returns `None` while the request is running or when it completed without
    /// a native diagnostic. Callback panics are reported by [`wait`](Self::wait)
    /// rather than through this native string.
    pub fn native_error(&self) -> Result<Option<String>, Error> {
        if !self.is_done() {
            return Ok(None);
        }
        // SAFETY: done has acquired native publication of the request-owned error
        // string, and raw remains live for this copy.
        let pointer = unsafe { ffi::vllm_request_error(self.raw().as_ptr()) };
        if pointer.is_null() {
            return Err(Error::Runtime {
                message: "vllm_request_error returned a null pointer".to_owned(),
            });
        }
        // SAFETY: the C contract promises a NUL-terminated string valid until
        // request_free; this method copies it before returning.
        let error = unsafe { CStr::from_ptr(pointer) }
            .to_str()
            .map_err(|_| Error::InvalidUtf8 {
                field: "request error",
            })?
            .to_owned();
        Ok((!error.is_empty()).then_some(error))
    }

    fn raw(&self) -> NonNull<ffi::vllm_request> {
        self.raw.expect("live Request always has a native handle")
    }

    fn native_done(&self) -> bool {
        // SAFETY: raw remains a live request handle until Drop, and the native
        // completion probe is atomic and accepts concurrent callback delivery.
        unsafe { ffi::vllm_request_done(self.raw().as_ptr()) }
    }

    fn callback(&self) -> &AsyncCallbackState {
        self.callback
            .as_deref()
            .expect("live Request always has callback state")
    }

    fn logits_processor_error(&self) -> Option<Error> {
        self.logits_processor
            .as_ref()
            .and_then(LogitsProcessorRegistration::error)
    }

    fn is_native_callback_thread(&self) -> bool {
        self.callback().is_delivery_thread()
            || self
                .logits_processor
                .as_ref()
                .is_some_and(LogitsProcessorRegistration::is_active_on_current_thread)
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        let parts = (
            self.raw.take(),
            self.callback.take(),
            self.logits_processor.take(),
            self.engine.take(),
        );
        match parts {
            (Some(raw), Some(callback), logits_processor, Some(engine)) => {
                CleanupJob::new(raw, callback, logits_processor, engine).run();
            }
            parts => {
                // A partial Request would make either freeing or dropping its
                // Rust owners unsafe. This private invariant cannot fail without
                // an implementation bug, so preserve everything before aborting.
                std::mem::forget(parts);
                std::process::abort();
            }
        }
    }
}

// Moving exclusive request ownership between threads is valid under the native
// request contract. Callback state is Send, EngineInner is Send + Sync, and
// wait/free explicitly reject or defer the one prohibited delivery-thread case.
unsafe impl Send for Request {}

struct CallbackOutcome {
    stopped: bool,
    saw_finished: bool,
    error: Option<Error>,
    panic: Option<Box<dyn Any + Send>>,
    delivery_thread: Option<ThreadId>,
}

struct AsyncCallbackState {
    callback: Mutex<Box<dyn FnMut(StreamEvent) -> StreamControl + Send + 'static>>,
    outcome: Mutex<CallbackOutcome>,
}

impl AsyncCallbackState {
    fn new<F>(callback: F) -> Self
    where
        F: FnMut(StreamEvent) -> StreamControl + Send + 'static,
    {
        Self {
            callback: Mutex::new(Box::new(callback)),
            outcome: Mutex::new(CallbackOutcome {
                stopped: false,
                saw_finished: false,
                error: None,
                panic: None,
                delivery_thread: None,
            }),
        }
    }

    fn record_delivery_thread(&self) {
        // ABI 17 invokes user_data only from this request's single library-owned
        // delivery thread. Retain its ID through cleanup instead of marking only
        // an active trampoline, so every possible Rust re-entry from that thread
        // remains ineligible for synchronous wait/free.
        lock_unpoisoned(&self.outcome).delivery_thread = Some(thread::current().id());
    }

    fn is_delivery_thread(&self) -> bool {
        lock_unpoisoned(&self.outcome)
            .delivery_thread
            .as_ref()
            .is_some_and(|id| *id == thread::current().id())
    }

    fn record_error(&self, error: Error) {
        let mut outcome = lock_unpoisoned(&self.outcome);
        outcome.error = Some(error);
        outcome.stopped = true;
    }

    fn record_result(
        &self,
        result: Result<StreamControl, Box<dyn Any + Send>>,
        finished: bool,
    ) -> bool {
        let mut outcome = lock_unpoisoned(&self.outcome);
        outcome.saw_finished |= finished;
        match result {
            Ok(StreamControl::Continue) => true,
            Ok(StreamControl::Stop) => {
                outcome.stopped = true;
                false
            }
            Err(payload) => {
                outcome.panic = Some(payload);
                outcome.stopped = true;
                false
            }
        }
    }

    fn result(&self, cancellation_requested: bool) -> Result<Option<RequestOutcome>, Error> {
        let outcome = lock_unpoisoned(&self.outcome);
        if outcome.panic.is_some() {
            return Err(Error::CallbackPanicked);
        }
        if let Some(error) = &outcome.error {
            return Err(error.clone());
        }
        if outcome.stopped {
            return Ok(Some(RequestOutcome::StoppedByCallback));
        }
        if outcome.saw_finished {
            return Ok(Some(RequestOutcome::Completed));
        }
        if cancellation_requested {
            return Ok(Some(RequestOutcome::Cancelled));
        }
        Ok(None)
    }
}

unsafe extern "C" fn async_callback_trampoline(
    delta_text: *const c_char,
    finished: bool,
    user_data: *mut c_void,
) -> bool {
    // SAFETY: submit passes a stable boxed AsyncCallbackState, and request_free
    // joins this delivery before the box can be destroyed.
    let state = unsafe { &*user_data.cast::<AsyncCallbackState>() };
    state.record_delivery_thread();
    if delta_text.is_null() {
        state.record_error(Error::InvalidUtf8 {
            field: "stream delta",
        });
        return false;
    }
    // SAFETY: native code lends a NUL-terminated string for this invocation.
    let delta = match unsafe { CStr::from_ptr(delta_text) }.to_str() {
        Ok(delta) => delta.to_owned(),
        Err(_) => {
            state.record_error(Error::InvalidUtf8 {
                field: "stream delta",
            });
            return false;
        }
    };
    let event = StreamEvent { delta, finished };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut callback = lock_unpoisoned(&state.callback);
        callback(event)
    }));
    state.record_result(result, finished)
}

struct CleanupJob {
    raw: NonNull<ffi::vllm_request>,
    callback: Box<AsyncCallbackState>,
    logits_processor: Option<LogitsProcessorRegistration>,
    engine: Arc<EngineInner>,
}

impl CleanupJob {
    fn new(
        raw: NonNull<ffi::vllm_request>,
        callback: Box<AsyncCallbackState>,
        logits_processor: Option<LogitsProcessorRegistration>,
        engine: Arc<EngineInner>,
    ) -> Self {
        Self {
            raw,
            callback,
            logits_processor,
            engine,
        }
    }

    fn run(self) {
        if self.needs_deferral() {
            self.defer();
        } else {
            self.free();
        }
    }

    // request_free joins output delivery only; avoid both its callback thread and
    // the currently active logits thread before transferring ownership to reaper.
    fn needs_deferral(&self) -> bool {
        self.callback.is_delivery_thread()
            || self
                .logits_processor
                .as_ref()
                .is_some_and(LogitsProcessorRegistration::is_active_on_current_thread)
    }

    fn defer(self) {
        let sender = match CLEANUP_REAPER.get() {
            Some(Ok(sender)) => sender,
            // Submission starts the process-lifetime reaper before native code
            // can create a request. Without it, self-thread cleanup is impossible.
            _ => std::process::abort(),
        };
        if let Err(error) = sender.send(self) {
            // Dropping this error would drop the callback, logits processor, and
            // engine while the raw handle is live, before native free joins delivery.
            // Forget it and abort rather than run those destructors in this terminal state.
            std::mem::forget(error);
            std::process::abort();
        }
    }

    fn free(self) {
        let Self {
            raw,
            callback,
            logits_processor,
            engine,
        } = self;
        // Keep all Rust owners alive until request_free has returned. If it unwinds,
        // the native handle's state is unknown, so retain them without retrying cleanup.
        let mut owners = std::mem::ManuallyDrop::new((callback, logits_processor, engine));
        // SAFETY: the unique job owner calls free once, off both protected callback
        // contexts, and retains the callback, processor registration, and engine.
        let free_result = catch_unwind(AssertUnwindSafe(|| unsafe {
            ffi::vllm_request_free(raw.as_ptr())
        }));
        if let Err(payload) = free_result {
            discard_panic_payload(payload);
            return;
        }

        // SAFETY: free returned after joining output delivery; the owners were kept
        // in ManuallyDrop until that point.
        let (callback, logits_processor, engine) =
            unsafe { std::mem::ManuallyDrop::take(&mut owners) };
        drop_cleanup_owners(callback, logits_processor, engine);
    }
}

fn drop_cleanup_owners<C, L, E>(callback: C, logits_processor: L, engine: E) {
    // User captures and stored panic payloads can have arbitrary destructors.
    // Contain each panic so later owners are still released, with engine last.
    if let Err(payload) = catch_unwind(AssertUnwindSafe(|| drop(callback))) {
        discard_panic_payload(payload);
    }
    if let Err(payload) = catch_unwind(AssertUnwindSafe(|| drop(logits_processor))) {
        discard_panic_payload(payload);
    }
    drop(engine);
}

// The job transfers unique native-handle ownership to the reaper. Its callback
// is Send and its retained engine is Send + Sync; no aliases perform lifecycle
// operations while the job owns the handle.
unsafe impl Send for CleanupJob {}

static CLEANUP_REAPER: OnceLock<Result<mpsc::Sender<CleanupJob>, String>> = OnceLock::new();

fn cleanup_sender() -> Result<&'static mpsc::Sender<CleanupJob>, Error> {
    match CLEANUP_REAPER.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<CleanupJob>();
        thread::Builder::new()
            .name("vllm-request-reaper".to_owned())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    job.run();
                }
            })
            .map(|_| sender)
            .map_err(|error| error.to_string())
    }) {
        Ok(sender) => Ok(sender),
        Err(message) => Err(Error::Runtime {
            message: format!("failed to start request cleanup reaper: {message}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::drop_cleanup_owners;

    struct DropProbe {
        name: &'static str,
        drops: Arc<Mutex<Vec<&'static str>>>,
        panic: bool,
    }

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.drops.lock().unwrap().push(self.name);
            if self.panic {
                panic!("probe destructor panic");
            }
        }
    }

    #[test]
    fn owner_drop_panics_do_not_skip_later_owners() {
        for (callback_panics, logits_panics) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let drops = Arc::new(Mutex::new(Vec::new()));
            let probe = |name, panic| DropProbe {
                name,
                drops: Arc::clone(&drops),
                panic,
            };

            drop_cleanup_owners(
                probe("callback", callback_panics),
                Some(probe("logits", logits_panics)),
                probe("engine", false),
            );

            assert_eq!(*drops.lock().unwrap(), ["callback", "logits", "engine"]);
        }
    }
}
