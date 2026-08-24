use std::collections::HashMap;
use std::ffi::CString;
use std::fmt;
use std::mem::{align_of, size_of};
use std::os::raw::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::thread::{self, ThreadId};

use vllm_cpp_sys as ffi;

use crate::error::{invalid_configuration, Error};

const NATIVE_DEFAULT_MAX_TOKENS: u32 = 16;

/// Native scheduler admission order.
///
/// Raw and serde chat request JSON can carry a `priority` field that the native
/// OpenAI-compatible path parses and submits. Direct completion, completion
/// streaming, and [`crate::Request`] submissions currently default to priority zero
/// and tie by arrival; caller-selected priorities for those direct APIs require a
/// future C ABI/API change.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SchedulerPolicy {
    /// Process requests in arrival order.
    #[default]
    Fcfs,
    /// Order requests by priority and then arrival time.
    ///
    /// This variant selects the native priority queue; it does not itself assign a
    /// priority to a request.
    Priority,

    /// Prefer requests sharing the longest cached prefix.
    LongestPrefixMatch,
}

impl SchedulerPolicy {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Fcfs => "fcfs",
            Self::Priority => "priority",
            Self::LongestPrefixMatch => "lpm",
        }
    }
}

/// A native tri-state setting whose default is resolved by vllm.cpp.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Toggle {
    /// Let vllm.cpp resolve the model or environment default.
    #[default]
    Default,
    /// Force the feature on.
    On,
    /// Force the feature off.
    Off,
}

impl Toggle {
    pub(crate) const fn as_native(self) -> i32 {
        match self {
            Self::Default => 0,
            Self::On => 1,
            Self::Off => 2,
        }
    }
}

type DynLogitsProcessor = dyn Fn(&[i32], &mut [f32]) + Send + Sync + 'static;

#[derive(Clone)]
struct LogitsProcessor {
    callback: Arc<DynLogitsProcessor>,
}

impl fmt::Debug for LogitsProcessor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("LogitsProcessor { .. }")
    }
}

impl PartialEq for LogitsProcessor {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.callback, &other.callback)
    }
}

/// One engine-side structured decoding constraint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StructuredOutput {
    JsonSchema(String),
    Regex(String),
    Choice(Vec<String>),
    Grammar(String),
    JsonObject,
}

/// Owned sampling configuration for one generation request.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingParams {
    temperature: f32,
    top_p: f32,
    top_k: i32,
    min_p: f32,
    max_tokens: Option<u32>,
    seed: Option<u64>,
    presence_penalty: f32,
    frequency_penalty: f32,
    repetition_penalty: f32,
    min_tokens: u32,
    ignore_eos: bool,
    stop: Vec<String>,
    structured_output: Option<StructuredOutput>,
    logits_processor: Option<LogitsProcessor>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            max_tokens: Some(NATIVE_DEFAULT_MAX_TOKENS),
            seed: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            min_tokens: 0,
            ignore_eos: false,
            stop: Vec::new(),
            structured_output: None,
            logits_processor: None,
        }
    }
}

impl SamplingParams {
    /// Returns deterministic argmax sampling with native defaults otherwise.
    #[must_use]
    pub fn greedy() -> Self {
        Self::default().temperature(0.0)
    }

    #[must_use]
    pub fn temperature(mut self, value: f32) -> Self {
        self.temperature = value;
        self
    }

    #[must_use]
    pub fn top_p(mut self, value: f32) -> Self {
        self.top_p = value;
        self
    }

    #[must_use]
    pub fn top_k(mut self, value: i32) -> Self {
        self.top_k = value;
        self
    }

    #[must_use]
    pub fn min_p(mut self, value: f32) -> Self {
        self.min_p = value;
        self
    }

    /// Sets a finite generation limit.
    ///
    /// Zero is invalid; use [`unbounded`](Self::unbounded) to request native
    /// unbounded generation explicitly.
    #[must_use]
    pub fn max_tokens(mut self, value: u32) -> Self {
        self.max_tokens = Some(value);
        self
    }

    /// Removes the generation limit.
    #[must_use]
    pub fn unbounded(mut self) -> Self {
        self.max_tokens = None;
        self
    }

    #[must_use]
    pub fn seed(mut self, value: u64) -> Self {
        self.seed = Some(value);
        self
    }

    #[must_use]
    pub fn clear_seed(mut self) -> Self {
        self.seed = None;
        self
    }

    #[must_use]
    pub fn presence_penalty(mut self, value: f32) -> Self {
        self.presence_penalty = value;
        self
    }

    #[must_use]
    pub fn frequency_penalty(mut self, value: f32) -> Self {
        self.frequency_penalty = value;
        self
    }

    #[must_use]
    pub fn repetition_penalty(mut self, value: f32) -> Self {
        self.repetition_penalty = value;
        self
    }

    #[must_use]
    pub fn min_tokens(mut self, value: u32) -> Self {
        self.min_tokens = value;
        self
    }

    #[must_use]
    pub fn ignore_eos(mut self, value: bool) -> Self {
        self.ignore_eos = value;
        self
    }

    #[must_use]
    pub fn stop(mut self, value: impl Into<String>) -> Self {
        self.stop.push(value.into());
        self
    }

    #[must_use]
    pub fn stop_all<I, S>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.stop.extend(values.into_iter().map(Into::into));
        self
    }

    #[must_use]
    pub fn structured_output(mut self, value: StructuredOutput) -> Self {
        self.structured_output = Some(value);
        self
    }

    /// Installs a host-side processor that can inspect generated token IDs and
    /// mutate one request's logits before each sampling step.
    ///
    /// The processor may run concurrently for different requests, so it must be
    /// `Send + Sync`. Cloned parameters share the processor. A panic is contained
    /// before the C boundary and reported as [`Error::LogitsProcessorPanicked`]
    /// after the bounded generation call or from [`crate::Request::wait`]. The
    /// callback state remains registered through the call or request lifetime;
    /// stale native invocations after cleanup become no-ops.
    #[must_use]
    pub fn logits_processor<F>(mut self, processor: F) -> Self
    where
        F: Fn(&[i32], &mut [f32]) + Send + Sync + 'static,
    {
        self.logits_processor = Some(LogitsProcessor {
            callback: Arc::new(processor),
        });
        self
    }

    /// Removes a previously configured custom logits processor.
    #[must_use]
    pub fn clear_logits_processor(mut self) -> Self {
        self.logits_processor = None;
        self
    }

    pub(crate) fn marshal(&self) -> Result<MarshaledSamplingParams, Error> {
        MarshaledSamplingParams::new(self)
    }
}

pub(crate) struct MarshaledSamplingParams {
    raw: ffi::vllm_sampling_params,
    _stop: Vec<CString>,
    _stop_pointers: Vec<*const c_char>,
    _structured_string: Option<CString>,
    _choices: Vec<CString>,
    _choice_pointers: Vec<*const c_char>,
    logits_processor: Option<LogitsProcessorRegistration>,
}

impl MarshaledSamplingParams {
    fn new(params: &SamplingParams) -> Result<Self, Error> {
        // ABI equality is checked before this struct-returning call.
        let mut raw = unsafe { ffi::vllm_sampling_params_default() };
        raw.temperature = params.temperature;
        raw.top_p = params.top_p;
        raw.top_k = params.top_k;
        raw.min_p = params.min_p;
        raw.max_tokens = optional_u32_to_i32(params.max_tokens, "max_tokens")?;
        raw.seed = params.seed.unwrap_or(0);
        raw.has_seed = i32::from(params.seed.is_some());
        raw.presence_penalty = params.presence_penalty;
        raw.frequency_penalty = params.frequency_penalty;
        raw.repetition_penalty = params.repetition_penalty;
        raw.min_tokens = u32_to_i32(params.min_tokens, "min_tokens")?;
        raw.ignore_eos = i32::from(params.ignore_eos);

        let stop = strings_to_cstrings(&params.stop, "stop string")?;
        let stop_pointers = stop.iter().map(|value| value.as_ptr()).collect::<Vec<_>>();
        raw.stop = pointer_or_null(&stop_pointers);
        raw.n_stop = length_to_i32(stop_pointers.len(), "stop strings")?;

        let mut structured_string = None;
        let mut choices = Vec::new();
        let mut choice_pointers = Vec::new();
        if let Some(structured) = &params.structured_output {
            match structured {
                StructuredOutput::JsonSchema(value) => {
                    structured_string = Some(to_cstring(value, "JSON schema")?);
                    raw.structured_json = structured_string.as_ref().unwrap().as_ptr();
                }
                StructuredOutput::Regex(value) => {
                    structured_string = Some(to_cstring(value, "structured regex")?);
                    raw.structured_regex = structured_string.as_ref().unwrap().as_ptr();
                }
                StructuredOutput::Choice(values) => {
                    if values.is_empty() {
                        return Err(invalid_configuration(
                            "structured choices must contain at least one value",
                        ));
                    }
                    choices = strings_to_cstrings(values, "structured choice")?;
                    choice_pointers = choices
                        .iter()
                        .map(|value| value.as_ptr())
                        .collect::<Vec<_>>();
                    raw.structured_choice = pointer_or_null(&choice_pointers);
                    raw.n_structured_choice =
                        length_to_i32(choice_pointers.len(), "structured choices")?;
                }
                StructuredOutput::Grammar(value) => {
                    structured_string = Some(to_cstring(value, "structured grammar")?);
                    raw.structured_grammar = structured_string.as_ref().unwrap().as_ptr();
                }
                StructuredOutput::JsonObject => raw.structured_json_object = 1,
            }
        }

        let mut logits_processor = None;
        if let Some(processor) = &params.logits_processor {
            if params.max_tokens.is_none() || params.max_tokens == Some(0) {
                return Err(invalid_configuration(
                    "custom logits processors require bounded max_tokens because the native callback cannot abort generation",
                ));
            }
            let registration = LogitsProcessorRegistration::new(Arc::clone(&processor.callback));
            raw.logits_processor = Some(logits_processor_trampoline);
            raw.logits_processor_user_data = registration.user_data();
            logits_processor = Some(registration);
        }

        Ok(Self {
            raw,
            _stop: stop,
            _stop_pointers: stop_pointers,
            _structured_string: structured_string,
            _choices: choices,
            _choice_pointers: choice_pointers,
            logits_processor,
        })
    }

    pub(crate) const fn raw(&self) -> &ffi::vllm_sampling_params {
        &self.raw
    }

    pub(crate) fn logits_processor_error(&self) -> Option<Error> {
        self.logits_processor
            .as_ref()
            .and_then(LogitsProcessorRegistration::error)
    }

    pub(crate) fn take_logits_processor(&mut self) -> Option<LogitsProcessorRegistration> {
        self.logits_processor.take()
    }
}

const PROCESSOR_OK: u8 = 0;
const PROCESSOR_PANICKED: u8 = 1;
const PROCESSOR_INVALID_INPUT: u8 = 2;

static NEXT_PROCESSOR_ID: AtomicUsize = AtomicUsize::new(1);
static LOGITS_PROCESSORS: OnceLock<Mutex<HashMap<usize, Weak<LogitsProcessorState>>>> =
    OnceLock::new();

pub(crate) struct LogitsProcessorRegistration {
    id: usize,
    state: Arc<LogitsProcessorState>,
}

impl LogitsProcessorRegistration {
    fn new(callback: Arc<DynLogitsProcessor>) -> Self {
        let id = NEXT_PROCESSOR_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .unwrap_or_else(|_| std::process::abort());
        let state = Arc::new(LogitsProcessorState::new(callback));
        lock_unpoisoned(logits_processor_registry()).insert(id, Arc::downgrade(&state));
        Self { id, state }
    }

    fn user_data(&self) -> *mut c_void {
        ptr::without_provenance_mut(self.id)
    }

    pub(crate) fn error(&self) -> Option<Error> {
        self.state.error()
    }

    pub(crate) fn is_active_on_current_thread(&self) -> bool {
        self.state.is_active_on_current_thread()
    }
}

impl Drop for LogitsProcessorRegistration {
    fn drop(&mut self) {
        lock_unpoisoned(logits_processor_registry()).remove(&self.id);
    }
}

fn logits_processor_registry() -> &'static Mutex<HashMap<usize, Weak<LogitsProcessorState>>> {
    LOGITS_PROCESSORS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn registered_logits_processor(id: usize) -> Option<Arc<LogitsProcessorState>> {
    lock_unpoisoned(logits_processor_registry())
        .get(&id)
        .and_then(Weak::upgrade)
}

struct LogitsProcessorState {
    callback: Arc<DynLogitsProcessor>,
    failure: AtomicU8,
    active_threads: Mutex<Vec<ThreadId>>,
}

impl LogitsProcessorState {
    fn new(callback: Arc<DynLogitsProcessor>) -> Self {
        Self {
            callback,
            failure: AtomicU8::new(PROCESSOR_OK),
            active_threads: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn error(&self) -> Option<Error> {
        match self.failure.load(Ordering::Acquire) {
            PROCESSOR_OK => None,
            PROCESSOR_PANICKED => Some(Error::LogitsProcessorPanicked),
            PROCESSOR_INVALID_INPUT => Some(Error::Runtime {
                message: "native logits processor callback received invalid pointers or lengths"
                    .to_owned(),
            }),
            _ => Some(Error::Runtime {
                message: "native logits processor callback entered an unknown failure state"
                    .to_owned(),
            }),
        }
    }

    pub(crate) fn is_active_on_current_thread(&self) -> bool {
        let current = thread::current().id();
        lock_unpoisoned(&self.active_threads).contains(&current)
    }

    fn record_failure(&self, failure: u8) {
        let _ = self.failure.compare_exchange(
            PROCESSOR_OK,
            failure,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

struct ActiveProcessorGuard<'state> {
    state: &'state LogitsProcessorState,
    thread_id: ThreadId,
}

impl<'state> ActiveProcessorGuard<'state> {
    fn enter(state: &'state LogitsProcessorState) -> Self {
        let thread_id = thread::current().id();
        lock_unpoisoned(&state.active_threads).push(thread_id);
        Self { state, thread_id }
    }
}

impl Drop for ActiveProcessorGuard<'_> {
    fn drop(&mut self) {
        let mut active = lock_unpoisoned(&self.state.active_threads);
        if let Some(index) = active.iter().rposition(|id| *id == self.thread_id) {
            active.swap_remove(index);
        }
    }
}

unsafe extern "C" fn logits_processor_trampoline(
    token_ids: *const i32,
    n_token_ids: i32,
    logits: *mut f32,
    vocab_size: i32,
    user_data: *mut c_void,
) {
    if user_data.is_null() {
        return;
    }
    let Some(state) = registered_logits_processor(user_data.addr()) else {
        return;
    };
    if state.failure.load(Ordering::Acquire) != PROCESSOR_OK {
        return;
    }
    if n_token_ids < 0
        || vocab_size <= 0
        || (n_token_ids > 0 && token_ids.is_null())
        || logits.is_null()
        || (n_token_ids > 0 && !valid_slice_layout(token_ids, n_token_ids as usize))
        || !valid_slice_layout(logits, vocab_size as usize)
    {
        state.record_failure(PROCESSOR_INVALID_INPUT);
        return;
    }

    let _active = ActiveProcessorGuard::enter(&state);
    let tokens = if n_token_ids == 0 {
        &[]
    } else {
        // SAFETY: the native callback contract lends this aligned token slice for
        // this invocation, and the validated length fits Rust slice bounds.
        unsafe { slice::from_raw_parts(token_ids, n_token_ids as usize) }
    };
    // SAFETY: the native callback contract lends this aligned, uniquely mutable
    // logits row for this invocation, and the validated length fits slice bounds.
    let logits = unsafe { slice::from_raw_parts_mut(logits, vocab_size as usize) };
    if let Err(payload) = catch_unwind(AssertUnwindSafe(|| (state.callback)(tokens, logits))) {
        state.record_failure(PROCESSOR_PANICKED);
        discard_panic_payload(payload);
    }
}

fn valid_slice_layout<T>(pointer: *const T, length: usize) -> bool {
    !pointer.is_null()
        && (pointer as usize) % align_of::<T>() == 0
        && length <= (isize::MAX as usize) / size_of::<T>()
}

fn discard_panic_payload(payload: Box<dyn std::any::Any + Send>) {
    if let Err(payload) = catch_unwind(AssertUnwindSafe(|| drop(payload))) {
        std::mem::forget(payload);
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub(crate) fn to_cstring(value: &str, field: &'static str) -> Result<CString, Error> {
    CString::new(value).map_err(|_| Error::InteriorNul { field })
}

fn strings_to_cstrings(values: &[String], field: &'static str) -> Result<Vec<CString>, Error> {
    values
        .iter()
        .map(|value| to_cstring(value, field))
        .collect()
}

fn pointer_or_null(values: &[*const c_char]) -> *const *const c_char {
    if values.is_empty() {
        ptr::null()
    } else {
        values.as_ptr()
    }
}

fn optional_u32_to_i32(value: Option<u32>, field: &'static str) -> Result<i32, Error> {
    match value {
        Some(0) => Err(invalid_configuration(format!(
            "{field} must be greater than zero; use unbounded() for no limit"
        ))),
        Some(value) => u32_to_i32(value, field),
        None => Ok(0),
    }
}

fn u32_to_i32(value: u32, field: &'static str) -> Result<i32, Error> {
    i32::try_from(value)
        .map_err(|_| invalid_configuration(format!("{field} exceeds native i32 range")))
}

fn length_to_i32(value: usize, field: &'static str) -> Result<i32, Error> {
    i32::try_from(value).map_err(|_| invalid_configuration(format!("too many {field}")))
}

#[cfg(test)]
mod tests {
    use super::{logits_processor_trampoline, SamplingParams};
    use crate::Error;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn marshals_and_invokes_custom_logits_processor() {
        let params = SamplingParams::default()
            .max_tokens(2)
            .logits_processor(|tokens, logits| {
                assert_eq!(tokens, &[3, 5]);
                logits[1] = 9.0;
            });
        let marshaled = params.marshal().expect("marshal processor");
        let callback = marshaled
            .raw()
            .logits_processor
            .expect("processor callback");
        let mut logits = [1.0, 2.0, 3.0];
        let tokens = [3, 5];
        unsafe {
            callback(
                tokens.as_ptr(),
                tokens.len() as i32,
                logits.as_mut_ptr(),
                logits.len() as i32,
                marshaled.raw().logits_processor_user_data,
            );
        }
        assert_eq!(logits, [1.0, 9.0, 3.0]);
        assert_eq!(marshaled.logits_processor_error(), None);
    }

    #[test]
    fn stale_processor_user_data_is_a_noop() {
        let calls = Arc::new(AtomicUsize::new(0));
        let user_data = {
            let calls = Arc::clone(&calls);
            let params = SamplingParams::default()
                .max_tokens(1)
                .logits_processor(move |_, _| {
                    calls.fetch_add(1, Ordering::Relaxed);
                });
            let marshaled = params.marshal().expect("marshal processor");
            marshaled.raw().logits_processor_user_data
        };
        let mut logits = [1.0];
        unsafe {
            logits_processor_trampoline(
                std::ptr::null(),
                0,
                logits.as_mut_ptr(),
                logits.len() as i32,
                user_data,
            );
        }
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn contains_processor_panic_and_skips_later_calls() {
        let params = SamplingParams::default()
            .max_tokens(2)
            .logits_processor(|_, _| panic!("processor panic"));
        let marshaled = params.marshal().expect("marshal processor");
        let mut logits = [1.0, 2.0];
        unsafe {
            logits_processor_trampoline(
                std::ptr::null(),
                0,
                logits.as_mut_ptr(),
                logits.len() as i32,
                marshaled.raw().logits_processor_user_data,
            );
        }
        assert_eq!(
            marshaled.logits_processor_error(),
            Some(Error::LogitsProcessorPanicked)
        );
        unsafe {
            logits_processor_trampoline(
                std::ptr::null(),
                0,
                logits.as_mut_ptr(),
                logits.len() as i32,
                marshaled.raw().logits_processor_user_data,
            );
        }
    }

    #[test]
    fn rejects_zero_or_unbounded_processor_and_invalid_native_shape() {
        for params in [
            SamplingParams::default().max_tokens(0),
            SamplingParams::default()
                .unbounded()
                .logits_processor(|_, _| {}),
        ] {
            let error = params.marshal().err().expect("invalid bounds rejection");
            assert!(matches!(error, Error::InvalidConfiguration { .. }));
        }

        let params = SamplingParams::default().logits_processor(|_, _| {});
        let marshaled = params.marshal().expect("marshal processor");
        unsafe {
            logits_processor_trampoline(
                std::ptr::null(),
                -1,
                std::ptr::null_mut(),
                0,
                marshaled.raw().logits_processor_user_data,
            );
        }
        assert!(matches!(
            marshaled.logits_processor_error(),
            Some(Error::Runtime { .. })
        ));
    }
}
