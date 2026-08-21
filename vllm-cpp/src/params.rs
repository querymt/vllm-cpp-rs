use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;

use vllm_cpp_sys as ffi;

use crate::error::{invalid_configuration, Error};

const NATIVE_DEFAULT_MAX_TOKENS: u32 = 16;

/// Native scheduler admission order.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SchedulerPolicy {
    /// Process requests in arrival order.
    #[default]
    Fcfs,
    /// Order requests by priority and then arrival time.
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

    #[must_use]
    pub fn max_tokens(mut self, value: u32) -> Self {
        self.max_tokens = Some(value);
        self
    }

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

        Ok(Self {
            raw,
            _stop: stop,
            _stop_pointers: stop_pointers,
            _structured_string: structured_string,
            _choices: choices,
            _choice_pointers: choice_pointers,
        })
    }

    pub(crate) const fn raw(&self) -> &ffi::vllm_sampling_params {
        &self.raw
    }
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
        Some(0) | None => Ok(0),
        Some(value) => u32_to_i32(value, field),
    }
}

fn u32_to_i32(value: u32, field: &'static str) -> Result<i32, Error> {
    i32::try_from(value)
        .map_err(|_| invalid_configuration(format!("{field} exceeds native i32 range")))
}

fn length_to_i32(value: usize, field: &'static str) -> Result<i32, Error> {
    i32::try_from(value).map_err(|_| invalid_configuration(format!("too many {field}")))
}
