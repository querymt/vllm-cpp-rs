use std::any::Any;
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::error::Error;

/// Controls whether native streaming continues after a callback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamControl {
    Continue,
    Stop,
}

/// One copied streaming delta.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamEvent {
    pub delta: String,
    pub finished: bool,
}

/// How a successful blocking stream ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamOutcome {
    pub stopped_by_callback: bool,
}

pub(crate) struct CallbackState<'callback, F> {
    callback: &'callback mut F,
    stopped: bool,
    error: Option<Error>,
    panic: Option<Box<dyn Any + Send>>,
}

impl<'callback, F> CallbackState<'callback, F> {
    pub(crate) fn new(callback: &'callback mut F) -> Self {
        Self {
            callback,
            stopped: false,
            error: None,
            panic: None,
        }
    }

    pub(crate) const fn stopped(&self) -> bool {
        self.stopped
    }

    pub(crate) fn take_error(&mut self) -> Option<Error> {
        self.error.take()
    }

    pub(crate) fn take_panic(&mut self) -> Option<Box<dyn Any + Send>> {
        self.panic.take()
    }

    fn apply_control(&mut self, control: StreamControl, finished: bool) -> bool {
        match control {
            StreamControl::Continue => true,
            StreamControl::Stop => {
                if !finished {
                    self.stopped = true;
                }
                false
            }
        }
    }
}

pub(crate) unsafe extern "C" fn callback_trampoline<F>(
    delta_text: *const c_char,
    finished: bool,
    user_data: *mut c_void,
) -> bool
where
    F: FnMut(StreamEvent) -> StreamControl,
{
    // SAFETY: callers pass a stable pointer to CallbackState<F> and the native
    // blocking function cannot retain it after returning.
    let state = unsafe { &mut *user_data.cast::<CallbackState<'_, F>>() };
    if delta_text.is_null() {
        state.error = Some(Error::InvalidUtf8 {
            field: "stream delta",
        });
        return false;
    }
    // SAFETY: vllm.cpp promises a borrowed NUL-terminated string for the callback.
    let delta = match unsafe { CStr::from_ptr(delta_text) }.to_str() {
        Ok(delta) => delta.to_owned(),
        Err(_) => {
            state.error = Some(Error::InvalidUtf8 {
                field: "stream delta",
            });
            return false;
        }
    };
    let event = StreamEvent { delta, finished };
    match catch_unwind(AssertUnwindSafe(|| (state.callback)(event))) {
        Ok(control) => state.apply_control(control, finished),
        Err(payload) => {
            state.panic = Some(payload);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CallbackState, StreamControl, StreamEvent};

    #[test]
    fn only_nonterminal_stop_marks_callback_stop() {
        let mut callback = |_: StreamEvent| StreamControl::Continue;
        let mut state = CallbackState::new(&mut callback);

        assert!(!state.apply_control(StreamControl::Stop, true));
        assert!(!state.stopped());
        assert!(!state.apply_control(StreamControl::Stop, false));
        assert!(state.stopped());
    }
}
