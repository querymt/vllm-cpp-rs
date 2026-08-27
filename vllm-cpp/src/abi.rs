use vllm_cpp_sys as ffi;

use crate::Error;

/// Proof that the linked library exactly matches the generated ABI.
pub(crate) struct Compatibility {
    _private: (),
}

impl Compatibility {
    pub(crate) fn check() -> Result<Self, Error> {
        // SAFETY: this base ABI function takes no pointers or versioned structs.
        let actual = unsafe { ffi::vllm_abi_version() };
        Self::from_actual(actual)
    }

    pub(crate) fn model_params_default(&self) -> ffi::vllm_model_params {
        // SAFETY: possession of this token proves exact ABI equality for this
        // engine construction before returning the versioned struct by value.
        unsafe { ffi::vllm_model_params_default() }
    }

    pub(crate) fn sampling_params_default(&self) -> ffi::vllm_sampling_params {
        // SAFETY: the engine retained this token after exact ABI equality was
        // established, so this versioned struct may be returned by value.
        unsafe { ffi::vllm_sampling_params_default() }
    }

    pub(crate) fn transcription_params_default(&self) -> ffi::vllm_transcription_params {
        // SAFETY: the engine retained this token after exact ABI equality was
        // established, so this versioned struct may be returned by value.
        unsafe { ffi::vllm_transcription_params_default() }
    }

    pub(crate) fn video_model_params_default(&self) -> ffi::vllm_video_model_params {
        // SAFETY: possession of this token proves exact ABI equality before
        // this versioned struct is returned by value.
        unsafe { ffi::vllm_video_model_params_default() }
    }

    pub(crate) fn video_params_default(&self) -> ffi::vllm_video_params {
        // SAFETY: possession of this token proves exact ABI equality before
        // this versioned struct is returned by value.
        unsafe { ffi::vllm_video_params_default() }
    }

    pub(crate) fn video_mux_params_default(&self) -> ffi::vllm_video_mux_params {
        // SAFETY: possession of this token proves exact ABI equality before
        // this versioned struct is returned by value.
        unsafe { ffi::vllm_video_mux_params_default() }
    }

    fn from_actual(actual: i32) -> Result<Self, Error> {
        let expected = ffi::VLLM_ABI_VERSION as i32;
        if actual != expected {
            return Err(Error::AbiMismatch { expected, actual });
        }
        Ok(Self { _private: () })
    }

    #[cfg(test)]
    pub(crate) fn check_with(abi_version: impl FnOnce() -> i32) -> Result<Self, Error> {
        Self::from_actual(abi_version())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::Compatibility;
    use crate::Error;

    #[test]
    fn mismatch_produces_no_token_or_default_access() {
        let calls = RefCell::new(Vec::new());
        let result = Compatibility::check_with(|| {
            calls.borrow_mut().push("abi");
            10
        });

        assert!(matches!(
            result,
            Err(Error::AbiMismatch {
                expected: 17,
                actual: 10
            })
        ));
        assert_eq!(*calls.borrow(), ["abi"]);
    }
}
