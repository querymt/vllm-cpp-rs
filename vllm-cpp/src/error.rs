use std::ffi::CStr;
use std::fmt;

use vllm_cpp_sys as ffi;

/// An error returned while resolving a model from the Hugging Face Hub.
///
/// External transport errors are converted to contextual strings so this type
/// remains stable, cloneable, and comparable.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum HuggingFaceError {
    /// A repository, revision, or filename is invalid.
    InvalidInput { message: String },
    /// The requested revision is not present in the selected local cache.
    CacheMiss { message: String },
    /// A repository snapshot lacks required runtime files or metadata.
    Incomplete { message: String },
    /// A Hugging Face API or download operation failed.
    Hub { message: String },
    /// A local cache operation failed.
    Io { message: String },
}

impl fmt::Display for HuggingFaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { message } => write!(f, "invalid Hugging Face model: {message}"),
            Self::CacheMiss { message } => write!(f, "Hugging Face cache miss: {message}"),
            Self::Incomplete { message } => {
                write!(f, "incomplete Hugging Face snapshot: {message}")
            }
            Self::Hub { message } => write!(f, "Hugging Face Hub failure: {message}"),
            Self::Io { message } => write!(f, "Hugging Face cache I/O failure: {message}"),
        }
    }
}

impl std::error::Error for HuggingFaceError {}

/// An error returned by the safe vllm.cpp wrapper.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// The loaded native library does not match the generated C ABI.
    AbiMismatch { expected: i32, actual: i32 },
    /// Native code rejected caller input.
    InvalidArgument { message: String },
    /// The model, tokenizer, configuration, or weights could not be loaded.
    ModelLoad { message: String },
    /// Native generation failed at runtime.
    Runtime { message: String },
    /// Native code reported an unclassified failure.
    NativeUnknown { message: String },
    /// Native code returned a status unknown to these bindings.
    UnknownStatus { status: u32, message: String },
    /// A value cannot cross the C boundary because it contains a NUL byte.
    InteriorNul { field: &'static str },
    /// A platform path cannot be represented by the native UTF-8 API.
    PathEncoding,
    /// Native code returned bytes that are not valid UTF-8.
    InvalidUtf8 { field: &'static str },
    /// An asynchronous callback panicked.
    CallbackPanicked,
    /// A request operation was attempted from that request's callback thread.
    RequestCallbackThread { operation: &'static str },
    /// A Rust-side parameter cannot be represented by the native API.
    InvalidConfiguration { message: String },
    /// JSON serialization or parsing failed.
    Json {
        context: &'static str,
        message: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AbiMismatch { expected, actual } => {
                write!(
                    f,
                    "vllm.cpp ABI mismatch: expected {expected}, found {actual}"
                )
            }
            Self::InvalidArgument { message } => write!(f, "invalid argument: {message}"),
            Self::ModelLoad { message } => write!(f, "model load failed: {message}"),
            Self::Runtime { message } => write!(f, "vllm.cpp runtime failure: {message}"),
            Self::NativeUnknown { message } => write!(f, "unknown native failure: {message}"),
            Self::UnknownStatus { status, message } => {
                write!(f, "unknown native status {status}: {message}")
            }
            Self::InteriorNul { field } => write!(f, "{field} contains an interior NUL byte"),
            Self::PathEncoding => write!(f, "path cannot be represented by the native API"),
            Self::InvalidUtf8 { field } => write!(f, "native {field} is not valid UTF-8"),
            Self::CallbackPanicked => write!(f, "asynchronous request callback panicked"),
            Self::RequestCallbackThread { operation } => {
                write!(
                    f,
                    "cannot {operation} a request from its own callback thread"
                )
            }
            Self::InvalidConfiguration { message } => {
                write!(f, "invalid configuration: {message}")
            }
            Self::Json { context, message } => write!(f, "{context}: {message}"),
        }
    }
}

impl std::error::Error for Error {}

pub(crate) fn status_result(status: ffi::vllm_status) -> Result<(), Error> {
    if status == ffi::vllm_status_VLLM_OK {
        return Ok(());
    }

    // The native diagnostic is thread-local and valid only until the next C API
    // call on this thread, so copy it before doing any other FFI work.
    let message = unsafe {
        let pointer = ffi::vllm_last_error();
        if pointer.is_null() {
            String::new()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    let error = match status {
        ffi::vllm_status_VLLM_ERR_INVALID_ARGUMENT => Error::InvalidArgument { message },
        ffi::vllm_status_VLLM_ERR_MODEL_LOAD => Error::ModelLoad { message },
        ffi::vllm_status_VLLM_ERR_RUNTIME => Error::Runtime { message },
        ffi::vllm_status_VLLM_ERR_UNKNOWN => Error::NativeUnknown { message },
        status => Error::UnknownStatus { status, message },
    };
    Err(error)
}

pub(crate) fn invalid_configuration(message: impl Into<String>) -> Error {
    Error::InvalidConfiguration {
        message: message.into(),
    }
}
