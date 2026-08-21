use std::ffi::{c_char, c_void};

use vllm_cpp_sys as ffi;

unsafe extern "C" fn token_callback(
    _delta_text: *const c_char,
    _finished: bool,
    _user_data: *mut c_void,
) -> bool {
    true
}

#[test]
fn every_c_api_symbol_links() {
    assert_ne!(ffi::vllm_model_params_default as *const () as usize, 0);
    assert_ne!(ffi::vllm_sampling_params_default as *const () as usize, 0);
    assert_ne!(ffi::vllm_engine_load as *const () as usize, 0);
    assert_ne!(ffi::vllm_engine_free as *const () as usize, 0);
    assert_ne!(ffi::vllm_complete as *const () as usize, 0);
    assert_ne!(ffi::vllm_complete_stream as *const () as usize, 0);
    assert_ne!(ffi::vllm_request_submit as *const () as usize, 0);
    assert_ne!(ffi::vllm_request_cancel as *const () as usize, 0);
    assert_ne!(ffi::vllm_request_wait as *const () as usize, 0);
    assert_ne!(ffi::vllm_request_done as *const () as usize, 0);
    assert_ne!(ffi::vllm_request_error as *const () as usize, 0);
    assert_ne!(ffi::vllm_request_free as *const () as usize, 0);
    assert_ne!(ffi::vllm_chat as *const () as usize, 0);
    assert_ne!(ffi::vllm_chat_stream as *const () as usize, 0);
    assert_ne!(ffi::vllm_string_free as *const () as usize, 0);
    assert_ne!(ffi::vllm_completion_free as *const () as usize, 0);
    assert_ne!(ffi::vllm_last_error as *const () as usize, 0);
    assert_ne!(ffi::vllm_version as *const () as usize, 0);
    assert_ne!(ffi::vllm_abi_version as *const () as usize, 0);
}

#[test]
fn reports_expected_abi_and_handles_invalid_model_path() {
    assert_eq!(unsafe { ffi::vllm_abi_version() }, 10);

    let mut params = unsafe { ffi::vllm_model_params_default() };
    params.model_path = c"/nonexistent/vllm-cpp-rs-bootstrap-model".as_ptr();
    let mut engine = std::ptr::null_mut();
    let status = unsafe { ffi::vllm_engine_load(&params, &mut engine) };

    assert_eq!(status, 2);
    assert!(engine.is_null());
    let error = unsafe { std::ffi::CStr::from_ptr(ffi::vllm_last_error()) };
    assert!(!error.to_bytes().is_empty());
}

#[test]
fn callback_type_matches_header_contract() {
    let callback: ffi::vllm_token_callback = Some(token_callback);
    assert!(callback.is_some());
}
