use std::collections::BTreeSet;
use std::ffi::{c_char, c_void, CStr, CString};

use vllm_cpp_sys as ffi;

const _: [(); 17] = [(); ffi::VLLM_ABI_VERSION as usize];

unsafe extern "C" fn token_callback(
    _delta_text: *const c_char,
    _finished: bool,
    _user_data: *mut c_void,
) -> bool {
    true
}

unsafe extern "C" fn logits_processor(
    _token_ids: *const i32,
    _n_token_ids: i32,
    _logits: *mut f32,
    _vocab_size: i32,
    _user_data: *mut c_void,
) {
}

macro_rules! c_api_inventory {
    ($($name:ident: $signature:ty;)+) => {
        const C_API_SYMBOL_NAMES: &[&str] = &[$(stringify!($name)),+];

        #[test]
        fn every_c_api_symbol_links_with_the_header_signature() {
            let mut addresses = Vec::new();
            $(
                let function: $signature = ffi::$name;
                addresses.push(function as *const ());
            )+
            assert_eq!(addresses.len(), 35);
            assert!(addresses.iter().all(|address| !address.is_null()));
            assert_eq!(C_API_SYMBOL_NAMES.len(), 35);
            assert_eq!(
                C_API_SYMBOL_NAMES.iter().copied().collect::<BTreeSet<_>>().len(),
                35
            );
        }
    };
}

c_api_inventory! {
    vllm_model_params_default: unsafe extern "C" fn() -> ffi::vllm_model_params;
    vllm_sampling_params_default: unsafe extern "C" fn() -> ffi::vllm_sampling_params;
    vllm_engine_load: unsafe extern "C" fn(*const ffi::vllm_model_params, *mut *mut ffi::vllm_engine) -> ffi::vllm_status;
    vllm_engine_free: unsafe extern "C" fn(*mut ffi::vllm_engine);
    vllm_complete: unsafe extern "C" fn(*mut ffi::vllm_engine, *const c_char, *const ffi::vllm_sampling_params, *mut ffi::vllm_completion) -> ffi::vllm_status;
    vllm_complete_tokens: unsafe extern "C" fn(*mut ffi::vllm_engine, *const i32, i32, *const ffi::vllm_sampling_params, *mut i32, i32, *mut i32, *mut ffi::vllm_completion) -> ffi::vllm_status;
    vllm_complete_stream: unsafe extern "C" fn(*mut ffi::vllm_engine, *const c_char, *const ffi::vllm_sampling_params, ffi::vllm_token_callback, *mut c_void) -> ffi::vllm_status;
    vllm_request_submit: unsafe extern "C" fn(*mut ffi::vllm_engine, *const c_char, *const ffi::vllm_sampling_params, ffi::vllm_token_callback, *mut c_void, *mut *mut ffi::vllm_request) -> ffi::vllm_status;
    vllm_request_cancel: unsafe extern "C" fn(*mut ffi::vllm_request) -> ffi::vllm_status;
    vllm_request_wait: unsafe extern "C" fn(*mut ffi::vllm_request) -> ffi::vllm_status;
    vllm_request_done: unsafe extern "C" fn(*const ffi::vllm_request) -> bool;
    vllm_request_error: unsafe extern "C" fn(*const ffi::vllm_request) -> *const c_char;
    vllm_request_free: unsafe extern "C" fn(*mut ffi::vllm_request);
    vllm_chat: unsafe extern "C" fn(*mut ffi::vllm_engine, *const c_char, *mut *mut c_char) -> ffi::vllm_status;
    vllm_chat_stream: unsafe extern "C" fn(*mut ffi::vllm_engine, *const c_char, ffi::vllm_token_callback, *mut c_void) -> ffi::vllm_status;
    vllm_transcription_params_default: unsafe extern "C" fn() -> ffi::vllm_transcription_params;
    vllm_transcribe: unsafe extern "C" fn(*mut ffi::vllm_engine, *const ffi::vllm_transcription_params, *mut ffi::vllm_transcription) -> ffi::vllm_status;
    vllm_transcription_free: unsafe extern "C" fn(*mut ffi::vllm_transcription);
    vllm_embed: unsafe extern "C" fn(*mut ffi::vllm_engine, *const *const c_char, i32, *mut ffi::vllm_embedding_result) -> ffi::vllm_status;
    vllm_embedding_result_free: unsafe extern "C" fn(*mut ffi::vllm_embedding_result);
    vllm_video_model_params_default: unsafe extern "C" fn() -> ffi::vllm_video_model_params;
    vllm_video_params_default: unsafe extern "C" fn() -> ffi::vllm_video_params;
    vllm_video_engine_load: unsafe extern "C" fn(*const ffi::vllm_video_model_params, *mut *mut ffi::vllm_video_engine) -> ffi::vllm_status;
    vllm_video_engine_free: unsafe extern "C" fn(*mut ffi::vllm_video_engine);
    vllm_video_generate: unsafe extern "C" fn(*mut ffi::vllm_video_engine, *const ffi::vllm_video_params, *mut ffi::vllm_video_result) -> ffi::vllm_status;
    vllm_video_result_free: unsafe extern "C" fn(*mut ffi::vllm_video_result);
    vllm_video_mux_params_default: unsafe extern "C" fn() -> ffi::vllm_video_mux_params;
    vllm_video_mux_argv: unsafe extern "C" fn(*const ffi::vllm_video_mux_params, *mut *mut *mut c_char, *mut i32) -> ffi::vllm_status;
    vllm_video_mux_argv_free: unsafe extern "C" fn(*mut *mut c_char, i32);
    vllm_string_free: unsafe extern "C" fn(*mut c_char);
    vllm_completion_free: unsafe extern "C" fn(*mut ffi::vllm_completion);
    vllm_last_error: unsafe extern "C" fn() -> *const c_char;
    vllm_version: unsafe extern "C" fn() -> *const c_char;
    vllm_server_main: unsafe extern "C" fn(i32, *mut *mut c_char) -> i32;
    vllm_abi_version: unsafe extern "C" fn() -> i32;
}

#[test]
fn reports_target_identity_and_handles_invalid_model_path() {
    assert_eq!(ffi::VLLM_ABI_VERSION, 17);
    assert_eq!(unsafe { ffi::vllm_abi_version() }, 17);

    let version = unsafe { CStr::from_ptr(ffi::vllm_version()) };
    assert!(version.to_bytes().starts_with(b"0.0.2"), "{version:?}");

    let mut params = unsafe { ffi::vllm_model_params_default() };
    params.model_path = c"/nonexistent/vllm-cpp-rs-sys-model".as_ptr();
    let mut engine = std::ptr::null_mut();
    let status = unsafe { ffi::vllm_engine_load(&params, &mut engine) };

    assert_eq!(status, ffi::vllm_status_VLLM_ERR_MODEL_LOAD);
    assert!(engine.is_null());
    let error = unsafe { CStr::from_ptr(ffi::vllm_last_error()) };
    assert!(!error.to_bytes().is_empty());
}

#[test]
fn server_entry_point_is_present_and_nonblocking_for_help_or_server_off() {
    let mut arguments = [
        CString::new("vllm-server").unwrap(),
        CString::new("--help").unwrap(),
    ];
    let mut argv = arguments
        .iter_mut()
        .map(|argument| argument.as_ptr().cast_mut())
        .collect::<Vec<_>>();
    let status = unsafe { ffi::vllm_server_main(argv.len() as i32, argv.as_mut_ptr()) };
    assert!(matches!(status, 0 | 1));
    if status == 1 {
        let error = unsafe { CStr::from_ptr(ffi::vllm_last_error()) };
        assert!(error
            .to_bytes()
            .windows(13)
            .any(|part| part == b"without VLLM_"));
    }
}

#[test]
fn callback_types_match_header_contract_and_are_nullable() {
    let token: ffi::vllm_token_callback = Some(token_callback);
    let no_token: ffi::vllm_token_callback = None;
    let logits: ffi::vllm_logits_processor = Some(logits_processor);
    let no_logits: ffi::vllm_logits_processor = None;
    assert!(token.is_some());
    assert!(no_token.is_none());
    assert!(logits.is_some());
    assert!(no_logits.is_none());
}
