//! Raw bindings to the stable vllm.cpp C API.
//!
//! Complete generated bindings will be added in a follow-up. This bootstrap
//! declares the current symbols so the packaged native library can be linked and probed.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_float, c_int, c_void};

#[repr(C)]
pub struct vllm_engine {
    _private: [u8; 0],
}

#[repr(C)]
pub struct vllm_request {
    _private: [u8; 0],
}

pub type vllm_status = c_int;
pub type vllm_token_callback = Option<
    unsafe extern "C" fn(delta_text: *const c_char, finished: bool, user_data: *mut c_void) -> bool,
>;
pub type vllm_logits_processor = Option<
    unsafe extern "C" fn(
        token_ids: *const i32,
        n_token_ids: i32,
        logits: *mut c_float,
        vocab_size: i32,
        user_data: *mut c_void,
    ),
>;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct vllm_model_params {
    pub model_path: *const c_char,
    pub tokenizer_config_path: *const c_char,
    pub block_size: i32,
    pub num_blocks: i32,
    pub max_model_len: i32,
    pub max_num_seqs: i32,
    pub tool_parser: *const c_char,
    pub reasoning_parser: *const c_char,
    pub speculative_config: *const c_char,
    pub enable_prefix_caching: i32,
    pub max_num_batched_tokens: i32,
    pub scheduling_policy: *const c_char,
    pub kv_transfer_config: *const c_char,
    pub enable_jump_forward: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct vllm_sampling_params {
    pub temperature: c_float,
    pub top_p: c_float,
    pub top_k: i32,
    pub min_p: c_float,
    pub max_tokens: i32,
    pub seed: u64,
    pub has_seed: i32,
    pub presence_penalty: c_float,
    pub frequency_penalty: c_float,
    pub repetition_penalty: c_float,
    pub min_tokens: i32,
    pub ignore_eos: i32,
    pub stop: *const *const c_char,
    pub n_stop: i32,
    pub structured_json: *const c_char,
    pub structured_regex: *const c_char,
    pub structured_choice: *const *const c_char,
    pub n_structured_choice: i32,
    pub structured_grammar: *const c_char,
    pub structured_json_object: i32,
    pub logits_processor: vllm_logits_processor,
    pub logits_processor_user_data: *mut c_void,
}

#[repr(C)]
pub struct vllm_completion {
    pub text: *mut c_char,
    pub finish_reason: *const c_char,
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
}

unsafe extern "C" {
    pub fn vllm_model_params_default() -> vllm_model_params;
    pub fn vllm_sampling_params_default() -> vllm_sampling_params;
    pub fn vllm_engine_load(
        params: *const vllm_model_params,
        out: *mut *mut vllm_engine,
    ) -> vllm_status;
    pub fn vllm_engine_free(engine: *mut vllm_engine);
    pub fn vllm_complete(
        engine: *mut vllm_engine,
        prompt: *const c_char,
        params: *const vllm_sampling_params,
        out: *mut vllm_completion,
    ) -> vllm_status;
    pub fn vllm_complete_stream(
        engine: *mut vllm_engine,
        prompt: *const c_char,
        params: *const vllm_sampling_params,
        callback: vllm_token_callback,
        user_data: *mut c_void,
    ) -> vllm_status;
    pub fn vllm_request_submit(
        engine: *mut vllm_engine,
        prompt: *const c_char,
        params: *const vllm_sampling_params,
        callback: vllm_token_callback,
        user_data: *mut c_void,
        out: *mut *mut vllm_request,
    ) -> vllm_status;
    pub fn vllm_request_cancel(request: *mut vllm_request) -> vllm_status;
    pub fn vllm_request_wait(request: *mut vllm_request) -> vllm_status;
    pub fn vllm_request_done(request: *const vllm_request) -> bool;
    pub fn vllm_request_error(request: *const vllm_request) -> *const c_char;
    pub fn vllm_request_free(request: *mut vllm_request);
    pub fn vllm_chat(
        engine: *mut vllm_engine,
        request_json: *const c_char,
        out_response_json: *mut *mut c_char,
    ) -> vllm_status;
    pub fn vllm_chat_stream(
        engine: *mut vllm_engine,
        request_json: *const c_char,
        callback: vllm_token_callback,
        user_data: *mut c_void,
    ) -> vllm_status;
    pub fn vllm_string_free(string: *mut c_char);
    pub fn vllm_completion_free(completion: *mut vllm_completion);
    pub fn vllm_last_error() -> *const c_char;
    pub fn vllm_version() -> *const c_char;
    pub fn vllm_abi_version() -> i32;
}
