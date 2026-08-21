#include <vllm.h>

#include <stddef.h>
#include <stdio.h>

#define PRINT_LAYOUT(type)                                                     \
  printf(#type " %zu %zu\n", sizeof(type), _Alignof(type))
#define PRINT_OFFSET(type, field)                                              \
  printf(#type "." #field " %zu\n", offsetof(type, field))

int main(void) {
  PRINT_LAYOUT(vllm_status);

  PRINT_LAYOUT(vllm_model_params);
  PRINT_OFFSET(vllm_model_params, model_path);
  PRINT_OFFSET(vllm_model_params, tokenizer_config_path);
  PRINT_OFFSET(vllm_model_params, block_size);
  PRINT_OFFSET(vllm_model_params, num_blocks);
  PRINT_OFFSET(vllm_model_params, max_model_len);
  PRINT_OFFSET(vllm_model_params, max_num_seqs);
  PRINT_OFFSET(vllm_model_params, tool_parser);
  PRINT_OFFSET(vllm_model_params, reasoning_parser);
  PRINT_OFFSET(vllm_model_params, speculative_config);
  PRINT_OFFSET(vllm_model_params, enable_prefix_caching);
  PRINT_OFFSET(vllm_model_params, max_num_batched_tokens);
  PRINT_OFFSET(vllm_model_params, scheduling_policy);
  PRINT_OFFSET(vllm_model_params, kv_transfer_config);
  PRINT_OFFSET(vllm_model_params, enable_jump_forward);

  PRINT_LAYOUT(vllm_sampling_params);
  PRINT_OFFSET(vllm_sampling_params, temperature);
  PRINT_OFFSET(vllm_sampling_params, top_p);
  PRINT_OFFSET(vllm_sampling_params, top_k);
  PRINT_OFFSET(vllm_sampling_params, min_p);
  PRINT_OFFSET(vllm_sampling_params, max_tokens);
  PRINT_OFFSET(vllm_sampling_params, seed);
  PRINT_OFFSET(vllm_sampling_params, has_seed);
  PRINT_OFFSET(vllm_sampling_params, presence_penalty);
  PRINT_OFFSET(vllm_sampling_params, frequency_penalty);
  PRINT_OFFSET(vllm_sampling_params, repetition_penalty);
  PRINT_OFFSET(vllm_sampling_params, min_tokens);
  PRINT_OFFSET(vllm_sampling_params, ignore_eos);
  PRINT_OFFSET(vllm_sampling_params, stop);
  PRINT_OFFSET(vllm_sampling_params, n_stop);
  PRINT_OFFSET(vllm_sampling_params, structured_json);
  PRINT_OFFSET(vllm_sampling_params, structured_regex);
  PRINT_OFFSET(vllm_sampling_params, structured_choice);
  PRINT_OFFSET(vllm_sampling_params, n_structured_choice);
  PRINT_OFFSET(vllm_sampling_params, structured_grammar);
  PRINT_OFFSET(vllm_sampling_params, structured_json_object);
  PRINT_OFFSET(vllm_sampling_params, logits_processor);
  PRINT_OFFSET(vllm_sampling_params, logits_processor_user_data);

  PRINT_LAYOUT(vllm_completion);
  PRINT_OFFSET(vllm_completion, text);
  PRINT_OFFSET(vllm_completion, finish_reason);
  PRINT_OFFSET(vllm_completion, prompt_tokens);
  PRINT_OFFSET(vllm_completion, completion_tokens);

  PRINT_LAYOUT(vllm_token_callback);
  PRINT_LAYOUT(vllm_logits_processor);
  return 0;
}
