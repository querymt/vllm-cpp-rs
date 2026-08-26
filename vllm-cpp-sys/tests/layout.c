#include <vllm.h>

#include <stddef.h>
#include <stdio.h>

#define PRINT_LAYOUT(type)                                                     \
  printf(#type " %zu %zu\n", sizeof(type), _Alignof(type))
#define PRINT_NAMED_LAYOUT(name, type)                                         \
  printf(#name " %zu %zu\n", sizeof(type), _Alignof(type))
#define PRINT_OFFSET(type, field)                                              \
  printf(#type "." #field " %zu\n", offsetof(type, field))
#define PRINT_VALUE(name) printf(#name " %d\n", (int)(name))

int main(void) {
  PRINT_LAYOUT(vllm_status);
  PRINT_VALUE(VLLM_OK);
  PRINT_VALUE(VLLM_ERR_INVALID_ARGUMENT);
  PRINT_VALUE(VLLM_ERR_MODEL_LOAD);
  PRINT_VALUE(VLLM_ERR_RUNTIME);
  PRINT_VALUE(VLLM_ERR_UNKNOWN);

  PRINT_NAMED_LAYOUT(vllm_engine_ptr, vllm_engine*);
  PRINT_NAMED_LAYOUT(vllm_request_ptr, vllm_request*);
  PRINT_NAMED_LAYOUT(vllm_video_engine_ptr, vllm_video_engine*);

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
  PRINT_OFFSET(vllm_model_params, device);
  PRINT_OFFSET(vllm_model_params, gpu_memory_utilization);
  PRINT_OFFSET(vllm_model_params, kv_cache_memory_bytes);

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

  PRINT_LAYOUT(vllm_transcription_params);
  PRINT_OFFSET(vllm_transcription_params, audio_path);
  PRINT_OFFSET(vllm_transcription_params, pcm);
  PRINT_OFFSET(vllm_transcription_params, n_samples);
  PRINT_OFFSET(vllm_transcription_params, sample_rate);

  PRINT_LAYOUT(vllm_transcription);
  PRINT_OFFSET(vllm_transcription, text);
  PRINT_OFFSET(vllm_transcription, token_ids);
  PRINT_OFFSET(vllm_transcription, n_token_ids);
  PRINT_OFFSET(vllm_transcription, has_text);

  PRINT_LAYOUT(vllm_embedding_result);
  PRINT_OFFSET(vllm_embedding_result, values);
  PRINT_OFFSET(vllm_embedding_result, n_embeddings);
  PRINT_OFFSET(vllm_embedding_result, dim);
  PRINT_OFFSET(vllm_embedding_result, prompt_tokens);

  PRINT_LAYOUT(vllm_video_model_params);
  PRINT_OFFSET(vllm_video_model_params, dit_path);
  PRINT_OFFSET(vllm_video_model_params, encoder_path);
  PRINT_OFFSET(vllm_video_model_params, tokenizer_path);
  PRINT_OFFSET(vllm_video_model_params, video_vae_path);
  PRINT_OFFSET(vllm_video_model_params, video_vae_config_path);
  PRINT_OFFSET(vllm_video_model_params, audio_vae_path);
  PRINT_OFFSET(vllm_video_model_params, audio_vae_config_path);
  PRINT_OFFSET(vllm_video_model_params, prompt_embeds_path);
  PRINT_OFFSET(vllm_video_model_params, partition);
  PRINT_OFFSET(vllm_video_model_params, device);
  PRINT_OFFSET(vllm_video_model_params, dequant_bf16);
  PRINT_OFFSET(vllm_video_model_params, fp4_resident);

  PRINT_LAYOUT(vllm_video_params);
  PRINT_OFFSET(vllm_video_params, prompt);
  PRINT_OFFSET(vllm_video_params, width);
  PRINT_OFFSET(vllm_video_params, height);
  PRINT_OFFSET(vllm_video_params, num_frames);
  PRINT_OFFSET(vllm_video_params, steps);
  PRINT_OFFSET(vllm_video_params, seed);
  PRINT_OFFSET(vllm_video_params, has_seed);
  PRINT_OFFSET(vllm_video_params, first_frame);
  PRINT_OFFSET(vllm_video_params, last_frame);
  PRINT_OFFSET(vllm_video_params, ref_image);
  PRINT_OFFSET(vllm_video_params, ref_video);
  PRINT_OFFSET(vllm_video_params, ref_audio);
  PRINT_OFFSET(vllm_video_params, noise_aug);
  PRINT_OFFSET(vllm_video_params, output_dir);

  PRINT_LAYOUT(vllm_video_result);
  PRINT_OFFSET(vllm_video_result, frame_dir);
  PRINT_OFFSET(vllm_video_result, audio_path);
  PRINT_OFFSET(vllm_video_result, frame_count);
  PRINT_OFFSET(vllm_video_result, width);
  PRINT_OFFSET(vllm_video_result, height);
  PRINT_OFFSET(vllm_video_result, fps);
  PRINT_OFFSET(vllm_video_result, sample_rate);
  PRINT_OFFSET(vllm_video_result, mux_argv);
  PRINT_OFFSET(vllm_video_result, mux_argc);

  PRINT_LAYOUT(vllm_video_mux_params);
  PRINT_OFFSET(vllm_video_mux_params, frames);
  PRINT_OFFSET(vllm_video_mux_params, audio_path);
  PRINT_OFFSET(vllm_video_mux_params, output_path);
  PRINT_OFFSET(vllm_video_mux_params, fps);
  PRINT_OFFSET(vllm_video_mux_params, crf);

  PRINT_LAYOUT(vllm_token_callback);
  PRINT_LAYOUT(vllm_logits_processor);
  return 0;
}
