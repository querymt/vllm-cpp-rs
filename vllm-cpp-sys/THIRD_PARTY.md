# Third-Party Notices

`vllm-cpp-sys` redistributes the pinned vllm.cpp source required for offline native builds. vllm.cpp is Apache-2.0 and incorporates or vendors components under their own licenses.

| Component | Packaged source | Provenance | License text |
|---|---|---|---|
| vllm.cpp | `vllm.cpp/**` | vllm.cpp commit `34aedfbe8ed9779697905541a62e2160ccfd9c05` | `vllm.cpp/LICENSE`, `vllm.cpp/NOTICE` |
| BLAKE3 C reference | `vllm.cpp/third_party/blake3/**` | BLAKE3 1.5.5, commit `81f772a` | `vllm.cpp/third_party/blake3/LICENSE_A2`, `vllm.cpp/third_party/blake3/LICENSE_CC0` |
| google/minja | `vllm.cpp/third_party/minja/**` | minja commit `021c229` | `vllm.cpp/third_party/minja/LICENSE` |
| Vulkan-Headers | `vllm.cpp/third_party/vulkan/**` | Vulkan SDK 1.4.328.1, generated Khronos headers | Apache-2.0 in each generated header and `vllm.cpp/LICENSE` |
| FlashAttention-2 slice | `vllm.cpp/src/vt/cuda/flash_attn/**` | `vllm-project/flash-attention` commit `2c839c33`, as recorded by the vllm.cpp import | `licenses/FLASH-ATTENTION-BSD-3-CLAUSE.txt` |
| Marlin / GPTQ-Marlin slice | `vllm.cpp/src/vt/cuda/marlin/**` | vLLM commit `e24d1b24`, with retained file notices and vllm.cpp adapter changes | Apache-2.0 in `vllm.cpp/LICENSE` and retained source notices |
| Flash Linear Attention Triton kernels | `vllm.cpp/triton_kernels/**`, generated AOT files under `vllm.cpp/src/vt/cuda/triton_aot_vendored/**` | FLA source ported through vLLM 0.24.0; exact source and artifact hashes are pinned in each AOT `MANIFEST` | `licenses/FLASH-LINEAR-ATTENTION-MIT.txt` |
| nlohmann/json | `vllm.cpp/third_party/nlohmann/**` | nlohmann/json 3.12.0 | MIT notice retained in `json.hpp` |

The package excludes upstream tests, fixtures, benchmarks, models, external SDKs, and build output. It contains only native source/build inputs and required licenses/notices.

vllm.cpp is an independent community project and is not affiliated with or endorsed by the vLLM project, the PyTorch Foundation, or the Linux Foundation.
