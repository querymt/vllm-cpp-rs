# vllm-cpp-sys

Raw Rust bindings and a bundled native build for [vllm.cpp](https://github.com/mudler/vllm.cpp).

This crate is an implementation detail for `vllm-cpp`. Its public functions are unsafe and mirror the stable C API. The bundled source is pinned to commit `34aedfbe8ed9779697905541a62e2160ccfd9c05` and builds CPU-only by default.

The Rust crate is dual-licensed under MIT or Apache-2.0. The bundled vllm.cpp source retains its upstream Apache-2.0 license and notices.

vllm.cpp is an independent community project and is not affiliated with or endorsed by the vLLM project, the PyTorch Foundation, or the Linux Foundation.
