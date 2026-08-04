#!/bin/bash
# Dedicated LLM-as-judge backend (Qwen2.5-14B-Instruct) on :18003.
# Not part of the Ray Serve pool -- only eval_cuad.py talks to this.
# Shares GPU 1 with vllm2, so bring vllm2 down before starting this
# (see README's "LLM-as-judge evaluation" section).
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

export LD_LIBRARY_PATH="/venv/main/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=1

source /venv/main/bin/activate
cd "${WORKSPACE}"
pty vllm serve Qwen/Qwen2.5-14B-Instruct \
    --host 127.0.0.1 \
    --port 18003 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --enable-prefix-caching 2>&1
