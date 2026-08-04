#!/bin/bash
# GPU 1 -- second generation backend (zephyr-7b-alpha), OpenAI-compatible on :18002
# Identical to vllm.sh except CUDA_VISIBLE_DEVICES and the port.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

export LD_LIBRARY_PATH="/venv/main/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=1

source /venv/main/bin/activate
cd "${WORKSPACE}"
pty vllm serve HuggingFaceH4/zephyr-7b-alpha \
    --host 127.0.0.1 \
    --port 18002 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --enable-prefix-caching 2>&1
