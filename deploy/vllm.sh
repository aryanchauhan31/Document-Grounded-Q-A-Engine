#!/bin/bash
# GPU 0 -- primary generation backend (zephyr-7b-alpha), OpenAI-compatible on :18000
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

# Forward-compat libs: lets a newer CUDA-major torch build run on an older
# host driver (see README's "Real infra bugs" section for why this is needed).
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH}"
export FLASHINFER_DISABLE_VERSION_CHECK=1

source /venv/main/bin/activate
cd "${WORKSPACE}"
pty vllm serve HuggingFaceH4/zephyr-7b-alpha \
    --host 127.0.0.1 \
    --port 18000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --enable-prefix-caching 2>&1
