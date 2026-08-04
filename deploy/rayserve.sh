#!/bin/bash
# Ray Serve entrypoint -- autoscaling orchestration/agent tier.
# Binds Ray Serve's HTTP proxy to 127.0.0.1:8000 by default.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

export LD_LIBRARY_PATH="/venv/main/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH}"
export FLASHINFER_DISABLE_VERSION_CHECK=1

source /venv/main/bin/activate
cd /workspace
pty serve run rag_serve:rag_app 2>&1
