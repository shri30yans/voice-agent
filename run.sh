#!/usr/bin/env bash
# ./run.sh            start the agent (LLM from .env)
# ./run.sh --local    also serve a local LLM with mlx_lm.server and use it instead
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  uv venv -q --python 3.11 .venv && uv pip install -q --python .venv/bin/python -r requirements.txt
fi
[ -f models/silero_vad.onnx ] || { mkdir -p models; curl -sL -o models/silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx; }
if [ "$1" = "--local" ]; then
  .venv/bin/mlx_lm.server --model "${LOCAL_LLM:-mlx-community/Qwen3-4B-Instruct-2507-4bit}" --port 8081 >/tmp/voiceagent-llm.log 2>&1 &
  trap "kill $! 2>/dev/null" EXIT
  export LLM_BASE_URL=http://127.0.0.1:8081/v1 LLM_API_KEY=local LLM_MODEL="${LOCAL_LLM:-mlx-community/Qwen3-4B-Instruct-2507-4bit}"
fi
exec .venv/bin/python server.py
