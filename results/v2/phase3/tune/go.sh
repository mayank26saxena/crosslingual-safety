#!/bin/bash
set -e
cd /workspace/p3
export HF_HOME=/workspace/hf
pip install -q --break-system-packages transformers==5.22.2 accelerate==1.15.0 tokenizers==0.23.3
{ python -c "import sys,torch,transformers;print('python',sys.version.split()[0]);print('torch',torch.__version__,'cuda',torch.version.cuda);print('transformers',transformers.__version__)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; pip freeze | grep -E "^(torch|transformers|accelerate|tokenizers|safetensors)=="; } > env.txt || true
python gated_steering.py tune phase3_prompts.jsonl . smoke --smoke > smoke.log 2>&1
python gated_steering.py tune phase3_prompts.jsonl . tune > tune.log 2>&1
ls /workspace/hf/hub/models--Qwen--Qwen3-8B/snapshots/ >> env.txt
touch done.flag
