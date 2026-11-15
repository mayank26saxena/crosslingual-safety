#!/bin/bash
set -e
cd /workspace/p5
export HF_HOME=/workspace/hf
{ python -c "import sys,torch,transformers;print('python',sys.version.split()[0]);print('torch',torch.__version__,'cuda',torch.version.cuda);print('transformers',transformers.__version__)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; pip freeze | grep -E "^(torch|transformers|accelerate|tokenizers|safetensors)=="; } > env.txt || true
python phase5_neurons.py run phase5_prompts.jsonl sets.json . out --smoke > smoke.log 2>&1
grep -q DONE smoke.log
touch smoke.flag
python phase5_neurons.py run phase5_prompts.jsonl sets.json . out > run.log 2>&1
ls /workspace/hf/hub/models--Qwen--Qwen3-8B/snapshots/ >> env.txt || true
touch done.flag
