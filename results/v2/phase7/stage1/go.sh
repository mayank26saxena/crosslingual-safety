#!/bin/bash
set -e
cd /workspace/p7
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
pip install -q --break-system-packages transformers==5.22.2 accelerate==1.15.0 tokenizers==0.23.3
{ python -c "import sys,torch,transformers;print('python',sys.version.split()[0]);print('torch',torch.__version__,'cuda',torch.version.cuda);print('transformers',transformers.__version__)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; pip freeze | grep -E "^(torch|transformers|accelerate|tokenizers|safetensors)=="; ls /workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/; } > env.txt
python phase7_llama_fix.py run1 phase7_prompts.jsonl stageB_dirs.npz smoke --smoke > smoke.log 2>&1
grep -q DONE smoke.log
touch smoke.flag
python phase7_llama_fix.py run1 phase7_prompts.jsonl stageB_dirs.npz out > run.log 2>&1
touch done.flag
