#!/bin/bash
# go_2b: identical to go_2.sh, but outputs on the container disk (/root) after a network-volume I/O error killed the first stage 2 run
# usage: bash go_2.sh ARM   (ARM is the stage 1 choice, e.g. R3; combined arms as RE1:R2)
set -e
ARM="$1"
cd /workspace/p7
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
{ python -c "import sys,torch,transformers;print('python',sys.version.split()[0]);print('torch',torch.__version__,'cuda',torch.version.cuda);print('transformers',transformers.__version__)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; pip freeze | grep -E "^(torch|transformers|accelerate|tokenizers|safetensors)=="; ls /workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/; echo "arm $ARM"; } > /root/env_2.txt
mkdir -p /root/smoke2 /root/out2 && cp out/stage1_meta.json /root/smoke2/ && cp out/stage1_meta.json /root/out2/
python phase7_llama_fix.py run2 phase7_prompts.jsonl stageB_dirs.npz /root/smoke2 "$ARM" --smoke > /root/smoke_2.log 2>&1
grep -q DONE /root/smoke_2.log
python phase7_llama_fix.py run2 phase7_prompts.jsonl stageB_dirs.npz /root/out2 "$ARM" > /root/run_2.log 2>&1
touch /root/done_2.flag
