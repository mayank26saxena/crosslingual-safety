export HF_HOME=/workspace/hf
python regen_calibration.py generate hindi_all_prompts_hi.jsonl rep_hi.jsonl --max-new 1024 --batch 16 > run_hi.log 2>&1
python regen_calibration.py generate hindi_all_prompts_en.jsonl rep_en.jsonl --max-new 512 --batch 16 > run_en.log 2>&1
{ python -c "import torch,transformers,sys;print(\"python\",sys.version.split()[0]);print(\"torch\",torch.__version__,\"cuda\",torch.version.cuda);print(\"transformers\",transformers.__version__)"; ls /workspace/hf/hub/models--Qwen--Qwen3-8B/snapshots/; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; pip freeze | grep -iE "^(torch|transformers|accelerate|tokenizers|safetensors)=" ; } > env.txt
echo ALLDONE > done.flag
