#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
# Runs one baseline model over the four evaluation inputs of the release.
#   sbatch --export=ALL,MODEL=Qwen/Qwen2.5-VL-7B-Instruct,TAG=qwen25vl7b configs/baseline_eval.sbatch
export VLLM_WORKER_MULTIPROC_METHOD=spawn
R=$TCCC_VQA_RELEASE; O=${TCCC_VQA_RUNS:-$TCCC_VQA_WORK/runs}; mkdir -p "$O"
python -m tccc_vqa.eval.eval_vlm      --model "$MODEL" --eval-input "$R/eval/recognition_input.json"   --out "$O/eval_recog_${TAG}.jsonl" --tensor-parallel-size 2
python -m tccc_vqa.eval.eval_doctrine --model "$MODEL" --doctrine-input "$R/eval/doctrine_open_input.json" --out "$O/doctrine_${TAG}.jsonl" --tensor-parallel-size 2
python -m tccc_vqa.eval.eval_doctrine --model "$MODEL" --doctrine-input "$R/eval/how_input.json"      --out "$O/how_${TAG}.jsonl"      --tensor-parallel-size 2
python -m tccc_vqa.eval.eval_refusal  --model "$MODEL" --refusal-input "$R/eval/refusal_input.json"   --out "$O/refusal_${TAG}.jsonl"  --tensor-parallel-size 2
