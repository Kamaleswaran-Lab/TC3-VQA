#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00
# Blinded LLM judge for the reference-versus-model-authored comparison.
#   sbatch --export=ALL,RUN=full,MODEL=meta-llama/Llama-3.3-70B-Instruct,TAG=llama70b,TP=2,EXTRA="--text-only" configs/judge.sbatch
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python -m tc3_vqa.ablation.judge --run "$RUN" --model "$MODEL" --tag "$TAG" --tensor-parallel-size "${TP:-2}" ${EXTRA:-}
