#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=08:00:00
# Any vLLM step that serves a 72B-class model at tensor-parallel 4, e.g.
#   sbatch --export=ALL,SCRIPT=tccc_vqa.frame_select.score_frames configs/vllm_4gpu.sbatch
#   sbatch --export=ALL,SCRIPT=tccc_vqa.perception.perceive_concepts configs/vllm_4gpu.sbatch
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
python -m "$SCRIPT" --tensor-parallel-size 4 ${EXTRA:-}
