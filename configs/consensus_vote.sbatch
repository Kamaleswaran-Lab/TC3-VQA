#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
# One recognition voter for the consensus check.
#   sbatch --gres=gpu:2 --export=ALL,MODEL=OpenGVLab/InternVL3-78B,TAG=internvl3_78b,TP=2 configs/consensus_vote.sbatch
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
python -m tc3_vqa.consensus.consensus_perceive --model "$MODEL" --tag "$TAG" --tensor-parallel-size "${TP:-1}" ${EXTRA:-}
