#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
# Frame selection for one video (job array, one task per line of VIDEO_LIST).
set -euo pipefail
VIDEO=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$VIDEO_LIST")
python -m tc3_vqa.frame_select.select_frames --video "$TC3_VQA_VIDEOS/$VIDEO.mp4" --video-id "$VIDEO" --out-dir "$TC3_VQA_WORK/frames"
