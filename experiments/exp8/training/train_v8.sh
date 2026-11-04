#!/usr/bin/env bash
# Fine-tune SmolVLA on v8a and v8b (run from the repository root). ~16 h per model on an RTX 3090.
# Usage: training/train_v8.sh a    or    training/train_v8.sh b
set -euo pipefail
V=${1:?usage: train_v8.sh a|b}
DATASETS=${DATASETS:-datasets}

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/so101_sim_grasp_v8$V \
  --dataset.root=$DATASETS/so101_sim_grasp_v8$V \
  --output_dir=outputs/train/smolvla_sim_v8$V \
  --job_name=smolvla_sim_v8$V \
  --rename_map='{"observation.images.left": "observation.images.camera1", "observation.images.right": "observation.images.camera2"}' \
  --policy.empty_cameras=1 \
  --batch_size=64 --steps=40000 --save_freq=5000 --num_workers=8 \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false
