#!/usr/bin/env bash
# Generate the v8 raw episodes and export the two LeRobot datasets (run from the repository root).
# Generation of 2000 episodes takes ~12-14 h with 6 workers (CPU renderer).
set -euo pipefail
RAW=${RAW:-data/raw_v8}
DATASETS=${DATASETS:-datasets}
WORKERS=${WORKERS:-6}

python dataset/record_episodes.py --out "$RAW" --scene-preset v8 --width 640 --height 360 \
  --num 2000 --workers "$WORKERS" --p-perturb 0.5 --early-lift --image-format jpg --no-wrist
python dataset/repair_raw.py --raw "$RAW"          # must report "all complete" for every shard

# model A: stereo images + 6 joint values
python dataset/export_lerobot.py --raw "$RAW" --root "$DATASETS/so101_sim_grasp_v8a" \
  --repo-id local/so101_sim_grasp_v8a --cameras left,right --overwrite
# model B: same episodes, state = 6 joint values + 10-number camera description
python dataset/export_lerobot.py --raw "$RAW" --root "$DATASETS/so101_sim_grasp_v8b" \
  --repo-id local/so101_sim_grasp_v8b --cameras left,right --append-camera --overwrite
