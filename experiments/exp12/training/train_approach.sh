#!/usr/bin/env bash
# Fine-tune SmolVLA (from lerobot/smolvla_base) on the approach dataset. Run from the repository root, in the
# LeRobot 0.6 environment:
#     DATA=/media/inkoziev/corpora/tmp/lerobot_datasets CUDA_VISIBLE_DEVICES=0 training/train_approach.sh v12a
# EPOCHS=1: one pass over the dataset (the step count is computed from its frame count); otherwise STEPS.
# Budget: about one hour on an RTX 5080 (16 GB). Defaults: batch 32 x 7000 steps = 224k samples (~3.7 passes
# over 600 episodes of ~100 frames). The speed is not known in advance: after ~300 steps read "updt_s" and
# "data_s" in the log and restart with STEPS = 3600 / (updt_s + data_s) if the estimate is far off, e.g.
#     STEPS=9000 training/train_approach.sh v12a
# If data_s is not small compared with updt_s, video decoding is the bottleneck: raise WORKERS.
# Video decoding: VIDEO_BACKEND=pyav (default) or torchcodec.
# Out of memory: BATCH=24 (and STEPS up by the same factor).
set -euo pipefail
TAG=${1:?usage: train_approach.sh TAG   (e.g. v12a)}
DATA=${DATA:-/media/inkoziev/corpora/tmp/lerobot_datasets}
DATASET=${DATASET:-so101_approach_v12}
BATCH=${BATCH:-32}
EPOCHS=${EPOCHS:-}          # set (e.g. EPOCHS=1): STEPS = ceil(EPOCHS * frames / BATCH), from the dataset
if [ -n "$EPOCHS" ]; then
  INFO=$DATA/$DATASET/meta/info.json
  FRAMES=$(python -c "import json; print(json.load(open('$INFO'))['total_frames'])")
  STEPS=$(python -c "import math; print(math.ceil($EPOCHS * $FRAMES / $BATCH))")
  echo "dataset $DATASET: $FRAMES frames -> $EPOCHS epoch(s) at batch $BATCH = $STEPS steps"
fi
STEPS=${STEPS:-7000}
SAVE=${SAVE:-$(( STEPS / 2 ))}
WORKERS=${WORKERS:-8}
VIDEO_BACKEND=${VIDEO_BACKEND:-pyav}   # torchcodec is faster once it loads in the environment
OUT=${OUT:-outputs/train/smolvla_approach_$TAG}

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/$DATASET \
  --dataset.root=$DATA/$DATASET \
  --dataset.video_backend=$VIDEO_BACKEND \
  --output_dir=$OUT \
  --job_name=smolvla_approach_$TAG \
  --rename_map='{"observation.images.left": "observation.images.camera1", "observation.images.right": "observation.images.camera2"}' \
  --policy.empty_cameras=1 \
  --batch_size=$BATCH --steps=$STEPS --save_freq=$SAVE --log_freq=100 --num_workers=$WORKERS \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false
