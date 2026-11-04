#!/usr/bin/env bash
# Closed-loop evaluation matrix of experiment v8 (run from the repository root).
# 187 scenes per run (87 held-out dataset seeds + 100 new); five runs in parallel took ~20 h.
set -euo pipefail
DATASETS=${DATASETS:-datasets}
EVS=$DATASETS/so101_sim_grasp_v8a/eval_seeds.json
CKPT=${CKPT:-040000}
mkdir -p outputs

run () {   # name, model (a/b), camera split, extra flags
  python evaluation/eval_policy.py --policy outputs/train/smolvla_sim_v8$2/checkpoints/$CKPT/pretrained_model \
    --scene-preset v8 --camera-split $3 --width 640 --height 360 \
    --eval-seeds $EVS --new-scenes 100 --n-action-steps 10 $4 \
    --results outputs/eval_$1.json > outputs/eval_$1.log 2>&1 &
}
run v8a_train      a train
run v8a_unseen     a unseen
run v8b_train      b train  "--camera-desc"
run v8b_unseen     b unseen "--camera-desc"
run v8b_unseen_err b unseen "--camera-desc --desc-noise 2,3"
wait
