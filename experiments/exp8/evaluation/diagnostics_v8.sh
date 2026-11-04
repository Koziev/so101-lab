#!/usr/bin/env bash
# Diagnostics of the v8 failure on 50 shared scenes (run from the repository root).
#   20k / 30k checkpoints   -> is the model still improving?
#   v8a on the v7 scene     -> what did the broad training cost? (compare with the v7 policy on the same scenes)
#   --no-actuator           -> does the servo model cause the failure? (same scenes and cameras)
set -euo pipefail
DATASETS=${DATASETS:-datasets}
EVS=$DATASETS/so101_sim_grasp_v8a/eval_seeds.json
mkdir -p outputs

diag () {   # name, checkpoint, extra flags
  python evaluation/eval_policy.py --policy outputs/train/smolvla_sim_v8a/checkpoints/$2/pretrained_model \
    --width 640 --height 360 --eval-seeds $EVS --held-out-limit 0 --new-scenes 50 --n-action-steps 10 $3 \
    --results outputs/diag_$1.json > outputs/diag_$1.log 2>&1 &
}
diag v8a_20k   020000 "--scene-preset v8 --camera-split train"
diag v8a_30k   030000 "--scene-preset v8 --camera-split train"
diag v8a_on_v7 040000 "--scene-preset v7 --texture-split eval"
diag v8a_noact 040000 "--scene-preset v8 --camera-split train --no-actuator"
wait
