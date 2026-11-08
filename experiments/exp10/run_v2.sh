#!/usr/bin/env bash
# Model v2 of experiment 10: 10,000 samples around the calibrated camera pose (s = 1.5), synonyms. Run from this folder.
set -euo pipefail
CALIB=${CALIB:?set CALIB to the stereo calibration file with pose_left_eye}
[ -f assets_so101/so101_sim.urdf ] || python simulator/prepare_assets.py --out assets_so101
python dataset/generate.py --out data_v2 --train 10000 --test 200 --train-scale 1.5 --workers ${WORKERS:-6} --calib "$CALIB" --synonyms
python training/train.py --data data_v2 --out outputs/qwen35_08b_v2 --epochs 2 --batch 8 --grad-accum 2
python evaluation/evaluate.py --model outputs/qwen35_08b_v2/final --data data_v2 --out outputs/eval_v2_sim.json
