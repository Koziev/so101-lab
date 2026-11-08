#!/usr/bin/env bash
# Smoke test of experiment 10 (all stages, ~1-2 h): 3,000 training samples at s = 1, default camera centre. Run from this folder.
set -euo pipefail
[ -f assets_so101/so101_sim.urdf ] || python simulator/prepare_assets.py --out assets_so101
python dataset/generate.py --out data --train 3000 --test 200 --train-scale 1 --workers ${WORKERS:-6}
python training/train.py --data data --out outputs/qwen35_08b_full --epochs 3 --batch 8 --grad-accum 2 ${TRAIN_FLAGS:-}
python evaluation/evaluate.py --model outputs/qwen35_08b_full/final --data data --out outputs/eval_qwen35_08b_full.json
