#!/usr/bin/env bash
# exp11 localization with the new tasks (full / colour / shape / superlative). Run from the exp11 folder.
#   SETUP=setups/2026-11_12v_stereo28mm_wrist bash run_v11.sh
# TAG names the run (data_<TAG>, outputs/qwen35_08b_<TAG>, outputs/eval_<TAG>_sim.json). The first run ("v11":
# synthetic scene, labels at mid-height, objects 12-24 cm) is kept; the default now is "v11b": the setup's real
# background picture, the probed board height, grasp point 2.0 cm above the board, objects 12-30 cm.
set -euo pipefail
SETUP=${SETUP:?set SETUP to the setup folder}
TAG=${TAG:-v11b}
# IMAGE: what the model sees (common.IMAGE_MODES): full = whole view at 640x360 (v11b); crop2x / crop15 = the
# workspace window at 2x / 1.5x the pixel density. Example: IMAGE=crop2x TAG=v11c2x bash run_v11.sh
IMAGE=${IMAGE:-full}
BATCH=${BATCH:-8}; ACCUM=${ACCUM:-2}      # effective batch 16; larger images: BATCH=2 ACCUM=8
[ -f assets_so101/so101_sim.urdf ] || python simulator/prepare_assets.py --out assets_so101
# 1) smoke test: a few scenes, look at data_<TAG>_smoke/examples/*.png before the full run
python dataset/generate.py --setup "$SETUP" --out data_${TAG}_smoke --train 60 --test 20 --workers 2 --image "$IMAGE"
# 2) the dataset
python dataset/generate.py --setup "$SETUP" --out data_${TAG} --train ${TRAIN:-20000} --test 300 --train-scale 1.5 --workers ${WORKERS:-6} --image "$IMAGE"
# 3) Qwen3.5-0.8B (model environment)
python training/train.py --data data_${TAG} --out outputs/qwen35_08b_${TAG} --epochs 2 --batch $BATCH --grad-accum $ACCUM
python evaluation/evaluate.py --model outputs/qwen35_08b_${TAG}/final --data data_${TAG} --out outputs/eval_${TAG}_sim.json
