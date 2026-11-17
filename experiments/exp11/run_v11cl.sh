#!/usr/bin/env bash
# exp11, second stage: closed-loop control (stereo pair + current servo angles -> change of the angles + gripper).
#   SETUP=setups/2026-11_12v_stereo28mm_wrist bash run_v11cl.sh
# Run from the exp11 folder. TAG names the run (data_<TAG>, outputs/qwen35_08b_<TAG>, outputs/eval_<TAG>_sim.json).
# INIT is the model the training starts from: the one-shot model of the same cell if it exists (it already
# locates the objects), otherwise the base model.
set -euo pipefail
SETUP=${SETUP:?set SETUP to the setup folder}
TAG=${TAG:-v11cl2}        # v11cl = version 1 of the labels (joint-space target), kept for comparison
INIT=${INIT:-outputs/qwen35_08b_v11b/final}
[ -d "$INIT" ] || INIT=Qwen/Qwen3.5-0.8B
[ -f assets_so101/so101_sim.urdf ] || python simulator/prepare_assets.py --out assets_so101
# 1) smoke test: a few scenes (look at data_<TAG>_smoke/examples/*.png and *.json), and the label-making expert
#    run through the control loop with the servo model - it must close (nearly) every episode
python dataset/generate_cl.py --setup "$SETUP" --out data_${TAG}_smoke --train 60 --test 20 --workers 2
python evaluation/evaluate_cl.py --policy expert --data data_${TAG}_smoke --episodes 60 --out outputs/eval_${TAG}_expert.json
# 2) the dataset: TRAIN scenes x 2 arm poses
python dataset/generate_cl.py --setup "$SETUP" --out data_${TAG} --train ${TRAIN:-12000} --states 2 --test 200 --train-scale 1.5 --workers ${WORKERS:-6}
# 3) training (model environment), then single steps and whole approaches in the simulator
echo "training starts from $INIT"
python training/train.py --data data_${TAG} --model "$INIT" --out outputs/qwen35_08b_${TAG} --epochs 2 --batch 8 --grad-accum 2
python evaluation/evaluate_cl.py --model outputs/qwen35_08b_${TAG}/final --data data_${TAG} --out outputs/eval_${TAG}_sim.json
