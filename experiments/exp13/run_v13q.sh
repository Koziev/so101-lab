#!/usr/bin/env bash
# Experiment 13, second part: Qwen3.5-0.8B as an "Approach the object" policy, on the same 4,000 episodes as the
# SmolVLA run v13a. Run from the repository root, in the exp11 Qwen environment (transformers + PyBullet):
#   SETUP=setups/2026-11_12v_stereo28mm_wrist ./run_v13q.sh smoke      # minutes: every stage end to end
#   SETUP=... ./run_v13q.sh prepare | train | eval
# Budget: TRAIN_SAMPLES samples x EPOCHS. Check the speed in the first minutes of "train" (the Trainer's time
# estimate) and restart with another TRAIN_SAMPLES (prepare again) if it is far from what you want.
# Evaluation: the same scenes and splits as the SmolVLA v13a run (eval_v13a_nas50), whole 50-step chunks.
set -euo pipefail
STAGE=${1:?usage: run_v13q.sh smoke|prepare|train|eval}
SETUP=${SETUP:?set SETUP=setups/<name>}
DATA=${DATA:-/media/inkoziev/corpora/tmp/lerobot_datasets}
TAG=${TAG:-v13a}
RAW=${RAW:-$DATA/data_v13a/raw}
DATASET=${DATASET:-so101_approach_v13a}         # the LeRobot export: its eval_seeds.json gives the held-out seeds
QDATA=${QDATA:-$DATA/qwen_approach_$TAG}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-100000}
EPOCHS=${EPOCHS:-1}
MODEL=${MODEL:-Qwen/Qwen3.5-0.8B}
OUT=${OUT:-outputs/train/qwen_approach_$TAG}
CKPT=${CKPT:-$OUT/final}
N_EXEC=${N_EXEC:-50}
BATCH=${BATCH:-4}                 # per step; out of memory: BATCH=2 GRAD_ACCUM=8 (same effective batch 16)
GRAD_ACCUM=${GRAD_ACCUM:-$(( 16 / BATCH ))}
# scenes as in v12e / v13a
SCENE_ARGS="--objects 1,1 --weights full=0.5,colour=0.25,shape=0.25 --white-weight 0.15 \
  --shapes cube,ball,cylinder,hex_prism,dodecahedron,pyramid"
START_ARGS="--random-start 0.5 --near-start 0.12"

case $STAGE in
  smoke)
    python dataset/prepare_qwen.py --raw $RAW --out $DATA/qwen_approach_smoke --train-samples 64 --eval-samples 8
    python training/train_qwen_approach.py --data $DATA/qwen_approach_smoke --model $MODEL \
        --out outputs/train/qwen_approach_smoke --max-steps 10 --eval-steps 10 \
        --batch $BATCH --grad-accum $GRAD_ACCUM
    python evaluation/eval_approach.py --setup $SETUP --policy outputs/train/qwen_approach_smoke/final \
        --scenes 2 --splits T0 --max-steps 60 --n-action-steps $N_EXEC $SCENE_ARGS \
        --results outputs/smoke_v13q/results.json --videos outputs/smoke_v13q/videos --n-videos 2 ;;
  prepare)
    python dataset/prepare_qwen.py --raw $RAW --out $QDATA --train-samples $TRAIN_SAMPLES ;;
  train)
    python training/train_qwen_approach.py --data $QDATA --model $MODEL --out $OUT --epochs $EPOCHS \
        --batch $BATCH --grad-accum $GRAD_ACCUM ;;
  eval)
    mkdir -p outputs/eval_qwen_$TAG
    python evaluation/eval_approach.py --setup $SETUP --policy $CKPT --scenes 100 --splits T0,T1,T1w \
        $SCENE_ARGS $START_ARGS --held-out $DATA/$DATASET/eval_seeds.json --n-action-steps $N_EXEC \
        --results outputs/eval_qwen_$TAG/results.json --videos outputs/eval_qwen_$TAG/videos --n-videos 4 ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
