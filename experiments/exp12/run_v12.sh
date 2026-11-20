#!/usr/bin/env bash
# Experiment 12, "Approach the object": record -> export -> train -> evaluate. Run from the repository root.
#   SETUP=setups/2026-11_12v_stereo28mm_wrist ./run_v12.sh smoke     # minutes: checks every stage end to end
#   SETUP=... ./run_v12.sh record | export | train | eval | inspect  # the full run, stage by stage
# Variants via environment variables, e.g. v12b (one object per scene, no superlatives):
#   TAG=v12b RAW=$DATA/data_v12b/raw DATASET=so101_approach_v12b OBJECTS=1,1 \
#   WEIGHTS=full=0.5,colour=0.25,shape=0.25 ./run_v12.sh record
# v12c (random starts from the taught workspace in half the episodes, one epoch):
#   TAG=v12c RAW=$DATA/data_v12c/raw DATASET=so101_approach_v12c OBJECTS=1,1 WEIGHTS=full=0.5,colour=0.25,shape=0.25 \
#   RANDOM_START=0.5 EPOCHS=1 ./run_v12.sh record | export | train | inspect | eval
# v12d (as v12c + 12 % near-goal starts, 40 % long holds):
#   TAG=v12d RAW=$DATA/data_v12d/raw DATASET=so101_approach_v12d OBJECTS=1,1 WEIGHTS=full=0.5,colour=0.25,shape=0.25 \
#   RANDOM_START=0.5 NEAR_START=0.12 LONG_HOLD=0.4 EPOCHS=1 ./run_v12.sh record | export | train | inspect | eval
# record and the expert check run in the simulator environment (PyBullet); export, train and eval of the
# policy in the LeRobot 0.6 environment (it also needs PyBullet for eval).
set -euo pipefail
STAGE=${1:?usage: run_v12.sh smoke|record|export|train|eval|inspect}
SETUP=${SETUP:?set SETUP=setups/<name>}
DATA=${DATA:-/media/inkoziev/corpora/tmp/lerobot_datasets}
RAW=${RAW:-$DATA/data_v12/raw}
NUM=${NUM:-600}
WORKERS=${WORKERS:-8}
TAG=${TAG:-v12a}
DATASET=${DATASET:-so101_approach_v12}
OBJECTS=${OBJECTS:-2,4}
WEIGHTS=${WEIGHTS:-}
RANDOM_START=${RANDOM_START:-0}
NEAR_START=${NEAR_START:-0}
LONG_HOLD=${LONG_HOLD:-0}
WHITE_WEIGHT=${WHITE_WEIGHT:-0.5}
SHAPES=${SHAPES:-}
SCENE_ARGS="--objects $OBJECTS ${WEIGHTS:+--weights $WEIGHTS} --white-weight $WHITE_WEIGHT ${SHAPES:+--shapes $SHAPES}"
SPLITS=${SPLITS:-T0,T1}
[ "$RANDOM_START" != "0" ] && SPLITS=${SPLITS_W:-T0,T1,T1w}
CKPT=${CKPT:-outputs/train/smolvla_approach_$TAG/checkpoints/last/pretrained_model}

case $STAGE in
  smoke)
    python evaluation/eval_approach.py --setup $SETUP --policy expert --scenes 10 --videos outputs/smoke_v12/expert --n-videos 3
    python dataset/record_approach.py --setup $SETUP --out $DATA/smoke_v12/raw --num 8 --workers 2
    python dataset/preview_raw.py --raw $DATA/smoke_v12/raw --n 4 --out smoke_v12/preview
    python dataset/export_lerobot.py --raw $DATA/smoke_v12/raw --root $DATA/so101_approach_smoke \
        --repo-id local/so101_approach_smoke --eval-fraction 0.2 --overwrite
    DATASET=so101_approach_smoke STEPS=60 SAVE=60 OUT=outputs/train/smolvla_approach_smoke \
        training/train_approach.sh smoke
    python evaluation/eval_approach.py --setup $SETUP --scenes 3 --splits T0 --max-steps 60 \
        --policy outputs/train/smolvla_approach_smoke/checkpoints/last/pretrained_model --videos outputs/smoke_v12/policy ;;
  record)
    python dataset/record_approach.py --setup $SETUP --out $RAW --num $NUM --workers $WORKERS $SCENE_ARGS \
        --random-start $RANDOM_START --near-start $NEAR_START --long-hold $LONG_HOLD ;;
  export)
    python dataset/export_lerobot.py --raw $RAW --root $DATA/$DATASET --repo-id local/$DATASET ;;
  train)
    DATASET=$DATASET EPOCHS=${EPOCHS:-} training/train_approach.sh $TAG ;;
  eval)
    mkdir -p outputs/eval_expert_$TAG outputs/eval_$TAG
    [ -f outputs/eval_expert_$TAG/results.json ] || \
      python evaluation/eval_approach.py --setup $SETUP --policy expert --scenes 100 $SCENE_ARGS --splits $SPLITS \
          --results outputs/eval_expert_$TAG/results.json
    python evaluation/eval_approach.py --setup $SETUP --policy $CKPT --scenes 100 $SCENE_ARGS --splits $SPLITS \
        --random-start $RANDOM_START --near-start $NEAR_START \
        --held-out $DATA/$DATASET/eval_seeds.json --results outputs/eval_$TAG/results.json \
        --videos outputs/eval_$TAG/videos --n-videos 4 ;;
  inspect)
    python tools/inspect_policy.py openloop --policy $CKPT --raw $RAW --episodes 10 --out outputs/inspect_$TAG/train
    python tools/inspect_policy.py openloop --policy $CKPT --raw $RAW --episodes 10 \
        --held-out $DATA/$DATASET/eval_seeds.json --out outputs/inspect_$TAG/held_out
    python tools/inspect_policy.py closed --setup $SETUP --policy $CKPT --seeds 920000000-920000005 $SCENE_ARGS \
        --out outputs/inspect_$TAG/closed
    if [ "$RANDOM_START" != "0" ]; then
      python tools/inspect_policy.py closed --setup $SETUP --policy $CKPT --seeds 920000000-920000005 $SCENE_ARGS \
          --start workspace --camera-scale 1 --out outputs/inspect_$TAG/closed
    fi ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
