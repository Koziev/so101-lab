#!/usr/bin/env bash
# Experiment 12e, "Approach the object": as v12d with
#   - 2000 episodes (one training epoch),
#   - all shapes except cones,
#   - white targets/objects 15 % (was 50 %).
# Run from the repository root:
#   SETUP=setups/2026-11_12v_stereo28mm_wrist CUDA_VISIBLE_DEVICES=0 ./run_v12e.sh            # all stages
#   SETUP=... ./run_v12e.sh record | export | train | inspect | eval | compare                 # one stage
# "compare" evaluates the v12d checkpoint on the v12e scenes (no cones, 15 % white), so the two models can be
# compared on exactly the same scenes.
set -euo pipefail
export SETUP=${SETUP:?set SETUP=setups/<name>}
export DATA=${DATA:-/media/inkoziev/corpora/tmp/lerobot_datasets}
export TAG=v12e
export RAW=$DATA/data_v12e/raw
export DATASET=so101_approach_v12e
export NUM=${NUM:-2000}
export WORKERS=${WORKERS:-8}
export OBJECTS=1,1
export WEIGHTS=full=0.5,colour=0.25,shape=0.25
export SHAPES=cube,ball,cylinder,hex_prism,dodecahedron,pyramid
export WHITE_WEIGHT=0.15
export RANDOM_START=0.5
export NEAR_START=0.12
export LONG_HOLD=0.4
export EPOCHS=1
export VIDEO_BACKEND=${VIDEO_BACKEND:-torchcodec}

stage() { echo; echo "=== v12e: $1 ($(date '+%F %T')) ==="; ./run_v12.sh "$1"; }
compare() {
  local ckpt=outputs/train/smolvla_approach_v12d/checkpoints/last/pretrained_model
  [ -d "$ckpt" ] || { echo "no v12d checkpoint at $ckpt, skipped"; return 0; }
  mkdir -p outputs/eval_v12d_on_v12e
  python evaluation/eval_approach.py --setup $SETUP --policy $ckpt --scenes 100 --splits T0,T1,T1w \
      --objects $OBJECTS --weights $WEIGHTS --white-weight $WHITE_WEIGHT --shapes $SHAPES \
      --results outputs/eval_v12d_on_v12e/results.json
}

case ${1:-all} in
  all) for s in record export train inspect eval; do stage $s; done; echo; echo "=== compare ==="; compare ;;
  compare) compare ;;
  record|export|train|inspect|eval) stage $1 ;;
  *) echo "usage: run_v12e.sh [all|record|export|train|inspect|eval|compare]"; exit 1 ;;
esac
