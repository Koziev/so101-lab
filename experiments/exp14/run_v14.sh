#!/usr/bin/env bash
# Experiment 14: five atomic commands (approach, grasp, lift, lower, release) in simulator v14.
# Run from the repository root.
#   SETUP=setups/2026-11_12v_stereo28mm_wrist ./run_v14.sh check    # simulator v14: wrist camera, look, objects
#   SETUP=... ./run_v14.sh smoke      # minutes: 2 episodes per command, preview videos, export
#   SETUP=... ./run_v14.sh expert     # the expert alone, SCENES per command at T0, with the contact log
#   SETUP=... ./run_v14.sh record     # NUM_PER_COMMAND (100) episodes per command, WORKERS (8) shards;
#                                     # per command: NUM_PER_COMMAND=approach=2000,grasp=1500,lift=500,lower=500,release=500
#   SETUP=... ./run_v14.sh preview    # videos of 3 recorded episodes per command
#   SETUP=... ./run_v14.sh export     # LeRobot dataset (+ _eval with the held-out seeds)
#   SETUP=... ./run_v14.sh train      # SmolVLA, EPOCHS (1)
#   SETUP=... ./run_v14.sh eval       # expert (once) and the checkpoint: SCENES (30) per command at T0, T1 + held-out
# check, smoke, record and preview run in the simulator environment (PyBullet); export, train and eval in the
# LeRobot 0.6 environment (eval also needs PyBullet). Recording resumes: re-run the same command after an interruption.
set -euo pipefail
STAGE=${1:?usage: run_v14.sh check|smoke|expert|record|preview|export|train|eval [extra args for the stage script]}
shift
EXTRA=("$@")                 # e.g. ./run_v14.sh eval --n-action-steps 10 (passed on to eval_commands.py)
SETUP=${SETUP:?set SETUP=setups/<name>}
DATA=${DATA:-/media/inkoziev/corpora/tmp/lerobot_datasets}
TAG=${TAG:-v14a}
RAW=${RAW:-$DATA/data_$TAG/raw}
DATASET=${DATASET:-so101_commands_$TAG}
NUM_PER_COMMAND=${NUM_PER_COMMAND:-100}
WORKERS=${WORKERS:-8}
COMMANDS=${COMMANDS:-approach,grasp,lift,lower,release}
EPOCHS=${EPOCHS:-1}
SCENES=${SCENES:-30}
CKPT=${CKPT:-outputs/train/smolvla_commands_$TAG/checkpoints/last/pretrained_model}
case $TAG in
  v14*) ;;
  *) echo "TAG=$TAG does not look like an exp14 run (left over from an earlier experiment?)."
     echo "Run 'unset TAG RAW DATASET' or set TAG=v14..., or FORCE_TAG=1 to proceed."; [ "${FORCE_TAG:-0}" = 1 ] || exit 1 ;;
esac
echo "run_v14.sh $STAGE: TAG=$TAG RAW=$RAW DATASET=$DATASET"
# scenes: one object of the v14 set, equal colour shares; starts and holds as in v12e/v13a
REC_ARGS="--sim v14 --commands $COMMANDS --objects 1,1 --weights full=0.5,colour=0.25,shape=0.25 \
  --white-weight 0.1667 --random-start ${RANDOM_START:-0.5} --near-start ${NEAR_START:-0.12} \
  --long-hold ${LONG_HOLD:-0.4}"

case $STAGE in
  check)
    python tools/check_sim_v14.py --setup $SETUP --out outputs/check_v14 ;;
  smoke)
    rm -rf $DATA/smoke_v14
    python dataset/record_commands.py --setup $SETUP --out $DATA/smoke_v14/raw --num-per-command 2 --workers 1 \
        $REC_ARGS
    python dataset/preview_raw.py --raw $DATA/smoke_v14/raw --per-command 2 --out outputs/smoke_v14/preview ;;
  record)
    python dataset/record_commands.py --setup $SETUP --out $RAW --num-per-command $NUM_PER_COMMAND \
        --workers $WORKERS $REC_ARGS ;;
  preview)
    python dataset/preview_raw.py --raw $RAW --per-command 3 --out outputs/preview_$TAG ;;
  export)
    python dataset/export_lerobot.py --raw $RAW --root $DATA/$DATASET --repo-id local/$DATASET ;;
  train)
    DATASET=$DATASET EPOCHS=$EPOCHS OUT=outputs/train/smolvla_commands_$TAG training/train_approach.sh $TAG ;;
  expert)
    python evaluation/eval_commands.py --setup $SETUP --policy expert --commands $COMMANDS --scenes $SCENES \
        --splits T0 --results outputs/contacts_expert_$TAG/results.json ;;
  eval)
    [ -f outputs/eval_expert_$TAG/results.json ] || \
      python evaluation/eval_commands.py --setup $SETUP --policy expert --commands $COMMANDS --scenes $SCENES \
          --results outputs/eval_expert_$TAG/results.json
    NAS=${N_EXEC:-50}
    for ((i=0; i<${#EXTRA[@]}; i++)); do      # --n-action-steps given on the command line names the output too
      [ "${EXTRA[$i]}" = "--n-action-steps" ] && NAS=${EXTRA[$((i+1))]}
    done
    OUT=outputs/eval_$TAG; [ "$NAS" = "50" ] || OUT=${OUT}_nas$NAS
    echo "eval: n_action_steps=$NAS -> $OUT"
    python evaluation/eval_commands.py --setup $SETUP --policy $CKPT --commands $COMMANDS --scenes $SCENES \
        --n-action-steps $NAS --held-out $DATA/$DATASET/eval_seeds.json \
        --results $OUT/results.json --videos $OUT/videos --n-videos 3 ${EXTRA[@]+"${EXTRA[@]}"} ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
