# Experiment 13: twice the data for SmolVLA, and Qwen3.5-0.8B as a policy

The code is that of experiment 12 (`README_v12.md`), with the Qwen pipeline added (`README_v13q.md`).

## SmolVLA on 4,000 episodes (v13a)

These steps add 2,000 episodes to the 2,000 of v12e (raw folder renamed to `data_v13a`), then export and train for
3 epochs. Recording resumes: `NUM` is the total, and each shard continues from its next seed. Keep `WORKERS=8`
as in v12e.

```bash
export SETUP=setups/2026-11_12v_stereo28mm_wrist
export DATA=/media/inkoziev/corpora/tmp/lerobot_datasets
export TAG=v13a RAW=$DATA/data_v13a/raw DATASET=so101_approach_v13a WORKERS=8
export OBJECTS=1,1 WEIGHTS=full=0.5,colour=0.25,shape=0.25 WHITE_WEIGHT=0.15
export SHAPES=cube,ball,cylinder,hex_prism,dodecahedron,pyramid
export RANDOM_START=0.5 NEAR_START=0.12 LONG_HOLD=0.4 VIDEO_BACKEND=torchcodec

NUM=4000 ./run_v12.sh record        # ~5.5 h for the 2,000 new episodes
./run_v12.sh export
EPOCHS=3 ./run_v12.sh train         # ~2.5 h on an RTX 5080
```

The `eval` stage of `run_v12.sh` executes 10 actions per chunk. The exp13 results use 50, so run
the evaluation directly:

```bash
python evaluation/eval_approach.py --setup $SETUP \
    --policy outputs/train/smolvla_approach_v13a/checkpoints/last/pretrained_model \
    --scenes 100 --splits T0,T1,T1w --objects $OBJECTS --weights $WEIGHTS --white-weight $WHITE_WEIGHT \
    --shapes $SHAPES --random-start $RANDOM_START --near-start $NEAR_START \
    --held-out $DATA/$DATASET/eval_seeds.json --n-action-steps 50 \
    --results outputs/eval_v13a_nas50/results.json --videos outputs/eval_v13a_nas50/videos --n-videos 4
```

## Datasets

- [inkoziev/so101_approach_v13a](https://huggingface.co/datasets/inkoziev/so101_approach_v13a): training split
- [inkoziev/so101_approach_v13a_eval](https://huggingface.co/datasets/inkoziev/so101_approach_v13a_eval): held-out
  episodes, with `eval_seeds.json`

## Qwen3.5-0.8B

See `README_v13q.md` (`run_v13q.sh`: smoke, prepare, train, eval). To reduce memory, use
`BATCH=2 ./run_v13q.sh train`; the effective batch stays 16.
