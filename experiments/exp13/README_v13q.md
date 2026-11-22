# Experiment 13b: Qwen3.5-0.8B as an "Approach the object" policy

The same 4,000 episodes as the SmolVLA run v13a (`data_v13a/raw`), the same held-out seeds, the same evaluation
scenes and splits. Run in the exp11 Qwen environment (transformers + PyBullet).

## Answer format (`qwen_waypoints.py`)

One call plans the next 50 control steps, like one SmolVLA chunk. The answer gives 11 waypoints, the servo
command every 5 steps at offsets 0, 5, …, 50, as the change of the five joint angles from the readings in the prompt,
plus one gripper command:

```
{"waypoints":[[0.1,0.3,-0.1,0.0,0.0],[0.7,2.3,-1.9,-0.3,0.3],...,[21.3,63.7,-58.5,-9.9,9.3]],"gripper":1}
```

- **Prompt:** the instruction and the six readings, one decimal.
- **Labels:** the expert's recorded commands, sag correction included. Past the end of an episode the last command
  is repeated, so the model also learns to hold still.
- **Controller:** interpolates the waypoints linearly into one command per step. On a synthetic min-jerk move,
  interpolation and rounding together are at most 0.3° off for a 2 s move and 1.2° for a fast 1 s move.

## Files

| File | Change |
|---|---|
| `qwen_waypoints.py` | New: prompt, labels, answer text, parsing, interpolation |
| `dataset/prepare_qwen.py` | New: raw episodes → `train.jsonl` / `eval.jsonl`; same seed-hash split as the LeRobot export |
| `training/train_qwen_approach.py` | New: full fine-tuning, exp10/11 recipe (bf16, answer-only loss, batch 4 × 4, lr 1e-5, cosine) |
| `evaluation/eval_approach.py` | `QwenWaypointPolicy`, chosen automatically for a checkpoint with `qwen_approach.json` |
| `run_v13q.sh` | Stages smoke, prepare, train, eval |

## Running

```bash
export SETUP=setups/2026-11_12v_stereo28mm_wrist
export DATA=/media/inkoziev/corpora/tmp/lerobot_datasets
export CUDA_VISIBLE_DEVICES=0

./run_v13q.sh smoke        # minutes: 64 samples, 10 steps, 2 scenes; checks every stage
./run_v13q.sh prepare      # TRAIN_SAMPLES=100000 by default
./run_v13q.sh train        # EPOCHS=1
./run_v13q.sh eval         # T0, T1, T1w + held-out, 50 steps per chunk -> outputs/eval_qwen_v13a/
```

## Budget

Training length = `TRAIN_SAMPLES` × `EPOCHS`; the samples are drawn uniformly from all steps of the training
episodes. 100,000 samples is about a fifth of the frames once. Speed estimate from exp11 (about 7 samples/s at
1,250 tokens per sample): about 3–4 h for 100,000 samples of about 900 tokens. The training script prints the
tokens per sample; the Trainer's time estimate after the first logging steps gives the real speed.

Evaluation: each chunk is a generation of about 250 tokens, an estimated 3–5 s, three chunks per 150-step
episode. The simulation waits for the model, so the latency does not change the results, only the run time.
