# Experiment 12: SmolVLA, the "Approach the object" command

The pipeline fine-tunes SmolVLA from `lerobot/smolvla_base` on synthetic trajectories of a single atomic command. Everything else comes from exp11: the calibrated cell, the real-background renderer, the painted jaw tips, the v8 servo model and the prompts built from tasks.

## The command

With the gripper closed, the arm moves from the rest pose until its tip is 1.5 cm from the target object's surface and points at the object's centre. It then stops and holds that pose. All the definitions are in `approach.py`:

| Item | Value |
|---|---|
| Tip | The closed gripper's farthest point along the fingers, between the two jaws |
| Pointing direction | The steepest one reachable: vertical, then tilted 15°, 30° or 45° (same rule as the exp10/11 labels) |
| Goal | Tip at object centre − direction × (support distance of the object along the direction + 1.5 cm) |
| Expert | Rest pose (jittered by 2°) → pre-approach pose 5 cm further back along the direction (via the ready pose if the direct path comes within 1.2 cm of the board or of another object) → goal → hold for 0.8–1.2 s. Min-jerk motion; transit at 55–75°/s on the largest joint; descent in 0.6–0.9 s. |
| Sag | During the descent and the hold, the command gets an integral correction from the joint readings (gain 0.3, at most 8°). The labels are the corrected commands. |
| Success | After `--max-steps`, all of the following hold: tip within 1 cm of the goal across the direction; 0.5–2.5 cm from the surface along it; axis within 15°; jaws closed; no contact of the arm with an object, and none of the gripper with the board. |
| Prompts | "approach the red cube", "move the gripper above …", "point the gripper at …", superlative forms, etc. The scenes and the referring expressions come from exp11's `tasks.py`, which now takes custom verbs. |

Scenes have 2–4 objects, and objects lie 12–30 cm from the pan axis. The camera is randomised at scale 1 in training. 15 % of the episodes use ideal servos; the rest use the v8 dead band and delay.

## Files

```
approach.py                    command definition, goal pose, expert, metrics
dataset/record_approach.py     raw episodes (sharded, resumable): left/right jpg 640x360, data.npz, meta.json
dataset/preview_raw.py         mp4 of raw episodes with prompt, phase, command vs reading
dataset/export_lerobot.py      raw -> LeRobotDataset 0.6 (+ held-out split by seed hash)
training/train_approach.sh     lerobot-train, smolvla_base, left->camera1, right->camera2, empty_cameras=1
evaluation/eval_approach.py    closed loop in the simulator: expert or checkpoint, T0/T1/held-out, outcomes
run_v12.sh                     smoke | record | export | train | eval
```

## Running

```bash
export SETUP=setups/2026-11_12v_stereo28mm_wrist
export DATA=/media/inkoziev/corpora/tmp/lerobot_datasets
export CUDA_VISIBLE_DEVICES=0

./run_v12.sh smoke      # ~10 min: expert check, 8 episodes, export, 60 training steps, 3 eval scenes
./run_v12.sh record     # 600 episodes (NUM=600, WORKERS=8)
./run_v12.sh export
./run_v12.sh train      # TAG=v12a
./run_v12.sh eval       # expert baseline + checkpoint, 100 scenes at T0 and T1 + held-out seeds
```

Before recording, check the smoke output. `smoke_v12/expert/*.mp4` and `smoke_v12/preview/*.mp4` should show the arm leaving the rest pose, passing above the other objects, and stopping just above the target with the tip pointing at its centre. The expert run should reach about 100 %. A low value means the goal or metric definitions are wrong, not the policy.

## One-hour budget

The defaults are batch 32 × 7,000 steps, which is 224k samples, or about 3.7 passes over 600 episodes of about 100 frames. This rests on exp9's speed: 64 × 6,000 steps took 2.5 h on an RTX 3090, about 42 samples/s. I assumed an RTX 5080 is about 1.4 times faster.

That assumption needs checking. After about 300 steps, read `updt_s` and `data_s` in the log, then restart with:

```
STEPS = 3600 / (updt_s + data_s)          # e.g. 0.45 s/step -> STEPS=8000
```

- If `data_s` is a large share of each step, video decoding is the bottleneck: raise `WORKERS`.
- On out-of-memory, use `BATCH=24` with proportionally more steps.
- To keep the number of passes near 4 at a different step count, scale `NUM` with it: about 85 episodes per 1,000 steps at batch 32.

## Video decoding

The scripts decode the videos with PyAV by default (`VIDEO_BACKEND=pyav` for training, `--video-backend` for the export check), because torchcodec did not load in env_lerobot06 (torch 2.15). Once torchcodec loads, `VIDEO_BACKEND=torchcodec` makes data loading faster.

## Not verified here

This workspace has no PyBullet, so none of the simulator code was run. Only the expert's control flow (planning, phases, sag correction) was tested, on a toy arm.

On the first real run, check:
- **The tip definition.** It comes from the `JawTips` probe. Check that the goal looks right in the preview.
- **The rest pose.** It is `REST_REAL` plus the setup offsets. If the real rest pose differs, pass `--rest a,b,c,d,e`, in real degrees.
- **Episode length.** About 100 frames is an estimate; `record` prints the actual frames per episode.
- **Missed episodes.** The share of episodes the expert misses because of sag. They are printed as "expert missed" and not kept.

## v12c: random starts, one epoch

The v12a/v12b trajectories all leave from the rest pose and are nearly deterministic, so the joint angles
partway along a trajectory already give away where it ends. v12c removes that link:

- **Start pose:** half of the episodes (`--random-start 0.5`) start from a random pose of the setup's taught safe
  workspace (`workspace.json`, `calib/workspace.py`; real readings + joint offsets). The pose is drawn between
  two nearby taught poses (`Workspace.sample`). It is kept only if, before any physics step:
  - both jaws of the closed gripper are at least 5 cm from the target object (`START_MIN_DIST`);
  - every link of the arm is at least 1 cm from every object;
  - the tip is above the board.

  After settling under the servo model, the arm must touch nothing. Up to 30 draws are made, otherwise the
  seed is skipped. Which seeds start from the workspace is decided from a separate random stream of the seed,
  so the evaluation reproduces the start of each held-out seed.
- **Expert:** the same from any start: transit to the pre-approach pose, either directly or via the ready
  pose, with the same path check (the tip may not go lower than where it started), then descend and hold.
- **Recorder:** episodes in which the target moves more than 1 cm (for example a lying cylinder rolling away)
  are no longer kept, because their labels would point where the object was.
- **Training:** `EPOCHS=1` sets the step count to ceil(frames / batch) from the exported dataset, i.e. one pass.
- **Evaluation:** the `T1w` split is T1 started from random workspace poses. The held-out split starts each seed
  as it was recorded (`--random-start`). `tools/inspect_policy.py closed --start workspace` does the same for
  inspection.

```bash
export TAG=v12c RAW=/media/inkoziev/corpora/tmp/lerobot_datasets/data_v12c/raw DATASET=so101_approach_v12c OBJECTS=1,1 \
       WEIGHTS=full=0.5,colour=0.25,shape=0.25 RANDOM_START=0.5 EPOCHS=1
./run_v12.sh record      # 600 episodes
./run_v12.sh export
VIDEO_BACKEND=torchcodec ./run_v12.sh train
./run_v12.sh inspect
./run_v12.sh eval        # splits T0, T1 (from rest), T1w (from the workspace), held-out
```

## v12d: near-goal starts and long holds

v12c never came to rest at the end of an episode. Two changes target that:

- **Near-goal starts** (`--near-start 0.12`). The arm starts close to the goal: the tip is moved 0.5–3 cm
  across the pointing direction and between 0.7 cm closer and 3 cm further back along it, so it ends up
  0.8–4.5 cm from the surface. The gripper axis is tilted up to 8°. The pose must be clear of every object by
  3 mm and keep the tip above the board. From there the expert makes one short correction (0.5–0.9 s) straight
  to the goal, then holds. These episodes show the step the policy is missing in closed loop: near the
  object but off, make a small correction, then stop.
- **Long holds** (`--long-hold 0.4`). 40 % of the episodes hold the goal for 2.0–2.4 s (60–72 steps)
  instead of 0.8–1.2 s. Their labels contain complete 50-step chunks with no movement. A normal hold ends
  before a chunk does, and LeRobot masks the rest of the chunk as padding.

Start shares: 50 % workspace (the same seeds as in v12c), 12 % near the goal, 38 % rest. Evaluation splits
are unchanged (T0, T1, T1w); `T1n` evaluates near-goal starts and `inspect_policy.py closed --start near`
inspects them.

## v12e: more data, no cones, mostly coloured objects

The setup is that of v12d, with three changes:

- **2,000 episodes** (instead of 600), still one training epoch.
- **No cones** (`--shapes cube,ball,cylinder,hex_prism,dodecahedron,pyramid`).
- **15 % white objects** (`--white-weight 0.15`, was 0.5). Most of the real printed bodies are painted now.

`./run_v12e.sh` runs record → export → train → inspect → eval, then **compare**: the v12d checkpoint
evaluated on the v12e scenes. The new-scene seeds are the same as before, but scenes without cones and
with other colours are different scenes, so v12d's earlier numbers can't be compared with v12e's directly.
Recording takes about 5.5 h with 8 workers.
