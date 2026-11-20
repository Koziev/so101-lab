---
date: 2026-11-20
categories:
  - Project
  - Lab notes
tags:
  - SO-101
  - SmolVLA
  - PyBullet
  - synthetic data
  - imitation learning
  - robotics
---
# Experiment 12: SmolVLA on an atomic "Approach the object" command, trained on synthetic trajectories

## Concept

Experiments 10 and 11 used Qwen3.5-0.8B to predict a grasp pose (one-shot) or small joint increments (closed loop) for "grab and lift" instructions. Experiment 12 returned to SmolVLA and narrowed the task to one atomic command:

**"Approach the object":** the closed gripper moves from its current pose to a point above the named object. It ends with its tip 1.5 cm from the object's surface and its axis pointing at the object's centre, then holds still.

Grasping, lifting and releasing were left for later experiments, and so was the planner that will chain the commands. All training data were synthetic: trajectories of a scripted expert, recorded in the calibrated simulator cell of experiment 11. All evaluations in this entry were run in the same simulator, on new scenes. Nothing in this experiment was tested on the real arm.

The experiment ran as a sequence of variants (v12a–v12e3). Each variant changed one or two things in the data or the training, based on the analysis of the previous one.

## Setup

- **Cell and simulator:** unchanged from experiment 11. The same setup folder (`setups/2026-11_12v_stereo28mm_wrist`) supplies the stereo calibration, joint offsets, board height and taught workspace. The PyBullet scene is composited onto the real background picture of the cell, the jaw tips are painted black as on the real arm, and servo sag is simulated with the v8 actuator model (gravity-dependent dead band and delay; ideal servos in 15 % of the episodes). Images are 640 × 360 per eye at 30 Hz.
- **Camera randomisation:** training uses the v9 camera randomisation at scale 1. Evaluation uses scale 0 (calibrated camera, "T0") and scale 1 ("T1").
- **Model:** SmolVLA from `lerobot/smolvla_base` (LeRobot 0.6), fine-tuned with the default settings. Only the action expert is trained; the vision encoder and the language model stay frozen.
  - The two stereo images are mapped to the model's `camera1` and `camera2` inputs, and the third camera input is left empty.
  - State and action are 6 values: five joint angles in degrees and the gripper (0–100).
  - The model predicts chunks of 50 actions.
- **Training hardware:** RTX 5080 (16 GB). Batch 32 used 12–13 GB of GPU memory. One inference (one 50-action chunk) took about 240 ms.

## The command and the expert

### Goal pose

The "tip" is the end of the longer jaw of the closed gripper, at the gripping face, i.e. between the two closed jaws. The pointing direction is the steepest one the arm can reach: vertical, or else tilted by 15°, 30° or 45° with the fingers pointing away from the base, the same rule as for the grasp labels of experiments 10 and 11. The goal tip position lies on the line through the object's centre along this direction, 1.5 cm outside the object's surface.

### Expert trajectory

The expert plans in joint space and runs in the physics simulation:

1. **Transit:** minimum-jerk motion from the start pose to a pre-approach pose 5 cm further back along the pointing direction. The largest joint moves at 55–75°/s. If the tip would come within 1.2 cm of the board or of another object on the direct path, the motion goes via a fixed ready pose instead.
2. **Descent:** minimum-jerk motion to the goal pose in 0.6–0.9 s.
3. **Hold:** the goal pose is held for 0.8–1.2 s (from v12d, 2.0–2.4 s in 40 % of the episodes).

During the descent and the hold, the command is corrected from the joint readings with an integral term (gain 0.3 per step, at most 8°). This lets the arm reach the goal despite servo sag. The training labels are the corrected commands.

An episode is kept only if the expert succeeds, as defined below. From v12c on, an episode is also discarded if the target moved by more than 1 cm during it; lying cylinders can roll away in the simulator.

### Success criterion

An episode is judged at its last step (150 steps, 5 s). It is a success if all of the following hold:
- the tip is within 1 cm of the goal across the pointing direction;
- the tip is 0.5–2.5 cm from the object's surface along the pointing direction;
- the gripper axis is within 15° of the goal direction;
- the jaws are closed;
- no part of the arm touched an object, and the gripper did not touch the board, during the episode.

Failures are classified as collision, wrong object, off target (more than 1 cm across), too low, too high, or bad angle. The evaluation also records whether the arm is at rest at the end: every joint moving less than 0.5° per step over the last 10 steps.

### Prompts

The prompts are built from the referring expressions of experiment 11 ("the red cube", "a white piece", "the cylinder") with approach verbs: "approach …", "move to …", "go to …", "move the gripper above …", "point the gripper at …", "bring the gripper to …". Superlative prompts ("find the biggest of the cubes and approach it") were used only in v12a, which had several objects per scene.

## Pipeline

| Stage | Tool | Notes |
|---|---|---|
| Recording | `dataset/record_approach.py` | Parallel shards, resumable; per episode: JPEG frames, `data.npz` (state, action, phase, tip position, contacts), `meta.json` |
| Export | `dataset/export_lerobot.py` | LeRobotDataset v3 with videos; held-out split (5 %) chosen by a hash of the seed |
| Training | `training/train_approach.sh` | `lerobot-train` from `smolvla_base`; step count from `EPOCHS` and the dataset's frame count |
| Evaluation | `evaluation/eval_approach.py` | Closed loop in the simulator; 100 new scenes per split; held-out recording seeds; expert baseline through the same harness |
| Inspection | `tools/inspect_policy.py` | Open-loop comparison of predicted chunks with labels; closed-loop videos and joint plots |

Recording took about 80 s per episode per worker process, mostly rendering. With 8 workers, 600 episodes took about 1.7 h and 2,000 episodes about 5.5 h.

The evaluation renders only the frames the policy uses: those at the start of each new chunk, plus every frame for saved videos. Rendering every step had made the first evaluations about ten times slower.

Evaluation splits:
- **T0:** calibrated camera, start from the rest pose.
- **T1:** randomised camera as in training, start from the rest pose.
- **T1w:** as T1, but starting from a random pose of the taught workspace.
- **held-out:** recording seeds excluded from training, started as recorded.

## Variants

| Variant | Episodes | Scenes | Start poses | Hold | Training |
|---|---|---|---|---|---|
| v12a | 600 | 2–4 objects, all 7 shapes, 50 % white, all prompt types | rest | 0.8–1.2 s | batch 32 × 7,000 steps (~3.7 epochs, 57 min) |
| v12b | 600 | 1 object, all 7 shapes, 50 % white, no superlatives | rest | 0.8–1.2 s | as v12a |
| v12c | 600 | as v12b | 50 % random workspace pose, 50 % rest | 0.8–1.2 s | 1 epoch |
| v12d | 600 | as v12b | 50 % workspace, 12 % near the goal, 38 % rest | 40 % long (2.0–2.4 s) | 1 epoch |
| v12e | 2,000 | 1 object, no cones, 15 % white | as v12d | as v12d | 1 epoch (~1 h) |
| v12e3 | as v12e | as v12e | as v12d | as v12d | 3 epochs on the v12e dataset |

The workspace starts are drawn between two nearby poses of the taught safe workspace. Both jaws must be at least 5 cm from the target, and every link at least 1 cm from every object.

The near-goal starts place the tip 0.5–3 cm off the goal sideways and 0.8–4.5 cm from the surface, with the axis tilted up to 8°. From there the expert makes one short correction (0.5–0.9 s) straight to the goal.

## Results

<video src="/so101-lab/assets/exp12/exp12e_approach_eval_left_eye.recoded.mp4" controls width="640"></video>

Video: [exp12e_approach_eval_left_eye.recoded.mp4](/so101-lab/assets/exp12/exp12e_approach_eval_left_eye.recoded.mp4). Closed-loop evaluation episodes in the simulator, seen by the left eye of the stereo camera: the policy brings the closed gripper from its start pose to the goal above the named object.

All numbers are from the simulator. "Across" is the tip's final distance from the goal across the pointing direction. A few scenes per split (3–4) had a lying cylinder that rolled away during the episode. They count as failures in the tables below; the cause is described under "Errors".

### Overview

Success rate and median error across, per split. Unless marked, the policy executed 10 of each 50-action chunk before predicting the next one.

| Model | Scenes | T0 | T1 | T1w | held-out |
|---|---|---|---|---|---|
| Expert | v12a | 77/83, 0.06 cm | 74/82, 0.06 cm | – | – |
| v12a | v12a | 0/94, 7.64 cm | – | – | – |
| v12b | v12b | 10/99, 2.63 cm | 6/99, 2.94 cm | – | 1/22, 1.76 cm |
| v12c | v12b–d | 7/99, 2.59 cm | 8/99, 2.64 cm | 6/99, 2.86 cm | 2/25, 2.81 cm |
| v12d | v12b–d | 10/99, 2.24 cm | 13/99, 2.30 cm | 9/99, 2.74 cm | 1/26, 2.94 cm |
| Expert | v12e | 88/98, 0.06 cm | 88/98, 0.06 cm | 76/96, 0.06 cm | – |
| v12d | v12e | 7/98, 2.44 cm | 14/98, 2.10 cm | 11/98, 2.35 cm | – |
| v12e | v12e | 18/98, 1.74 cm | 19/98, 1.87 cm | 12/98, 2.09 cm | 17/105, 2.04 cm |
| v12e, 50 steps | v12e | 25/98, 1.35 cm | 23/98, 1.66 cm | 20/98, 1.78 cm | 17/105, 1.72 cm |
| v12e3, 50 steps | v12e | 35/98, 1.25 cm | 28/98, 1.40 cm | 21/98, 1.51 cm | 32/105, 1.46 cm |

The evaluation scenes changed with the training distribution:
- v12a: several objects;
- v12b–d: one object, all shapes, 50 % white;
- v12e: no cones, 15 % white.

Results are directly comparable only within the same scene set. For that reason the v12d checkpoint was also evaluated on the v12e scenes.

### v12a: several objects, rest starts only

v12a succeeded in none of 94 T0 episodes. In 50 episodes it ended closest to a different object than the one named, and in 34 it was off target; the median error was 7.6 cm. Only 7 of 94 episodes ended with the arm at rest.

The inspection tool separated the causes with an open-loop test on recorded episodes. Starting from the first frame, the model predicted a 50-action chunk, and its end was compared with the expert's command at step 50:
- **Training episodes:** the predicted shoulder-pan angle was 2.1° off on average.
- **Held-out episodes:** it was 22.1° off on average, between 15° and 43° in five of six episodes. That is the angular spacing of neighbouring objects.

Errors averaged over whole episodes were small in both cases (pan 0.7° on training, 1.5° held-out). Once the arm moves, its joint angles already tell where it is heading, and continuing the motion is enough to predict the next 10 steps. The choice of object, made from the images at the rest pose, did not generalise beyond the 570 training scenes: the model had memorised them.

### v12b: one object per scene

Removing the choice between objects reduced the median error from 7.6 to 2.6–2.9 cm. The model directed the arm towards the object, but the final position was imprecise. The held-out recording seeds were no easier than new scenes (median 1.76 cm, 1 success in 22). So the remaining error was not a matter of memorised scenes.

### v12c: random start poses

With every training trajectory starting from the rest pose, the joint angles at any intermediate step nearly determine the end point, independent of the images and the prompt. In v12c, half of the episodes started from a random pose of the taught workspace, and training was reduced to one epoch.

The result was similar to v12b (median 2.5–2.9 cm) with about a quarter of the gradient steps. No episode ended at rest, however; v12b had 13–17 of 99.

### v12d: near-goal starts and long holds

v12d added short corrections from poses near the goal (12 % of episodes) and holds longer than one chunk (40 %). On the same scenes, v12d ended closer than v12c in 56–63 of 99 episodes, by about 0.2 cm at the median. Still no episode ended at rest.

The errors of v12c and v12d on the same scenes correlated at 0.83–0.88. Both models missed in nearly the same way on each scene.

### v12e: 3.3 times more data, no cones, fewer white objects

On the v12e scenes, v12e beat v12d:
- It ended closer in 69 of 95 scenes at T0, 58 at T1 and 62 at T1w (rolled-target scenes excluded).
- The median improvement was 0.48 cm at T0 (95 % bootstrap interval 0.27–0.74 cm), 0.24 cm at T1 and 0.47 cm at T1w.
- Large misses disappeared: no episode from rest ended more than 5 cm off, against 7 for v12d.

The training data differed in both size and composition (v12d also saw cones and 50 % white objects), so the gain mixes the two effects. The error correlation between the two models on the same scenes fell to 0.35–0.55. This suggests that much of the scene-specific error of the earlier models came from too little data, not from a limit of what the images show.

By object colour, no consistent difference was found.

### Executed actions per chunk

The same v12e checkpoint was evaluated executing 10, 25 or 50 of the 50 predicted actions before predicting the next chunk:

| Executed per chunk | T0 success | T0 median across | T1 success | T1 median across | At rest (T0 + T1) |
|---|---|---|---|---|---|
| 10 | 18/98 | 1.74 cm | 19/98 | 1.87 cm | 0/196 |
| 25 | 27/98 | 1.38 cm | 18/98 | 1.68 cm | 1/196 |
| 50 | 25/98 | 1.35 cm | 23/98 | 1.66 cm | 18/196 |

Against 10 steps, 50 steps lowered the paired median error by 0.25 cm at T0 and 0.35 cm at T1; both bootstrap intervals exclude zero. 25 and 50 steps did not differ in position. With full chunks the policy looks at the images only three times per 150-step episode, and the result did not get worse. The policy plans the approach largely from its first observations. Re-planning every 10 steps added jitter at the chunk boundaries rather than corrections.

### v12e3: three epochs on the same data

Training three epochs on the v12e dataset gave the best model of the experiment (50 executed steps, all splits):

| Split | v12e success | v12e3 success | v12e median across | v12e3 median across | v12e at rest | v12e3 at rest |
|---|---|---|---|---|---|---|
| T0 | 25/98 | 35/98 | 1.35 cm | 1.25 cm | 9/98 | 91/98 |
| T1 | 23/98 | 28/98 | 1.66 cm | 1.40 cm | 9/98 | 91/98 |
| T1w | 20/98 | 21/98 | 1.78 cm | 1.51 cm | 11/98 | 92/98 |
| held-out | 17/105 | 32/105 | 1.72 cm | 1.46 cm | 8/105 | 97/105 |

- **Stopping:** the largest change. With three epochs, more than 90 % of the episodes ended with the arm at rest; with one epoch it was about 10 %. The near-goal starts and long holds introduced in v12d needed more training than one epoch to take effect.
- **Precision:** on the same scenes, v12e3 ended closer than v12e in 60–73 of 95–105 episodes per split. The paired median improvement was 0.15–0.25 cm, with all four bootstrap intervals excluding zero.
- **Angle:** the median angle error halved (3.4–4.7° to 1.7–2.3°), and the share of heights within the allowed band rose from 79–93 to 88–95.
- **Generalisation:** held-out recording seeds and new scenes performed about the same (30 % and 29–37 % success from rest). The model does not overfit at three epochs.

Remaining failures of v12e3 (T0, T1 and T1w together, rolled-target scenes excluded):
- **Mostly position:** 183 of 201 failures were off target. In 75 of them the error across was between 1.0 and 1.5 cm, in another 49 between 1.5 and 2.0 cm.
- **Collisions:** 14, nine of them the gripper touching the board after a start from the workspace.
- **Harder cases:**
  - tilted approaches (median 1.50 cm, against 1.30 cm vertical);
  - objects farther than 21 cm from the pan axis (median 1.56–1.59 cm, against 1.25–1.28 cm closer in).
- **Shape:** dodecahedra and pyramids had the lowest success rates, 9 of 36 and 9 of 51.

### The expert as a ceiling

The expert reached the goal pose to within 0.06 cm (median) in every episode, but succeeded in only 89 % of rest starts and 78 % of workspace starts. All its failures were contacts:
- with objects, 10 per split;
- with the board, after starts from low workspace poses, 10 of 96.

Its path check tests only the tip, so other links occasionally graze the target or the board. Episodes with contacts were dropped from training, so the hardest scenes are slightly under-represented in the data. The policies had fewer collisions than the expert from rest (1–6 per split), mainly because they end farther from the object.

## Results in brief

- SmolVLA fine-tuned on 2,000 synthetic expert episodes and trained for three epochs reaches the goal with a median error of 1.25–1.5 cm across the pointing direction. It comes to rest in more than 90 % of the episodes and succeeds by the strict 1 cm criterion in 22–37 % of them, depending on the split. The expert is at 0.06 cm and 78–89 %.
- With several objects per scene and 600 episodes, the model memorised the training scenes and did not learn to choose the named object.
- More data (600 → 2,000 episodes) and more training on the same data (1 → 3 epochs) both improved precision. The second also made the arm stop at the goal.
- Executing the full 50-action chunk worked better than re-planning every 10 or 25 steps.

### Negative results

- **v12a:** 0 % success. Choosing the named object among 2–4 did not generalise from 600 episodes; the model memorised the training scenes.
- **v12c:** random start poses alone did not improve precision over v12b.
- **v12d:** near-goal starts and long holds did not make the arm stop with one epoch of training. The effect appeared only after three epochs (v12e3).
- **Re-planning:** every 10 steps was worse than every 50, in both precision and stopping.

### Errors in the course of the experiment

- **Rolling targets:** lying cylinders can roll away in the simulator. Up to v12b, the recorder kept such episodes, whose labels point where the object was. From v12c on, they are discarded. In the evaluation they still occur (3–4 scenes per split) and count as failures.
- **torchcodec:** the installed build was for an older PyTorch and CUDA, and could not load with PyTorch 2.15 and CUDA 13.2. The LeRobot dataset could be written but not read. It was fixed by installing a torchcodec build for the current PyTorch, plus the shared FFmpeg 7 libraries.
- **Slow evaluation:** the first evaluation rendered both images at every step, although the expert never uses them and the policy only at chunk starts. That took about 3 minutes per scene; rendering only the frames that are used made it about ten times faster.
- **Missing results folder:** the evaluation script did not create the folder for its results file, so the first runs with 25 and 50 executed steps finished without saving. The script now creates it.
- **The inspection test** first drew no plots because matplotlib was not installed in the LeRobot environment. The plots are now drawn with OpenCV.

## Open questions and next steps

- **Data scaling (experiment 13):** another 4,000 episodes, merged with the 2,000 of v12e, will test how far data scaling continues to help.
- **Qwen3.5-0.8B on the merged dataset (experiment 13):** a comparison with SmolVLA on the same data and evaluation.
- **More atomic commands (experiment 14):** grab the object, lift and lower the gripper, open the gripper and release the object.
- **Real arm:** the v12e3 policy has not been run on the real arm yet. The simulator was calibrated to the real cell in experiment 11, but the size of the gap between simulated and real images and servo behaviour for this policy is unknown.
- **The expert's collisions:** the path check covers only the tip, so other links occasionally touch the target or the board. A whole-arm check would raise the expert's ceiling and stop dropping the hardest scenes from the data.
- **Training more of SmolVLA:** unfreezing the language model and vision encoder would test whether the remaining 1.25–1.5 cm error is limited by data or by the frozen visual features.

## Files

| Path | Content |
|---|---|
| `approach.py` | Command definition, goal pose, expert, start poses, success criterion |
| `dataset/record_approach.py` | Episode recorder (sharded, resumable) |
| `dataset/export_lerobot.py` | Export to LeRobotDataset, held-out split |
| `dataset/preview_raw.py` | Videos of recorded episodes |
| `training/train_approach.sh` | SmolVLA fine-tuning (`EPOCHS` or `STEPS`) |
| `evaluation/eval_approach.py` | Closed-loop evaluation in the simulator |
| `tools/inspect_policy.py` | Open-loop and closed-loop inspection |
| `run_v12.sh`, `run_v12e.sh` | Stage scripts for the variants |
| Dataset, training split | [inkoziev/so101_approach_v12e](https://huggingface.co/datasets/inkoziev/so101_approach_v12e) |
| Dataset, evaluation split | [inkoziev/so101_approach_v12e_eval](https://huggingface.co/datasets/inkoziev/so101_approach_v12e_eval) (held-out recording seeds) |
| Codebase | [`experiments/exp12/`](https://github.com/Koziev/so101-lab/tree/main/experiments/exp12) |
| Media | `docs/assets/exp12/` |