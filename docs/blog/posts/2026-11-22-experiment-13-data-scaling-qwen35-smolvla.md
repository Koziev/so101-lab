---
date: 2026-11-22
categories:
  - Project
  - Lab notes
tags:
  - SO-101
  - SmolVLA
  - Qwen3.5
  - PyBullet
  - synthetic data
  - imitation learning
  - robotics
---
# Experiment 13: doubling the training data, and Qwen3.5-0.8B as a VLA backbone

## Goals

Experiment 13 keeps the task, the simulator, the expert and the evaluation of experiment 12 ("Approach the object": the closed gripper stops with its tip 1.5 cm from the named object, pointing at its centre). It asks two questions:

1. **Data scaling.** Does SmolVLA improve when the training set is doubled from 2,000 to 4,000 episodes, with the same number of epochs?
2. **Backbone.** How does Qwen3.5-0.8B, fine-tuned as a policy on the same 4,000 episodes, compare with SmolVLA?

All results are from the simulator. Nothing was run on the real arm.

## Data

2,000 new episodes were recorded with the v12e settings and added to the 2,000 of experiment 12:
- one object per scene, six shapes (no cones), 15 % white objects;
- start poses: 50 % random poses of the taught workspace, 12 % near the goal, 38 % the rest pose;
- 40 % of the episodes hold the goal pose for 2.0–2.4 s.

The recorder continued from each shard's next seed, so no episode repeats. About 5 % of the episodes are held out by a hash of the seed; the held-out set contains all 105 held-out seeds of experiment 12 plus 107 new ones.

| Dataset | Content |
|---|---|
| [inkoziev/so101_approach_v13a](https://huggingface.co/datasets/inkoziev/so101_approach_v13a) | Training split, LeRobotDataset v3 |
| [inkoziev/so101_approach_v13a_eval](https://huggingface.co/datasets/inkoziev/so101_approach_v13a_eval) | Held-out episodes; `eval_seeds.json` lists their seeds and start poses |

## Models

### SmolVLA (v13a)

Unchanged from v12e3: fine-tuned from `lerobot/smolvla_base`, batch 32, 3 epochs, 50-action chunks. The only difference is the dataset. Training took about 2.5 h on the RTX 5080.

### Qwen3.5-0.8B (v13a-qwen)

The model keeps the text interface of experiments 10 and 11. It sees the stereo pair (640 × 360 per eye), the instruction and the six servo readings, and answers in JSON. One answer plans the next 50 control steps, like a SmolVLA chunk, as 11 waypoints 5 steps apart: each waypoint is the change of the five joint angles from the current readings, followed by one gripper command.

```
{"waypoints":[[0.1,0.3,-0.1,0.0,0.0],[0.7,2.3,-1.9,-0.3,0.3],...,[21.3,63.7,-58.5,-9.9,9.3]],"gripper":1}
```

- **Labels:** the expert's commands at t, t+5, …, t+50, sag correction included; past the end of an episode the last command is repeated.
- **Controller:** the waypoints are interpolated linearly into one command per step, and all 50 are executed before the next call. On synthetic expert motions, interpolation and rounding cost at most 0.3° (1.2° on fast moves).
- **Training:** full fine-tuning in bf16, loss on the answer only, effective batch 16, learning rate 1e-5 with a cosine schedule. 100,000 samples (one step of one episode each), drawn uniformly from the training episodes, one pass. Training took about 4 h.

The training budgets differ: SmolVLA saw every training frame three times, the Qwen model saw a sample of the frames once. The Qwen samples are also much more expensive (about 500 image tokens and a 250-token answer each).

## Evaluation

The same harness and scenes as in experiment 12: 100 new scenes per split, of which 2–4 per split are dropped because a lying cylinder rolled away during the episode, plus the held-out recording seeds. Every model executes whole 50-step chunks.
- **T0:** calibrated camera, start from rest.
- **T1:** randomised camera, start from rest.
- **T1w:** randomised camera, start from a random workspace pose.
- **held-out:** recording seeds excluded from training, started as recorded.

An episode is successful if the final tip is within 1 cm of the goal across the pointing direction, 0.5–2.5 cm from the object's surface along that direction, the gripper axis is within 15° of the goal direction, the jaws are closed, and there has been no contact with an object or, for the gripper, with the board.

<video src="/so101-lab/assets/exp13/T1w_920000000_success.recoded.mp4" controls width="640"></video>

Video: [T1w_920000000_success.recoded.mp4](/so101-lab/assets/exp13/T1w_920000000_success.recoded.mp4). SmolVLA v13a, split T1w, "point the gripper at the purple pyramid", starting from a random workspace pose. The arm stops 0.8 cm off the goal across the pointing direction, 1.6 cm from the surface, with an angle error of 1.3°.

## Results

| Split | Expert | SmolVLA v12e3 (2,000 ep.) | SmolVLA v13a (4,000 ep.) | Qwen3.5-0.8B v13a |
|---|---|---|---|---|
| T0 | 88/98, 0.06 cm | 35/98, 1.25 cm | **48/98, 0.93 cm** | 3/98, 4.07 cm |
| T1 | 88/98, 0.06 cm | 28/98, 1.40 cm | **35/98, 1.17 cm** | 2/98, 4.31 cm |
| T1w | 76/96, 0.06 cm | 21/98, 1.51 cm | **28/98, 1.36 cm** | 0/98, 11.45 cm |
| held-out, exp12 seeds | – | 32/105, 1.46 cm | **38/105, 1.17 cm** | 3/105, 6.10 cm |
| held-out, all seeds | – | – | 68/212, 1.26 cm | 5/212, 6.19 cm |

Success count and median error across the pointing direction.

### Data scaling (SmolVLA)

On the same scenes, v13a ended closer to the goal than v12e3 in 57–63 episodes per split, by 0.11–0.23 cm at the median. The 95 % bootstrap intervals exclude zero for T0 and T1; for T1w and the held-out seeds they just reach zero. Both models came to rest in more than 90 % of episodes. Their collision counts were similar (13 and 15 episodes across the three new-scene splits); most collisions were the gripper touching the board after workspace starts. The expert also had frequent contacts in these scenes.

At T0 the median error fell below the 1 cm success limit for the first time. Across experiments 12 and 13, the T0 median error went 1.35 cm (2,000 episodes, 1 epoch) → 1.25 cm (3 epochs) → 0.93 cm (4,000 episodes, 3 epochs). Most remaining failures are near misses: 1–2 cm across.

With 10 executed steps per chunk instead of 50, the same checkpoint closed the last 2–3 mm towards the surface (median gap 1.5–1.6 cm against 1.6–1.8 cm), but only 59–69 % of the episodes ended at rest and collisions doubled. The table uses 50 steps, the protocol of experiment 12.

### Qwen3.5-0.8B as backbone

The Qwen model produced valid answers: all 1,518 answers parsed. It came to rest in 98, 97 and 87 of 98 episodes in the three new-scene splits, respectively. However, it rarely reached the goal:
- **Precision:** from rest, the median error across is 4.1–4.3 cm, and only 7–11 of 98 episodes end within 2 cm. It was closer than SmolVLA in 3–13 episodes per split.
- **Workspace starts:** the median error is 11.4 cm. 38 of 98 episodes end with the gripper touching the board, against 8 for SmolVLA.
- **Near-goal starts** (26 held-out episodes): median 2.0 cm. Short corrections work better than long approaches.
- **Height and angle:** the median gap to the surface is 0.8–1.3 cm against the 1.5 cm goal (22–24 of 98 within the allowed band), and the median angle error is 6–7° from rest.
- **Systematic, not random:** the errors of the same scene at T0 and T1 correlate at 0.85, so the model misses each scene in a consistent way. The error does not grow with the object's distance from the base (4.0–4.8 cm in every 5 cm band) and does not depend on the prompt type.
- **Speed:** one answer took 6.3 s (median), against 0.26 s for a SmolVLA chunk. The simulation waits for the policy, so the latency does not affect these results, but it rules out real-time use on the real arm in this form.

The cause was not isolated in this experiment. Three differences are candidates:
- **Training budget:** 100,000 samples once, against three passes over every frame for SmolVLA.
- **Label form:** the goal pose is never stated in the answer, only the next 50 steps relative to the current pose. From the rest pose, 50 steps do not reach the goal, so the precise target appears in the labels only for late frames.
- **Loss:** every digit of the 55 numbers weighs the same in the text loss, and the digits that set the final position are a small share.

For context, in experiment 11 the same Qwen backbone predicted the goal joint angles in a one-shot localization setup and reached about 0.6 cm median error in simulation after 40,000 samples. That was a different output formulation and is not a directly comparable policy benchmark.

## Results in brief

- Doubling the SmolVLA training data from 2,000 to 4,000 episodes, at 3 epochs, improved every split. The largest gain was on the calibrated camera: success 36 % → 49 %, median error 1.25 → 0.93 cm. On the randomised camera, workspace starts and held-out seeds, success rose by 6–7 points.
- Qwen3.5-0.8B with a waypoint text answer, trained on 100,000 samples of the same data, reached 0–3 % success and 4–11 cm median error. It learned the output format and to stop, but not to localize the goal precisely.

### Negative results

- **Qwen3.5-0.8B as a chunked policy** performed far below SmolVLA: 3/98 successes at T0 against 48/98, and 0/98 from workspace starts. It needed 1.6 times SmolVLA's training time and 24 times its inference time per chunk.

## Open questions and next steps

- **Isolating the Qwen gap:**
  - an open-loop check of the predicted waypoints against the labels on held-out frames, especially the first chunk from rest;
  - a run where the answer also states the goal angles (as in the exp11 one-shot answer), followed by the waypoints;
  - a larger sample budget.
- **Data scaling beyond 4,000 episodes:** recording takes about 80 s per episode per worker, so 8,000 episodes would need about 22 h with 8 workers.
- **Simulator appearance** (planned for experiment 14): the simulated arm looks too bright in the renders. The lighting and the arm colour should be matched to the real cell and randomised.
- **Experiment 14:** the next atomic commands: grab, lift, lower, release.

## Files

| Path | Content |
|---|---|
| Codebase | [ `experiments/exp13/` ](https://github.com/Koziev/so101-lab/tree/main/experiments/exp13) — simulator, dataset generation, training and evaluation code |
| `run_v12.sh` | Recording, export and SmolVLA training of v13a (`NUM=4000`, `EPOCHS=3`) |
| `qwen_waypoints.py` | Qwen answer format: prompt, labels, parsing, interpolation |
| `dataset/prepare_qwen.py` | Raw episodes → Qwen training samples; same held-out split as the LeRobot export |
| `training/train_qwen_approach.py` | Qwen3.5-0.8B fine-tuning |
| `evaluation/eval_approach.py` | Closed-loop evaluation; runs SmolVLA and Qwen checkpoints |
| `run_v13q.sh` | Qwen stages: smoke, prepare, train, eval |
| Datasets | [so101_approach_v13a](https://huggingface.co/datasets/inkoziev/so101_approach_v13a), [so101_approach_v13a_eval](https://huggingface.co/datasets/inkoziev/so101_approach_v13a_eval) |