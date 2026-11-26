---
date: 2026-11-26
categories:
  - Project
  - Lab notes
tags:
  - SO-101
  - SmolVLA
  - PyBullet
  - imitation learning
  - multi-task robotics
  - vision-language-action
---
# Experiment 14: five atomic commands — approach, grasp, lift, lower, release

## Goals

Experiments 12 and 13 trained SmolVLA on one atomic command: approach the named object. Experiment 14 extends the task to a set of five commands that together make a pick-and-place:

1. **approach** the named object (as in experiments 12–13);
2. **grasp** it from the approach pose;
3. **lift** the gripper, with or without the object;
4. **lower** the gripper, with or without the object;
5. **release** the object.

One policy is trained on all five commands. At inference, the requested command is specified by a natural-language instruction rather than a separate action ID; the policy also receives the stereo images and six servo readings described below. The experiment tests whether SmolVLA can infer which command to execute and how accurately it carries it out.

The simulator was also revised to match the real cell more closely: the wrist camera on the gripper, the arm's appearance, and a new set of larger objects that the gripper can hold.

All results are from the simulator. Nothing was run on the real arm.

Code: [experiments/exp14](https://github.com/Koziev/so101-lab/tree/main/experiments/exp14).

## Simulator v14

### Wrist camera

The real arm carries a camera module on the gripper, which hides part of the fixed jaw from the stereo camera. Simulator v14 adds it to the arm as two links on the gripper: the printed mount (white) and the camera board with its lens (black). Both are built from the STL files of the real parts and placed by the hand-eye calibration of the real wrist camera. They are rendered in the stereo images and count as part of the gripper in every contact test. The wrist camera's own image is not used in this experiment.

### Arm appearance

In experiment 13 the simulated arm looked brighter and whiter than the real one. Simulator v14 matches the brightness to images of the real cell and randomises it per episode:
- arm parts grey level 145–185 instead of near-white;
- a slight cyan tint (0.90, 1.0, 1.0) ± 2 % over the whole foreground;
- an overall gain of 0.85–1.15;
- the right eye 4–14 % brighter than the left, as in the real stereo camera, plus ± 4 % per eye.

### Objects

The earlier experiments used a broader set of shapes, including several the gripper cannot hold (balls, pyramids and lying cylinders). Version 14 uses three shapes that stand upright and fit between the jaws:

| Shape | Size |
|---|---|
| cube | edge 28–36 mm |
| cylinder (upright) | diameter 25–35 mm, height 30–35 mm |
| hexagonal prism (upright) | height 25–35 mm, 28–35 mm across corners |

Six colours, matched to the real objects: yellow, blue, pink (also called magenta in the prompts), white, green and red, in equal shares. Each scene has one object.

## Commands and the expert

### Chains

Every command except approach starts where the previous command left the arm and the object. An episode records one command. The commands that lead to its start are played first by the same expert without rendering, so the recorded command begins in a realistic state:

| Command | Variants (share) | Commands played before it |
|---|---|---|
| approach | from start (100 %) | — |
| grasp | after approach (100 %) | approach |
| lift | holding (70 %) | approach, grasp |
| | empty, gripper open (30 %) | approach, grasp, lift, lower, release |
| lower | holding (70 %) | approach, grasp, lift |
| | empty, gripper open (30 %) | approach, grasp, lift, lower, release, lift |
| release | low, after a lower (70 %) | approach, grasp, lift, lower |
| | lifted (30 %) | approach, grasp, lift |

The expert keeps one nominal joint path and one sag correction over the whole chain, so consecutive commands join without jumps. Start poses are as in experiments 12–13: the rest pose, random poses of the taught workspace (50 %), or near the goal (12 % of the approaches).

### Trajectories

- **Approach.** As in experiments 12–13: transit to a pre-approach pose 5 cm back from the goal, then a descent until the closed gripper's tip is 1.5 cm from the object, pointing at its centre. New in v14: for cubes and hexagonal prisms, the wrist roll turns the jaws onto a pair of parallel faces, so that the grasp that follows closes on flat faces.
- **Grasp**, from the approach pose:
  - *shift*: the gripper moves sideways until the fixed jaw's gripping face is 4–7 mm outside the object;
  - *open*: to 55–85 %, at least wide enough for the object;
  - *lower*: along the gripper axis until the fixed jaw's tip is 6–10 mm above the board;
  - *close*: the gripper closes, and stops 3 steps after the jaws are blocked by the object.
- **Lift:** the tip rises vertically to 10 cm above the board, the gripper as it is.
- **Lower:** holding, until the object's lowest point is 4 mm above the board; empty, until the tip is 1.25 cm above the board.
- **Release:** the gripper opens to 55–85 % and the arm does not move.

Every command ends with a hold of 0.8–1.2 s, or 2.0–2.4 s in 40 % of the episodes, so that the policy learns to stand still once the command is done.

Instructions: approach uses the scene prompt of experiments 12–13 ("go to the blue hex prism"); grasp names the object ("take hold of the yellow cube"); lift, lower and release use either a plain gripper command ("go up", "lower the gripper", "let it go") or, when the object is held, its name half of the time ("lift the red cylinder").

### Expert development

The first expert (v14a) succeeded at approach in 75 % and at grasp in 64 % of the checks, all failures being contacts. A contact log per link and phase traced the causes, and four rounds of fixes brought the expert to approach 27/28, grasp 25/27, and lift, lower and release 20/20 each, on the calibrated camera. The fixes:

- **Contact test.** PyBullet's contact list also contains surfaces that are close but apart (positive contact distance). Half of the failures were such near misses. A contact now counts only with a negative distance.
- **Whole-gripper clearance.** Heights and distances were planned with the tip point only. The expert now measures the real collision shapes of the arm at the planned poses: the grasp height keeps the gripper 3 mm above the board, the approach keeps every link 3 mm from the object, and a scene whose object touches the folded arm is not used.
- **Sag correction.** The integral correction of servo sag also integrated the lag of the servos while moving, and carried the arm several millimetres past the goal at the end of every descent. It now compares the readings with the planned pose of 4 steps earlier while the arm moves, and with the planned pose once it is still.
- **Command-specific margins:**
  - the grasp shift is moved back along the gripper axis when it would pass within 4 mm of the object;
  - an empty lower steps 3–12 mm sideways when the released object stands right below the open jaws;
  - a lower aims at the measured position of the arm rather than the planned one, and stops with the gripper 4 mm above the board.

The recorder keeps an episode only if every command of its chain succeeds.

## Data

Three datasets were recorded with the same simulator and expert settings:

| Run | Episodes per command (approach / grasp / lift / lower / release) | Total | Expert |
|---|---|---|---|
| v14a | 100 / 100 / 100 / 100 / 100 | 500 | first version |
| v14b | 100 / 100 / 100 / 100 / 100 | 500 | final version |
| **v14c** | **4,000 / 2,000 / 500 / 500 / 500** | **7,500** | final version |

v14c gives approach as many episodes as experiment 13 (4,000), and grasp, which depends on approach precision, twice the share of the remaining commands. The recorder interleaves the commands in a fixed 15-seed cycle in proportion to these numbers, so all commands progress together.

About 5 % of the episodes are held out by a hash of the seed.

| Dataset | Content |
|---|---|
| [inkoziev/so101_commands_v14c](https://huggingface.co/datasets/inkoziev/so101_commands_v14c) | Training split, LeRobotDataset v3; `eval_seeds.json` lists the held-out seeds and their commands |
| [inkoziev/so101_commands_v14c_eval](https://huggingface.co/datasets/inkoziev/so101_commands_v14c_eval) | Held-out episodes |

## Model

SmolVLA, fine-tuned from `lerobot/smolvla_base` with LeRobot 0.6, as in experiments 12–13: the stereo pair (640 × 360 per eye) as two cameras, the six servo readings, the instruction, 50-action chunks.

- **v14c:** batch 32, 3 epochs over the v14c training split, about 9.5 h on the RTX 5080.
- v14a and v14b: 1 epoch over 500 episodes each.

Model: [inkoziev/smolvla_so101_commands_v14c](https://huggingface.co/inkoziev/smolvla_so101_commands_v14c).

## Evaluation

Every command is evaluated on 30 new scenes per split, plus the held-out recording seeds. A scene is prepared exactly as in the recorder: the start pose, then the chain of commands before the evaluated one, played by the expert. The policy then gets the instruction and runs for a fixed number of steps (approach 150, grasp 160, lift 120, lower 140, release 75), executing whole 50-action chunks.

- **T0:** calibrated camera.
- **T1:** camera randomised as in training.
- **held-out:** recording seeds excluded from training, randomised camera.

Each command is evaluated separately, not as part of a complete five-command rollout by the learned policy. To prepare a test, the expert executes the preceding chain; for example, a grasp starts from the expert's approach pose, not from an approach generated by the policy. A few scenes per command are skipped because the expert's chain fails before the evaluated command starts.

**Evaluation scope:** the reported rates measure individual command execution from expert-prepared starting states. They do not measure the success rate of an end-to-end pick-and-place sequence controlled entirely by the learned policy.

Success criteria, with no contact of the gripper or the wrist camera with the board in any command:

| Command | Success |
|---|---|
| approach | as in experiments 12–13: tip within 1 cm of the goal across the pointing direction, 0.5–2.5 cm from the surface, axis within 15°, closed jaws, no contact with the object |
| grasp | object held by both jaws, tilted at most 25° |
| lift | tip 10 ± 1.5 cm above the board; holding: object still held; empty: gripper still open |
| lower | holding: object held, its lowest point 0–12 mm above the board; empty: tip 0.7–2.0 cm above the board, gripper open |
| release | gripper at least 30 % open, object no longer held, tip moved at most 1 cm |

### Videos

SmolVLA v14c executing each command in the simulator.

<video src="/so101-lab/assets/exp14/exp14c_approach_action.mp4" controls width="640"></video>

Approach: [exp14c_approach_action.mp4](/so101-lab/assets/exp14/exp14c_approach_action.mp4)

<video src="/so101-lab/assets/exp14/exp14c_grasp_action.mp4" controls width="640"></video>

Grasp: [exp14c_grasp_action.mp4](/so101-lab/assets/exp14/exp14c_grasp_action.mp4)

<video src="/so101-lab/assets/exp14/exp14c_lift_action.mp4" controls width="640"></video>

Lift: [exp14c_lift_action.mp4](/so101-lab/assets/exp14/exp14c_lift_action.mp4)

<video src="/so101-lab/assets/exp14/exp14c_lower_action.mp4" controls width="640"></video>

Lower: [exp14c_lower_action.mp4](/so101-lab/assets/exp14/exp14c_lower_action.mp4)

<video src="/so101-lab/assets/exp14/exp14c_release_action.mp4" controls width="640"></video>

Release: [exp14c_release_action.mp4](/so101-lab/assets/exp14/exp14c_release_action.mp4)

## Results

### SmolVLA v14c by split

| Command | T0 | T1 | held-out | All |
|---|---|---|---|---|
| approach | 9/28 | 8/29 | 76/202 | **93/259 (36 %)** |
| grasp | 24/27 | 25/28 | 81/91 | **130/146 (89 %)** |
| lift | 19/20 | 19/22 | 24/25 | **62/67 (93 %)** |
| lower | 18/20 | 17/22 | 22/27 | **57/69 (83 %)** |
| release | 19/20 | 20/22 | 20/23 | **59/65 (91 %)** |

The held-out split accounts for 368 of the 606 evaluated command episodes. Its per-command success rates are broadly similar to the pooled T0/T1 results, with some variation: for example, approach success is 76/202 on held-out seeds versus 17/57 on T0 and T1 combined.

### Training data: v14a, v14b, v14c

| Command | v14a (500 ep., first expert) | v14b (500 ep.) | v14c (7,500 ep.) |
|---|---|---|---|
| approach | 1/61 (2 %) | 2/63 (3 %) | 93/259 (36 %) |
| grasp | 13/46 (28 %) | 25/60 (42 %) | 130/146 (89 %) |
| lift | 24/26 (92 %) | 29/45 (64 %) | 62/67 (93 %) |
| lower | 14/26 (54 %) | 30/46 (65 %) | 57/69 (83 %) |
| release | 9/23 (39 %) | 9/49 (18 %) | 59/65 (91 %) |
| episodes ending at rest | 5/182 | 5/263 | 567/606 (94 %) |

All splits pooled. v14b has more lift, lower and release episodes than v14a because more of the expert's chains succeed. The v14b-to-v14c comparison changes both dataset size (500 to 7,500 episodes) and training duration (1 to 3 epochs), so it does not isolate the effect of data volume alone.

**Command-specific motion** was already emerging with 500 episodes. In v14b, grasp, lift, lower and release produced the expected kind of motion in 82–100 % of episodes per command. For approach, the arm moved towards the object in 73 % of episodes; most remaining cases instead rose from the folded start. What v14a and v14b lacked was accurate termination: the arm kept moving after the command, overshot lifts by up to 3.7 cm, drifted by 1–3 cm while opening and pressed held objects into the board. With 15 times as many episodes and 3 epochs, v14c ends 94 % of evaluated episodes at rest, and:
- **lift:** median final height 10.3 cm (target 10 cm), 63 of 67 within ± 1.5 cm;
- **release:** median movement while opening 0.55 cm, 59 of 65 within 1 cm;
- **empty lower:** median final tip height 1.03 cm (target 1.25 cm);
- **held lower:** still too deep: the object ends a median 0.9 mm above the board, against 4 mm planned. Most episodes pass within the 0–12 mm band, and the 7 failures press the object into the board.

### Approach

The median error across the pointing direction is 1.17 cm at T0, 1.37 cm at T1 and 1.29 cm on held-out seeds (v14b: 4.3 cm). 157 of 259 approaches end within 1.5 cm and about 211 within 2 cm. Only approaches within 1.0 cm satisfy the cross-track part of the success criterion; overall success also requires the correct surface gap and angle, closed jaws, and no prohibited contacts. The height is generally right (median gap 1.3–1.4 cm, against the 1.5 cm target), and no approach touched the object or board. About 28 failures point the gripper 10–17° off the goal direction: the expert tilts the gripper by 0° or 15° depending on clearance around the object, and these episodes may have selected the other tilt.

By start pose: 15 of 23 approaches from near the goal succeed, 36 of 92 from the rest pose, 42 of 144 from a random workspace pose.

Comparison with SmolVLA v13a, trained on the same number of approach episodes (4,000, 3 epochs):

| | v13a (exp13) | v14c |
|---|---|---|
| T0 | 48/98, 0.93 cm | 9/28, 1.17 cm |
| T1 | 35/98, 1.17 cm | 8/29, 1.37 cm |
| held-out | 68/212, 1.26 cm | 76/202, 1.29 cm |

Success and median error across the pointing direction. The conditions differ: simulator v14 (wrist camera, darker arm, three new shapes in new colours), roll alignment of the goal pose, a policy that shares its capacity with four other commands, and fewer evaluation scenes (28–29 per split). On the held-out seeds, the two models are at a similar level.

### Grasp

Of the 16 failed grasps, 8 close on nothing, 5 knock the object over and 4 hold the object but touch the board. These failure categories overlap and are not mutually exclusive.

### Chunk length

With 10 executed actions per chunk instead of 50, the same checkpoint did worse on every command: approach 70/259, grasp 100/146, lift 51/67, lower 53/69, release 18/65, and 470 instead of 567 episodes ending at rest. The choice of motion did not change. Each new chunk is predicted from scratch and does not continue the previous one exactly, so frequent re-planning adds small jumps: release drifts a median 1.29 cm while opening, and grasps knock the object over three times as often. All results above use 50 actions per chunk.

## Results in brief

- One SmolVLA policy, trained on 7,500 simulated episodes covering five atomic commands, succeeds on grasp, lift, lower and release in 83–93 % of evaluated episodes, while approach remains harder at 36 %. The policy generally produces command-appropriate motion, but command recognition alone does not guarantee accurate completion.
- With 500 episodes, the policy often produced the intended motion pattern but rarely stopped at the right place. The v14c setup (7,500 episodes and 3 training epochs) raised the share of episodes ending at rest to 94 %, compared with about 2 % for v14a/v14b; this comparison changes both data volume and training duration.
- Approach precision is close to the 1 cm limit (median 1.2–1.4 cm) and at the level of experiment 13 on the held-out seeds.
- Most of the work went into the expert: the contact test, clearance of the whole gripper rather than the tip, and the servo sag correction.

### Negative results

- **v14a and v14b** (500 episodes, 1 epoch): approach 2–3 %, and the policy almost never came to rest.
- **Re-planning every 10 actions** lowered every command's success rate, release from 91 % to 28 %.
- **Held lower** ends consistently deeper than the expert, by about 3 mm.

## Open questions and next steps

- **Approach precision:** whether the error across has a direction bias; more approach data; scoring the angle against the nearer of the expert's two tilts.
- **Temporal ensembling** of overlapping chunks, as a way to close the loop more often without the jumps between chunks.
- **Command sequences:** running the five commands one after another with the policy, instead of starting each from the expert's state.
- **Real arm:** the commands on the real cell, with the real wrist camera.

## Files

| Path | Content |
|---|---|
| `commands.py` | The five commands: chains, phases, expert (`ChainExpert`), success criteria, instructions |
| `approach.py` | Approach expert, clearance probe of the collision shapes, sag correction |
| `simulator/so101_sim/wrist_camera.py` | Wrist camera mount and board from the STL files |
| `simulator/so101_sim/objects_v11.py` | Object sets v11 and v14 |
| `simulator/so101_sim/scene_real.py` | Arm appearance and per-eye brightness (`sample_look`) |
| `dataset/record_commands.py` | Recorder; per-command episode counts |
| `dataset/export_lerobot.py` | Raw episodes → LeRobotDataset, held-out split |
| `evaluation/eval_commands.py` | Closed-loop evaluation, contact and clearance logs |
| `run_v14.sh` | Stages: check, smoke, expert, record, preview, export, train, eval |
| `README_v14.md` | Details of every change |
| Datasets | [so101_commands_v14c](https://huggingface.co/datasets/inkoziev/so101_commands_v14c), [so101_commands_v14c_eval](https://huggingface.co/datasets/inkoziev/so101_commands_v14c_eval) |
| Model | [smolvla_so101_commands_v14c](https://huggingface.co/inkoziev/smolvla_so101_commands_v14c) |