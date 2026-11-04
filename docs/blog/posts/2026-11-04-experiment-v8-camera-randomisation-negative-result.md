---
date: 2026-11-04
categories:
  - Project
  - Lab notes
tags:
  - SO-101
  - SmolVLA
  - PyBullet
  - LeRobot
  - experiment-v8
---

# Experiment v8: wide camera randomisation and an explicit camera description (negative result)

## Idea

The first test of the v7 policy on the real SO-101 showed that the real camera stood outside the narrow range of camera positions used in training (closer, and looking down more steeply). Requiring the real camera to match the simulator is impractical, and it becomes impossible once the camera is moved by the robot itself (planned active-perception experiments). Experiment v8 therefore tried to make the policy independent of the exact camera placement, in two ways:

1. **Wide randomisation.** Each training episode uses a different stereo camera position, orientation and field of view, so that the policy must cope with a broad range of viewpoints.
2. **An explicit camera description.** Model B additionally receives ten numbers describing the camera (position, viewing direction, image-up direction, field of view), as it would obtain them on the real setup from a calibration. Model A receives no such information and must infer the viewpoint from the images and joint angles. Comparing A and B was meant to show whether explicit calibration is worth building.

The experiment also added a model of the real servos' behaviour, measured on the 7.4 V arm.

## Result in brief

**v8 is a clear failure.** Both models grasp successfully in only 13–20 % of scenes from their own training range of viewpoints and 9–11 % on held-out viewpoints, against 79 % for the narrowly trained v7 policy. Follow-up diagnostics show that the broad camera randomisation is the cause: the v8 policy lost the precise positioning that v7 had learned, even for viewpoints like those of v7. Training length and the servo model are not the cause. Because both models fail, the experiment cannot answer whether the explicit camera description helps.

## Setup

All v8 results are from simulation; the v8 policies were not tested on the real arm. The simulated setup is the v7 scene, modelled on the real one:

- **Arm and workspace:** SO-101 on a 50×49 cm plywood board (pan axis 7 cm from the back edge), on a desk with background clutter; textures made from photos of the real board, desk and room.
- **Camera:** a stereo pair modelled on the real side-by-side USB stereo camera (6.2 cm baseline, rectified, 640×360 per eye), the only camera; no wrist camera. The real camera, with its original 3.6 mm lenses, sees about 44° horizontally after rectification; its measured pose is 52.5 cm in front of the pan axis, 31 cm above the board, looking down 38.7°.
- **Servos:** a model of the 7.4 V follower arm. Measurements on the real arm showed that shoulder_lift and elbow_flex stop 2–4.5° short of their target when moving in one direction and arrive within about 0.5° in the other (a dead band that grows with the gravity load). The simulator reproduces this (about 8–10° per N·m of gravity torque, scaled ×0.5–1.5 per episode), adds a command delay of 0–2 control steps, and uses ideal servos in 15 % of episodes.
- **Task:** one cube (22–35 mm) or ball (24–40 mm), 12 colours, 12–24 cm from the pan axis within ±30°, instruction such as "pick up the red cube".

## Data

One set of about 2,000 generated episodes was exported twice, as two datasets with identical episodes: [so101-sim-grasp-v8a](https://huggingface.co/datasets/inkoziev/so101-sim-grasp-v8a) and [so101-sim-grasp-v8b](https://huggingface.co/datasets/inkoziev/so101-sim-grasp-v8b). Only successful demonstrations were kept (roughly 1,740), of which 87 were held out for evaluation.

- **Camera randomisation per episode:** azimuth −45…+30° around the front of the robot, 0.45–0.85 m from the pan axis, 0.15–0.60 m high, 15–55° looking down, roll ±4°, horizontal field of view 42–68° (covering the current lenses and the 2.8 mm and 2.1 mm lenses ordered for the real camera). A camera was accepted only if the robot base, the gripper's ready pose and the object area were visible in both eyes; objects were placed only where both eyes see them. Viewpoints with azimuth beyond +30° or farther than 0.85 m were excluded from training and kept for testing.
- **Demonstrations:** scripted expert as in v6/v7 (lift as soon as the grip is secure, sideways perturbations in half of the episodes). Under the servo model, the expert needed an explicit anticipation of the dead band; without it, its own success rate fell from 97 % to 87 %. With it, the expert succeeds in 97 % of v8 scenes, with or without the servo model.
- **State input:** v8a: 6 joint values; v8b: 6 joint values plus the 10-number camera description.

## Policies

Two SmolVLA models were fine-tuned from `lerobot/smolvla_base` with identical settings (40,000 steps, batch size 64, about 16 hours each on an RTX 3090): [smolvla-so101-sim-v8a](https://huggingface.co/inkoziev/smolvla-so101-sim-v8a) (model A, images and joint angles) and [smolvla-so101-sim-v8b](https://huggingface.co/inkoziev/smolvla-so101-sim-v8b) (model B, images, joint angles and camera description). Both receive the left and right images as two camera inputs and re-plan every 10 steps.

## Evaluation

Each run uses the same 187 scenes (87 held out from the dataset, 100 new); success means the object is held between both jaws and lifted at least 3 cm. "Training range" uses cameras drawn from the training distribution, "held out" uses only the excluded viewpoints. In the last run model B was given a camera description that was wrong by 2 cm in position and 3° in orientation.

| Model | Viewpoints | Success | Median distance gripper–object at closing |
|---|---|---|---|
| A | training range | 25/187 (13 %) | 4.3 cm |
| B | training range | 37/187 (20 %) | 3.7 cm |
| A | held out | 17/187 (9 %) | 5.2 cm |
| B | held out | 20/187 (11 %) | 4.9 cm |
| B, description 2 cm / 3° wrong | held out | 19/187 (10 %) | 5.0 cm |
| *v7 (narrow camera range, for reference)* | *v7 scenes* | *79 %* | *1.9 cm* |

Almost all failures are positioning errors: the arm moves towards the right object but closes 3–5 cm away from it, while grasps generally need less than 2 cm. Failures are spread evenly over camera distance, height, azimuth, pitch and field of view, and over episodes with and without the servo model. Model B is somewhat better than A within the training range (28 scenes gained, 16 lost, p ≈ 0.1), which is suggestive but not conclusive; on held-out viewpoints there is no difference, and the wrong camera description changes nothing.

Three diagnostics on 50 shared scenes separated the possible causes:

| Diagnostic | Result | Interpretation |
|---|---|---|
| Model A at 20k / 30k / 40k training steps | 7 / 7 / 5 successes, median distance 3.6–4.1 cm | no improvement after 20k steps: longer training will not help |
| Model A on v7 scenes (narrow cameras, ideal servos) vs the v7 policy | 5 vs 40 successes (35 scenes won by v7, none by v8a), 5.0 vs 1.1 cm | the broad training destroyed precision even under v7's own conditions |
| Model A with the servo model switched off | 2 vs 5 successes, 5.1 vs 4.1 cm | the servo model is not the cause |

## Conclusions

With about 1,650 training episodes and a policy that sees a single frame at a time, spreading the training over this range of camera positions and fields of view left too few examples per viewpoint to learn precise positioning anywhere. The early plateau suggests that the limit is the data and the breadth of the task rather than the training time. Whether a much larger dataset would recover the precision remains untested. The question that motivated the experiment, whether an explicit camera description helps, remains open: it needs a policy that positions precisely in the first place.

The next experiment (v9) narrows the camera range to a band around the measured real camera pose and the lens actually in use, keeps the servo model, and repeats the A/B comparison under conditions in which the policies can succeed.

## Research artifacts

**Datasets** (Hugging Face, LeRobot format):

- [inkoziev/so101-sim-grasp-v8a](https://huggingface.co/datasets/inkoziev/so101-sim-grasp-v8a): stereo images and joint angles (model A)
- [inkoziev/so101-sim-grasp-v8b](https://huggingface.co/datasets/inkoziev/so101-sim-grasp-v8b): the same episodes, with the camera description appended to the state (model B)

**Models** (Hugging Face):

- [inkoziev/smolvla-so101-sim-v8a](https://huggingface.co/inkoziev/smolvla-so101-sim-v8a): model A
- [inkoziev/smolvla-so101-sim-v8b](https://huggingface.co/inkoziev/smolvla-so101-sim-v8b): model B

**Code** (GitHub): [Koziev/so101-lab/experiments/exp8](https://github.com/Koziev/so101-lab/tree/main/experiments/exp8): simulator, data generation and LeRobot export, training and evaluation scripts.