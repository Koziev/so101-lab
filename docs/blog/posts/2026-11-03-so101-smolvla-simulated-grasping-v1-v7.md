---
date: 2026-11-03
categories:
  - Project
  - Lab notes
tags:
  - SO-101
  - SmolVLA
  - PyBullet
  - LeRobot
---

# SO-101 grasping with SmolVLA trained on simulated data: experiments v1–v7

This entry summarises the first seven experiments of a project that trains a small vision-language-action model to grasp objects with an SO-101 arm, using only synthetic trajectories from a simulator. The results below are from simulation unless stated otherwise. Each experiment changes one thing (or one group of related things) relative to the previous one.

## Common setup

**Simulator.** PyBullet with the official SO-101 model (`so101_new_calib.urdf`, joint zero = middle of each joint's range). The jaw collision meshes were replaced by convex decompositions; without that, PyBullet treats each jaw part as a single convex hull and physical grasping is impossible. Grasps are fully physical: gravity, position-controlled joints with torque limits, friction between jaws and object.

**Task.** One object on the table: a cube (22–35 mm) or a ball (24–40 mm diameter) in one of 12 colours, placed 12–24 cm from the arm's base. The instruction is a short template such as "pick up the red cube".

**Demonstrations.** A scripted expert with access to the object's true position plans the grasp with inverse kinematics: approach above the object, straight descent, close, lift, hold. The grasp point is computed from the measured gap between the jaws (the URDF's tool frame turned out to lie inside the fixed finger).

**Model.** `lerobot/smolvla_base` (about 450 M parameters), fine-tuned with LeRobot on an RTX 3090, batch size 64.

**Evaluation.** Closed-loop runs in the simulator on 129 scenes never used for training (182 in v7). Success means the object is held between both jaws and lifted at least 3 cm. Most comparisons are done scene by scene with the same random seeds; with about 130 scenes, a single success rate has an uncertainty of roughly ±8 percentage points. SmolVLA predicts chunks of 50 actions; we evaluate executing all 50 ("chunk 50") and re-planning every 10 steps ("re-plan 10").

## Results at a glance

| Exp. | Change | Episodes generated | Success, chunk 50 | Success, re-plan 10 |
|---|---|---|---|---|
| v1 | baseline: front + wrist camera, 320×240 | 600 | 40 % | 47 % |
| v2 | perturbed demonstrations | 600 | 63 % | 74 % |
| v3 | v2 + overhead camera | 600 | 54 % | 74 % |
| v4 | v2 + 512×512 images, new camera framing | 600 | 57 %* | 67 %* |
| v5 | v2 recipe with 2000 episodes | 2000 | 68 % | 81 % |
| v6 | v5 + lift as soon as the gripper is blocked | 2000 | – | 89–91 % |
| v7 | new scene modelled on the real setup, stereo camera only | 2000 | – | 79 % (86 %)** |

Only successful demonstrations are used for training (about 86–97 % of the generated ones, depending on the recipe).
\* on the 126 scenes shared with v2 (v2 on the same scenes: 62 % / 73 %).
\** 79 % on table/background textures held out from training, 86 % on training textures.

## v1: baseline

Front camera and wrist camera, 600 generated episodes, 20 k training steps. Success was 40–47 %. Analysis of the failures showed that almost all of them were positioning errors: grasps succeeded almost every time when the gripper closed within 2 cm of the intended grasp point, and almost never beyond 3 cm. The median distance at closing was 2.5 cm. Inference took about 220 ms per action chunk (32-bit precision, no optimisation).

## v2: perturbed demonstrations

In half of the episodes the executed arm motion was pushed sideways by 1–3 cm during approach and descent, while the recorded actions remained the expert's clean plan. The data then contains off-target states paired with corrective actions, which the clean demonstrations of v1 never showed. Success rose to 63 % / 74 %, the median closing distance fell to 1.5 cm (re-plan 10), and the difference to v1 is far beyond chance. Re-planning every 10 steps now helped significantly, which it had not in v1. Training curves showed a plateau after about 15 k steps.

## v3 and v4: more visual input (negative results)

**v3** added a third, overhead camera. With re-planning the result was identical to v2 (95 vs 95 successes); with 50-step chunks it was worse (54 % vs 63 %).
**v4** rendered 512×512 images instead of upscaled 320×240, aimed the front camera at the object area and moved the wrist camera. No measurable change (67 % vs 73 % on the same scenes, not significant).
Neither more views nor sharper images improved positioning, which suggested that the limit lay elsewhere (data quantity or content) rather than in image detail.

## v5: more data

The v2 recipe with 2000 episodes and 40 k training steps. Success with re-planning rose to 81 %; the gain was concentrated where data had been thin (objects close to the base: 8/23 → 15/23). A new failure type appeared: the gripper closed correctly on the object but the arm never lifted. The cause was in the data: the expert waited motionless for 0.8 s while closing, so a nearly identical image was labelled "wait" in some frames and "lift" in others. A policy without memory that re-plans every 10 steps kept choosing "wait". Longer chunks (25 steps) removed the stall but made positioning worse, with no net gain.

## v6: lift as soon as the grip is secure

The expert now monitors the measured gripper position and starts lifting once the jaws stop moving against the object, which reduces the ambiguous motionless frames from up to 14 to about 5. Success with re-planning reached 89–91 %, the stall almost disappeared, and the median closing distance improved to about 1.0 cm. Remaining failures were mostly large objects close to the base.

## v7: a scene modelled on the real setup

The scene was rebuilt after the real setup: a 50×49 cm plywood board on a desk, textures made from photos of the actual board, desk and room, background clutter, a mostly white arm, and lighting from behind the robot. The front and wrist cameras were replaced by a single stereo camera modelled after the real one (a side-by-side USB stereo camera, 6.2 cm baseline, about 45° horizontal field of view after rectification), placed 55–65 cm in front of the base. Objects were restricted to the ±30° sector this camera can see. Episodes start from a low ready pose so that the gripper is visible from the first frame, and rendered images pass through a noise model fitted to real frames (white balance, contrast, blur, sensor noise, JPEG).

Two rendering problems had to be solved: PyBullet's CPU renderer drew no shadows for small objects (a 1 cm near clipping plane left too little depth precision), and its dithered shadows were washed out by the camera noise model. Object shadows are now computed geometrically.

Result: 79 % on held-out textures and 86 % on training textures. Positioning was less precise than in v6 (median closing distance 1.9 cm), objects near the base remained the weakest zone (60 %), and balls were harder than cubes (74 % vs 84 %). The texture split contained only a few held-out textures, so the 7-point gap should be read as an indication, not a measurement.

## First tests on the real arm

The v7 policy was run on a real SO-101 with 7.4 V servos and the real stereo camera. After a configuration error was found (a wrongly recorded joint direction for the base rotation), the arm moved towards the correct object but closed about 2 cm away from it. Offline tests on the logged frames showed that the policy located the object on the correct side, but reacted less strongly to real images than to simulated ones. Three systematic differences from the simulation were measured:

- the shoulder and elbow servos stop 2–4.5° short when moving in one direction (a direction-dependent dead band), enough to shift the gripper by about 1 cm;
- the control loop initially ran at about 5 commands per second instead of 30 (about 17 after restructuring; inference still pauses the loop for about 240 ms per chunk);
- the real camera was closer (52.5 cm) and looked down more steeply (38.7°) than any camera in the training data.

## Summary

- Showing the policy how to recover from small offsets (v2) gave the largest single improvement.
- More cameras and higher image resolution (v3, v4) did not help in this setup.
- More data helped mainly where data had been sparse (v5).
- Ambiguous waiting frames in demonstrations can make a memoryless policy stall (v5, fixed in v6).
- A simulated success rate around 80–90 % did not transfer directly to the real arm; servo behaviour, control rate and camera placement all differ from the simulation.

All results come from a single training run per configuration; differences of a few percentage points between configurations are within the run-to-run and evaluation noise. The next entry covers v8, which addresses the real-arm findings: widely randomised camera positions, an explicit camera description as model input, and the measured servo behaviour in simulation.