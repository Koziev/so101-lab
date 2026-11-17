---
date: 2026-11-17
categories:
  - Project
  - Lab notes
tags:
  - SO-101
  - Qwen3.5
  - PyBullet
  - calibration
  - computer vision
  - robotics
---

# Experiment 11: calibrated cell, new prompt types, one-shot and closed-loop localization

## Concept

Experiment 10 showed that Qwen3.5-0.8B can locate a named object from a stereo pair and answer in the arm's joint space, but most of the effort went into finding geometric mismatches between the simulator and the real cell by hand. Experiment 11 had three aims:

1. Make the calibration of the new cell repeatable and largely automatic, and store every calibrated quantity in one place.
2. Extend the prompts from "colour + shape" to colour-only, shape-only and superlative ("the biggest cube") instructions.
3. Compare the one-shot formulation of experiment 10 with a closed-loop formulation, in which the model is called at every step and answers with a small change of the joint angles.

## Hardware

- **Arm:** SO-101 follower with 12 V servos (`my_follower_12v`).
- **Stereo camera:** side-by-side USB camera, 2 × 1280 × 720, 2.8 mm lenses, baseline 6.42 cm. It stands about 49 cm in front of the pan axis and 34 cm above the board, looking down at about 37°. The images given to the model are rectified and zoomed (factor 1.45) to a view of about 61° × 36°, then scaled to 640 × 360 per eye.
- **Wrist camera:** HBV-W202012HD, 1280 × 720 MJPG, mounted on the gripper. It was calibrated in this experiment but not used by any model.
- **Board:** plywood sheet; arm and objects stand on the same sheet.
- **Objects:** 3D-printed in white PLA (the real set) and in several colours (from earlier experiments).

## Calibration

### Setup folder

All calibrated quantities of the cell are stored in one setup folder (`setups/2026-11_12v_stereo28mm_wrist`), read by every tool:

| File | Content |
|---|---|
| `setup.json` | arm id, USB ports of the cameras |
| `arm.json` | joint offsets (simulator angle = real reading + offset) |
| `stereo_calib.json` | stereo intrinsics and extrinsics, rectification, pose of the left camera in the arm's base frame |
| `wrist_calib.json`, `wrist_mount.json` | wrist camera intrinsics and its pose on the gripper |
| `workspace.json` | taught workspace (recorded joint poses) |
| `table.json` | board height for the arm, measured with torque on |
| `stereo_reference.png`, `stereo_background.png` | reference frame for detecting camera displacement; median picture of the empty cell |

Every recording session starts with a comparison of the live stereo view against `stereo_reference.png`. Recording stops if either eye is displaced by more than 3 px. This check was added after the camera was found to have moved by about 1.5 cm between two sessions (see below).

### Two-stage procedure

The calibration is split into a manual and an automatic stage.

1. **Workspace teach-in.** With the torque off, the operator moves the arm by hand through the space it may use, starting and ending at the rest pose. Joint readings are recorded continuously (1,325 poses in the first recording, later recordings were longer). Positive wrist-roll angles were avoided on purpose.
2. **Automatic data collection.** The arm replays the recording and stops at selected poses. It travels only along recorded poses, keeps the jaw tips at least 20 mm above the board (from the kinematic model), and waits after each move until the image is still (1 s plus a frame-difference test). Some poses are revisited to measure repeatability.

### Hand-eye calibration of the wrist camera

**Boards.** The wrist camera's intrinsics were first calibrated with a single A4 ChArUco board (8 × 6 squares of 15 mm; 46 views, 0.99 px RMS; focal length 1,291 px, distortion model k1 + k2, square pixels). For the hand-eye calibration this board was too small to stay in view at the distances the arm reaches, so two A4 sheets are placed side by side without a gap. Each holds a 7 × 10 ChArUco board with 25 mm squares and 18 mm markers (dictionary DICT_4X4_100). The sheets carry disjoint marker ids (0–34 and 35–69), so every detected corner is assigned to its sheet. Marker corners are used in addition to ChArUco corners, which keeps close-up views usable when only part of a board is visible.

| Board A (ids 0–34) | Board B (ids 35–69) |
|---|---|
| ![ChArUco board A](/so101-lab/assets/exp11/charuco_board_a.png) | ![ChArUco board B](/so101-lab/assets/exp11/charuco_board_b.png) |

**Estimation.** One joint least-squares problem (SciPy, soft-L1 loss) estimates the camera's pose on the gripper, the pose of each board, and the offsets of shoulder_lift, elbow_flex and wrist_flex (shoulder_pan fixed). The wrist intrinsics can optionally be estimated as well. The start values come from a hand-eye solution (Park–Martin; the OpenCV 5.0 build used has no `calibrateHandEye`). Uncertainties are taken from the Jacobian.

**Results.**
- In a typical session, 31 of 55 stops gave usable views.
- The median reprojection error was about 11 px, with fixed or free intrinsics.
- Intrinsics estimated within the session (1,269–1,280 px) agreed with the separate calibration (1,291 px) within their uncertainty. So the wrist camera can be calibrated completely from the automatic session, without the separate board step.
- The reprojection error is limited by unreported joint play. Revisits of the same pose showed differences of 0.4–0.5° per joint, about 10 px in the wrist image.
- A gravity-sag term in the model did not reduce the error and was not adopted.

<video src="/so101-lab/assets/exp11/hand-eye-calibration.recoded.mp4" controls playsinline preload="metadata" width="100%"></video>

Video: [hand-eye-calibration.recoded.mp4](/so101-lab/assets/exp11/hand-eye-calibration.recoded.mp4). The arm replays the taught workspace and stops at selected poses; at each stop the wrist camera views the two ChArUco sheets.

**Servo behaviour** measured from the revisits (degrees):

| Joint | Bias | Dead band | Scatter (std) |
|---|---|---|---|
| shoulder_pan | 0 | 0.29 | 0.27 |
| shoulder_lift | +1.0 | 0.63 | ~1.0 |
| elbow_flex | +1.3 | 0.93 | 0.5–0.7 |
| wrist_flex | +0.2 | 0.34 | 0.42 |
| wrist_roll | – | 0.63 | – |

### Stereo camera calibration and the arm's parameters in the simulator

**Stereo pair.** The intrinsics and the stereo extrinsics were calibrated with a printed board in the usual way (RMS 0.30 / 0.36 / 0.35 px for left, right and stereo).

**Pose relative to the arm and joint offsets.** The camera's pose in the arm's base frame and the joint offsets are found with an object held by the arm itself:

1. A brightly coloured object is clamped between the jaws by hand.
2. The arm replays the taught workspace and stops at up to 300 poses.
3. At each stop the object is found in both eyes: by colour, in a window around its predicted position, optionally with an open-vocabulary detector (LLMDet). Pairs whose two detections are not consistent with the stereo geometry (more than 3–6 px off the epipolar constraint) are rejected.
4. One least-squares fit estimates the camera pose (6 parameters), the offsets of lift, elbow and wrist_flex, and the object's position in the gripper frame. Outliers are removed in rounds. Several sessions can be fitted together.

| Fit | Samples used | Held-out error | Pose uncertainty | Offsets lift / elbow / wrist_flex (deg) |
|---|---|---|---|---|
| 13 Nov, two sessions | 191 of 331 | 4.7–5.0 px | 2.7 mm, 0.29° | +0.61 / −6.46 / −0.44 |
| 14 Nov, three sessions | 177 of 272 | 5.8–5.9 px | 3.5 mm, 0.43° | +0.42 / −6.59 / −1.07 |

Notes on the two fits:
- **The 14 Nov fit was made after a suspected camera displacement.** It placed the camera 15 mm (mostly sideways) and 0.7° from the 13 Nov fit, about four times the uncertainty. So the camera had moved relative to the arm. Moving the plywood back to match the reference image had not restored the original geometry, because the reference image is dominated by the plywood's texture. This is the reason for the start-of-session check described above.
- **Collector fix.** In the first two sessions of 14 Nov only about a quarter of the requested stops were reached. The arm's rest pose matched the end of the taught recording, from where most of the workspace was cut off by the clearance rule. The collector now chooses the entry point into the recording from which most stops can be reached (148 of 200 in the third session).
- **The joint offsets were stable** between the two fits; only the camera pose changed.

<video src="/so101-lab/assets/exp11/front-stereocamera-calibration.recoded.mp4" controls playsinline preload="metadata" width="100%"></video>

Video: [front-stereocamera-calibration.recoded.mp4](/so101-lab/assets/exp11/front-stereocamera-calibration.recoded.mp4). The arm holds a coloured object and stops at poses of the taught workspace; the object is detected in both eyes of the stereo camera, and the fit gives the camera pose, the joint offsets and the object's position in the gripper.

### Board height and servo sag

The board's height for the arm was measured with the torque on: the arm lowers the tip of its fixed jaw in 1 mm steps at nine places until it touches a paper strip (`real/probe_table.py`).

- **From the read angles,** the board lies 3.7 ± 1.0 mm above the simulator's base plane, with no trend over the reach (14–26 cm). This value is used by the simulator.
- **The commanded and the read angles differ under load.** The shoulder_lift servo gives way by up to 4.3° with the arm stretched. At 26 cm reach the jaw tip ends 1.2–1.7 cm lower than commanded; at 14 cm there is no difference. This sag is not repeatable between neighbouring places (0.3° at one, 3° at the next).
- **The same measurement by hand with the torque off gave +12.5 ± 4.9 mm.** That method is affected by the play of the joints and was not used.
- **The elbow joint shows play** that lets the jaw tip wander by up to 1 cm in some poses. The screws were tightened; the play was not measured again.

The sag and the dead bands are modelled in the closed-loop evaluation (`closed_loop.ServoModel`): reading minus command for the lift is 0° at 14 cm reach, rising to about 4.3° at 26 cm, with a random scale per episode. They are not part of the one-shot data, whose labels are target angles.

## Tasks and prompts

Every scene is built around its prompt, so that the prompt refers to exactly one object; the uniqueness rule is checked again on the objects actually placed.

| Task type | Share | Example prompt | Rule |
|---|---|---|---|
| full | 40 % | "pick up the white cylinder" | colour + shape pair unique |
| colour | 20 % | "grab a violet thing" | colour unique |
| shape | 20 % | "take the cone" | shape unique |
| superlative | 20 % | "grab the biggest cube", "compare the balls on the table, choose the smallest, grab it and lift it" | 2–3 objects of that shape; the target's volume larger (smaller) than every other of its shape by a factor of at least 1.35 (about 10 % in linear size) |

- **Wording.** Synonyms for the object ("object", "item", "thing", "piece", "body"), for shapes ("box" or "block" for cube, "sphere" for ball, "hex prism"), for colours ("violet", "grey") and for superlatives ("largest", "tiniest"); eight verb templates; definite or indefinite article. Each sample also stores a canonical form of its prompt.
- **Objects.** Seven printed shapes: cube, ball, cylinder, hexagonal prism, dodecahedron, cone, pyramid. Sizes cover the measured ranges of the printed objects widened by 1 mm. Half of the objects are white (adjustable); a quarter lie on their side. There are 2–5 objects per scene.

Representative task clips can be generated from the simulator with `tools/make_docs_media.py tasks`. The scripted controller in those clips is the label-making expert, not a trained model; servos are ideal and the motion is kinematic, so the object is not physically lifted. The generated clips and thumbnails are not included in the published media set for this entry.

## Simulator changes

The simulator is the PyBullet scene of experiments 9 and 10, with these changes:

- **Calibrated camera.** The stereo camera is centred on the calibrated pose, with the rectified field of view and the measured baseline. The camera's pose includes the rotation of the rectification (a fraction of a degree, about 2 px at 640 × 360). Training cameras are randomised around this pose; test sets use *T* = 0 (calibrated pose), 1 and 2 standard deviations.
- **Real background.** The picture of the empty cell (`stereo_background.png`, the median of frames with the arm in different places) is rectified exactly like the model's input and used as the scene. Only the arm, the objects and their shadows are rendered and drawn on top.
  - For randomised cameras the picture is warped with the homography of the board plane: exact for the board, approximate for the walls.
  - The area around the arm's base appears in three variants per episode (untouched, partly or fully painted over), because the median picture keeps remnants of the arm there.
  - Random low-contrast patches and thin strokes are added to the board, so that incidental details do not stay fixed between training images.
  - Several background pictures can be stored; one is drawn per episode.
- **Board height** from `table.json` (3.7 mm).
- **Label height.** The grasp point is placed 2.0 cm above the board for every object, which keeps the fingertips about 8 mm clear. In the first datasets of this experiment the grasp point was at the object's mid-height (cones and pyramids lower). That put the fingertips 0–3 mm above the board, or below it for small objects, while the operator guiding the real jaws kept them at least 10 mm clear.
- **Object range** 12–30 cm from the pan axis (12–24 cm before). The camera does not see the board beyond about 26 cm straight ahead, and scenes with an object outside the image are skipped.
- **Arm appearance.** The arm is brightened to near white, like the printed arm. The last 20 mm of both jaws are drawn black on their outer surfaces, as painted on the real arm on 16 Nov (`--tip-paint-mm`, 0 for the unpainted arm). The tip of each jaw is located from the arm's collision geometry at start-up.
- **Image modes.** Besides the whole view at 640 × 360, a window around the workspace can be given to the model at 1.5× or 2× the pixel density (768 × 432 or 1,024 × 576). It is rendered directly at that size. Real images for any mode are cut from the raw camera frames, which the recording tool now stores with every sample.

The comparison figures are reproducible with `tools/make_docs_media.py compare` from the real set `data_real_v11b`, recorded before the jaw tips were painted. Each output places the real left image next to the simulator rendering of the same cell, with the arm in its ready pose and the target at the recorded grasp point; the other objects in the real scene are not reproduced. Those generated comparison PNGs are not included in the published media set for this entry.

Known differences that remain:
- The real arm carries the wrist camera and loose cables; the simulated arm has neither.
- The arm's shadow is approximated from the bounding boxes of its links.
- In training samples with strongly displaced cameras, the rendered arm can be blown out to pure white when only a small, shaded part of it is in view.
- In some closed-loop training samples the jaw tips are outside the image.

## Model and training

Qwen3.5-0.8B, fully fine-tuned as in experiment 10 (bf16, loss on the answer only, effective batch 16), on one RTX 5080 with 16 GB. One-shot models: 20,000 training scenes, 2 epochs, about 1.5 h. At 1,024 × 576 a pair of images takes 1,168 input tokens (584 per image, about 32 × 32 pixels per token); the processor does not reduce the images.

Evaluation metric as in experiment 10: the distance between the grasp point placed with the predicted angles and the target point, median over the test set. "Wrong object" counts predictions closer to another object than to the target.

## One-shot localization

The model sees the stereo pair with the arm in its ready pose and the instruction, and answers once with the four target angles.

### Simulation

| Model | Scene, labels, object range | Image | *T* = 0 | *T* = 1 | *T* = 2 |
|---|---|---|---|---|---|
| v11 | synthetic board and backdrop, mid-height labels, 12–24 cm | 640 × 360 | 0.78 cm (98 %) | 0.78 cm (95 %) | 0.92 cm (88 %) |
| v11b | real background, labels 2.0 cm above the board, 12–30 cm | 640 × 360 | **0.61 cm (99 %)** | **0.54 cm (98 %)** | **0.64 cm (95 %)** |
| v11c2x | as v11b | workspace window 1,024 × 576 | 0.65 cm (98 %) | 0.69 cm (96 %) | 0.83 cm (89 %) |

Median error, share within 2 cm in brackets. All answers were valid JSON. The v11 and v11b runs differ in several respects, so their difference is not attributable to one change. v11c2x was trained on 16,302 samples (scenes with an object outside the window are skipped) and tested on subsets of the test scenes.

- **Task types.** Once wrong-object cases are removed, all four task types are located equally well (v11: 0.71–0.87 cm at *T* = 0). Wrong-object errors are almost all superlatives. For v11: 3 of 54, 4 of 52 and 7 of 52 at *T* = 0, 1, 2, against 0–2 of about 248 for the other task types. Most of them pick the other object of the same shape.
- **Superlatives and perspective.** Superlative errors do not depend on whether perspective makes the target look larger or smaller in the image (10 of 11 right where perspective contradicts the answer). They become more frequent with small size differences under shifted cameras.
- **Direction of the misses (v11b, T = 0):** 0.44 cm in reach and 0.30 cm sideways (medians); height error 0.05 cm. Far targets have the largest misses (0.83 cm beyond 22 cm, 0.48 cm below 17 cm).
- **Higher pixel density did not improve precision.** Reach improved slightly (0.44 → 0.34 cm), sideways got worse (0.30 → 0.40 cm), and robustness to camera shifts decreased. The superlative errors were unchanged. The camera's sensor resolves about 1.35 times the density of the 640 × 360 images, so the 2× window adds pixels but little optical detail. In simulation, however, the objects are rendered natively at that density, and the model did not use the additional detail.

### Real arm

Reference points are recorded as in experiment 10: the stereo pair is captured at the ready pose, the operator guides the jaws onto the target, and the reference point is computed from the joint readings with the setup's offsets.

| Model | Real set | Median | Within 2 / 3 cm | Reach, mean ± std | Sideways, mean ± std | Height, mean ± std |
|---|---|---|---|---|---|---|
| v11 | 31 samples (14 Nov, first set) | 3.39 cm | 10 / 32 % | −1.95 ± 1.25 cm | −1.95 ± 1.20 cm | −0.87 ± 0.28 cm |
| v11 | 61 samples (15 Nov, second set) | 3.28 cm | 3 / 39 % | −1.60 ± 0.85 cm | −2.06 ± 1.05 cm | −1.31 ± 0.35 cm |
| v11b | 31 samples, reference points recomputed with the 14 Nov offsets | 1.57 cm | 68 / 74 % | −0.31 ± 1.34 cm | +0.27 ± 1.33 cm | +0.19 ± 0.26 cm |
| v11b | 61 samples (second set) | **1.00 cm** | **90 / 93 %** | −0.07 ± 0.77 cm | +0.15 ± 0.89 cm | −0.26 ± 0.38 cm |

The direction columns exclude gross errors (above 5 cm).

**The v11 model on the real arm** had a constant offset: about 2 cm short in reach, 2 cm sideways (a rotation of about 5.5° around the pan axis) and about 1 cm low. The causes found:
- **Targets out of range.** 11 of the 31 targets of the first set lay beyond the training range of 24 cm, some of them partly outside the image (4 of the 6 gross errors).
- **Label height.** The labels put the fingertips at the board while the operator kept them at least 1 cm above it, which explains the constant height offset.
- **Not the camera geometry.** A comparison of simulated and real images at the ready pose showed the arm and its base in the same place within about 10–15 px (about 1°). The rotation therefore did not come from the camera pose or the pan zero. The simulated scene around the arm differed strongly from the real one: a smaller, darker board whose edges and corner crossed the image, and synthetic surroundings.

**With the real background, the new label height and the wider object range (v11b)**, the constant offsets disappeared, and the remaining misses are scatter around zero.
- The rotation could only have come from the scene, so the synthetic board was most likely its cause. The three changes were made together, and their individual contributions were not measured.
- On the second real set, all task types and shapes are within 0.8–1.3 cm (median), including 15 of 16 superlatives within 2 cm. Several of those superlative instructions compared objects of different shapes ("the biggest object"), which is not in the training data.
- The two gross errors are shape confusions with three objects in the scene: a black cube taken for the green dodecahedron, and a yellow cube for the yellow hexagonal prism next to it.
- For comparison, experiment 10's real result was 1.88 cm median and 58 % within 2 cm. The two sets differ in objects, prompts and cell, so this is not a controlled comparison.

During the evaluation, the second set was first scored with the v11 model by mistake. That run is kept in the table above, because it reproduces the v11 offsets on an independent recording.

## Closed-loop control

The model is called at every step. It sees the stereo pair with the arm where it is at that moment, the instruction and the current joint angles (one decimal), and answers with the change of the four angles for the next step and a gripper decision:

```json
{"shoulder_pan": 1.6, "shoulder_lift": -0.4, "elbow_flex": 2.1, "wrist_flex": 0.0, "gripper": "open"}
```

- **Controller.** It adds each change to the servo command, not to the reading. With the shoulder sag, adding to the reading settles short of the target by the sag and never closes; adding to the command integrates the sag away. In a test with 4.3° of sag: closed after 7 steps with 0.0° lift error, against not closed after 40 steps with 4.3°.
- **Labels.** They come from a scripted expert: move towards a point 5 cm above the target, then down onto it, with at most 8° per joint and step.
- **Training data.** Arm poses are sampled along the approach: ready pose, on the way, during the descent, and around the target. The angles in the prompt carry 0.3° of noise relative to the rendered pose. 12,000 scenes × 2 poses = 24,000 samples, 2 epochs, starting from the v11b one-shot model.
- **Evaluation.** Single steps on test poses, and whole approaches in the simulator (100 per camera width): new scenes, the model called step after step, a servo model (sag, dead bands) between command and pose, at most 30 steps. The expert itself, run through the same loop, closed every episode within about 1 mm.
- **Label versions.**
  - **Version 1:** "close" when every joint is within 1° of the target angles; near the target, the label moves straight to those angles.
  - **Version 2:** "close" when the grasp point is within 6 mm horizontally and 3 mm vertically of the target; near the target, the label moves the grasp point straight to the target and keeps the gripper's pitch.
  - **Version 3:** version 2 with the jaw tips painted black in the simulator. The arm's brightness rendering also changed between versions 2 and 3.

### Baselines

The one-shot model v11b was run through the same scenes and servo model: asked once, its answer commanded directly ("direct"), or then corrected from the servo readings until they match the answer, with no further model call ("corrected").

### Results

| Policy | *T* | Horizontal miss | Within 1 / 2 cm | Wrong object | Jaws on the board | Steps |
|---|---|---|---|---|---|---|
| one-shot, direct | 0 | 0.73 cm | 70 / 97 % | 1 % | 12 % | 1 |
| | 1 | 0.69 cm | 77 / 97 % | – | 14 % | 1 |
| **one-shot, corrected** | 0 | **0.57 cm** | **87 / 98 %** | 1 % | **0 %** | 2 |
| | 1 | 0.59 cm | 81 / 97 % | – | 0 % | 2 |
| closed loop, version 1 | 0 | 0.80 cm | 63 / 92 % | 1 % | 1 % | 8 |
| | 1 | 0.76 cm | 62 / 89 % | 3 % | 4 % | 9 |
| closed loop, version 2 | 0 | 1.03 cm | 48 / 79 % | 6 % | 13 % | 10 |
| | 1 | 1.07 cm | 44 / 85 % | 5 % | 6 % | 10 |
| closed loop, version 3 (painted tips) | 0 | 1.14 cm | 41 / 75 % | 6 % | 10 % | 10 |
| | 1 | 0.95 cm | 53 / 81 % | 6 % | 5 % | 9 |

Medians at closing; "within" counts episodes closed on the right object; "–" = not extracted. Arm motion is kinematic: objects are not pushed and grasps are not tested.

**No closed-loop version reached the corrected one-shot baseline.** The details, version by version:

- **Version 1** closed every episode but stopped early. In the single-step test it said "close" in 12–14 % of the poses where it should not, mostly near the target with a predicted change close to zero. At closing its joints were a median of 4.4° from the target angles, while the grasp point was 0.8 cm off. Many joint combinations put the grasp point in nearly the same place, so near the target the labels asked for corrections that are not visible in the image.
- **Version 2** removed that ambiguity and did worse.
  - Near-target poses that needed a correction: the model's step ended a median of 0.65 cm from the expert's.
  - It declared "arrived" in about half of them, with a median of 0.63 cm still to go.
  - The height at closing scattered by 0.6 cm, which led to more board contacts.
  - In three episodes at *T* = 0 the accumulated commands drifted away from the target without closing.
- **Where the model takes the arm's pose from.** A probe shifted one joint's angle in the prompt by 3° while keeping the image (`evaluation/probe_prompt_shift.py`). Near the target the answers did not follow the shift: median follow factor 0.00 for shoulder_pan and 0.03 for shoulder_lift, where 1 would mean the answer follows the numbers completely. Only at the ready pose did the answer follow the lift angle in the prompt (8 samples). So the model reads the arm's pose from the image, and its visual estimate of the gripper-object offset is not more precise than its estimate of the object's position in the one-shot task (about 0.6 cm).
- **Version 3 (painted tips)** tested whether a clearly visible gripper helps. Near-target step error 0.75 / 0.65 cm (*T* = 0 / 1) against 0.65 / 0.65 cm in version 2, and premature closing 46 / 53 % against 55 / 50 %: no measurable difference. The change in the arm's brightness between the two versions and the training samples with blown-out arm or tips out of view could weaken an effect, but are unlikely to hide a large one.

## Results in brief

1. **Calibration.** The cell's camera poses, wrist-camera intrinsics and joint offsets are obtained from two semi-automatic sessions with the arm itself (ChArUco sheets for the wrist camera, a held object for the stereo camera), stored in one setup folder and checked at every session start. Precision is limited by joint play of about 0.5°.
2. **New prompt types.** Colour-only, shape-only and superlative instructions are localized as precisely as colour + shape instructions once the right object is chosen. Superlatives have the most wrong choices (4–14 % in simulation, depending on the model and the camera shift).
3. **One-shot on the real arm.** With the real background, the corrected label height and the wider object range, the one-shot model reached 1.0 cm median error and 90 % within 2 cm on 61 real samples, without systematic offsets. In simulation the corresponding figure is 0.6 cm.
4. **Servo sag.** The shoulder sag of up to 1.7 cm at full reach is removed by correcting the command from the servo readings after the one-shot answer. Commanding the answer directly led to board contact in 12–14 % of simulated approaches.

### Negative results

1. **Twice the pixel density did not improve the one-shot localization** in simulation (0.65 against 0.61 cm at *T* = 0), and reduced robustness to camera shifts.
2. **Closed-loop control did not improve on the one-shot model** in any of three label versions. The best version reached 0.80 cm against 0.57 cm for the corrected one-shot baseline, closed early, and touched the board in some episodes.
3. **A clearly marked gripper did not improve the closed-loop corrections** in simulation.
4. **A gravity-sag term did not improve the hand-eye calibration.**
5. **A displacement of the stereo camera by about 1.5 cm went unnoticed** until the second real evaluation; realigning the board to the reference image did not restore the geometry.

### Errors in the course of the experiment

- The label definition of the first datasets put the fingertips at or below the board; the first real set contained targets outside the trained range and partly outside the image. Both were found from the real results and corrected (see above).
- A rendering defect was suspected in the composited images and a training run was stopped. The suspicion was wrong (the gripper had been mistaken for part of the base), and the run was restarted.
- The first implementation of the painted jaw tips placed the paint wrongly twice. It was corrected and checked against a real frame before the version 3 dataset was generated.

## Open questions and next steps

- **Real grasp trials** with the one-shot model and reading-corrected motion, on a new real set recorded with the painted jaw tips.
- **Generator fixes** before any further closed-loop attempt: a brightness reference that does not depend on how much of the arm is visible, and skipping poses with the jaw tips out of view.
- **Superlatives:** a dedicated test set with smaller size differences would give an accuracy-versus-ratio curve.

## Files

- **Code:** [`experiments/exp11/`](https://github.com/Koziev/so101-lab/tree/main/experiments/exp11) — setup-folder handling, calibration and camera tools, simulator, dataset generators, training, evaluation, real-arm utilities and reproducible media-generation scripts.
- Media: `docs/assets/exp11/`.