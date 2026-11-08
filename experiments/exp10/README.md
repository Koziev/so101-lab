# Experiment 10: locating objects in the SO-101 arm's joint space with Qwen3.5-0.8B

A multimodal LLM (Qwen3.5-0.8B) receives the two images of a stereo camera and a short description of the
target ("blue ball") and answers with JSON: the angles of `shoulder_pan`, `shoulder_lift`, `elbow_flex` and
`wrist_flex` (degrees, LeRobot convention, 0 = middle of the range) that place the gripper's grasp point at the
centre of that object. Training data comes only from a PyBullet simulator of the SO-101; the model is evaluated
in simulation and on the real arm. Results and discussion: see the experiment 10 entry of the research log.

**Label convention.** wrist_roll fixed at 0; one fixed grasp point between the jaws (the one for a 30 mm
object); the approach as steep as reachable (vertical, else tilted 15/30/45° towards the base).

## Layout

```
common.py                   target definition, prompt/answer format, labels, grasp-point geometry,
                            camera centre from a calibration file
simulator/                  SO-101 simulator (scene v9: board, photo textures, servo model, stereo camera)
  prepare_assets.py         downloads the official SO-101 URDF + meshes, builds the jaw collision meshes
dataset/generate.py         parallel generation: train split + test splits T0/T1/T2 (Parquet, JPEG embedded)
training/train.py           transformers.Trainer, prompt-masked loss, optional LoRA (--lora)
evaluation/evaluate.py      greedy generation, JSON parsing, per-joint error, grasp-point distance
evaluation/analyze_real.py  real results per object count / arrangement, gross errors, sideways bias
evaluation/fit_offsets.py   constant joint offsets real <-> simulator, cross-validated on halves of a real set
evaluation/predict_server.py  the trained model behind a small HTTP API (for the real-arm tools)
camera/                     stereo camera: live view, calibration + pose measurement, realignment to a reference
real/record_real.py         records real test samples (stereo pair at the ready pose + hand-guided reference)
real/label_arrangements.py  labels the arrangement of each real sample (single, same-colour pair, ...)
real/apply_offsets.py       applies joint offsets to recorded samples (writes a corrected copy)
real/arm_convention.*       the joint-offset convention (simulator = real + offset) and the values of our arm
real/grasp_real.py          pointing / scripted grasping on the real arm driven by the model
run_smoke.sh, run_v2.sh     the two simulated training runs of the experiment
```

## Environments

Two Python environments are used, because Qwen3.5 needs a recent `transformers` that does not match the
version pinned by LeRobot:

- **model environment** (generation, training, evaluation, prediction server): `requirements.txt`. Install the
  optimised kernels for Qwen3.5's linear-attention layers (`flash-linear-attention`, `causal-conv1d`); without
  them training and inference are several times slower. `causal-conv1d` compiles against the CUDA toolkit
  (`nvcc`) whose major version must match the one PyTorch was built with.
- **robot environment** (everything under `real/` and `camera/`): LeRobot 0.6 with the Feetech extra, plus
  `opencv-python`, `pybullet`, `datasets`.

All commands are run from this folder. First: `python simulator/prepare_assets.py --out assets_so101`.

## Workflow

1. **Camera.** `camera/capture_stereocamera_stream.py` (live view, both eyes, checkerboard detection),
   `camera/calibrate_stereo.py` (stereo calibration and `pose` of the camera relative to the pan axis, with a
   checkerboard lying flat on the robot's forward line), `camera/camera_align.py --etalon ref.png` (realign the
   camera to a reference image of the empty board after any disturbance). The robot base and the ready-pose
   gripper must be visible: the simulator only generates cameras that see them.
2. **Data:** `python dataset/generate.py --out data_v2 --train 10000 --test 200 --train-scale 1.5 --calib stereo_calib_wide.json --synonyms`
   (cameras centred on the calibrated pose; the left-eye pose is converted to the stereo centre).
3. **Training:** `python training/train.py --data data_v2 --out outputs/qwen35_08b_v2 --epochs 2 --batch 8 --grad-accum 2`.
4. **Simulated evaluation:** `python evaluation/evaluate.py --model outputs/qwen35_08b_v2/final --data data_v2 --out outputs/eval_v2_sim.json`.
5. **Real test set:** `python real/record_real.py --port /dev/ttyACM0 --id my_follower --calib stereo_calib_wide.json --out data_real`
   (torque is released only after you confirm you are supporting the arm; between samples the arm rests folded).
6. **Joint offsets:** check the pan zero physically (arm along the board's centre line, read `shoulder_pan`),
   evaluate the real set, then `evaluation/fit_offsets.py --save offsets.json` and `real/apply_offsets.py`.
   Confirm on a newly recorded set.
7. **Grasping:** `python evaluation/predict_server.py --model ...` (model environment) and
   `python real/grasp_real.py --mode point|grasp ...` (robot environment).

## Results (summary)

| | median grasp-point error | within 2 cm |
|---|---|---|
| simulation, calibrated camera pose (T0) | 0.96 cm | 94 % |
| simulation, cameras displaced (T1 / T2) | 1.10 / 1.22 cm | 86 / 78 % |
| real, without joint-offset correction | 4.54 cm | 6 % |
| real, offsets fitted on one set, tested on a new set (74 samples) | 1.88 cm | 58 % |

Scripted grasps driven by the model: 11 of 21 successful (pointing: median miss 1.2 cm).

`real/arm_convention.json` holds the offsets of our own 7.4 V follower and camera setup; other arms need their
own values (steps 5–6).
