# so101-lab: experiment v8

Code to reproduce experiment v8 of the SO-101 research log: SmolVLA policies for grasping a single named
object, trained only on data from a PyBullet simulator, with widely randomised stereo-camera viewpoints
and a model of the real arm's servo behaviour. Model A sees stereo images and joint angles; model B
additionally receives a 10-number description of the camera.

Published artefacts:
datasets [inkoziev/so101-sim-grasp-v8a](https://huggingface.co/datasets/inkoziev/so101-sim-grasp-v8a),
[inkoziev/so101-sim-grasp-v8b](https://huggingface.co/datasets/inkoziev/so101-sim-grasp-v8b);
models [inkoziev/smolvla-so101-sim-v8a](https://huggingface.co/inkoziev/smolvla-so101-sim-v8a),
[inkoziev/smolvla-so101-sim-v8b](https://huggingface.co/inkoziev/smolvla-so101-sim-v8b).

## Layout

```
simulator/
  so101_sim/          simulator package (installed with pip install -e .)
    env.py            environment: scene presets v1/v7/v8, physics, cameras, servo model, observations
    kinematics.py     FK/IK, gripper geometry measured from the URDF
    expert.py         scripted grasping expert (perturbations, early lift, servo-lag compensation)
    scene_v7.py       board-on-desk scene, photo textures, stereo camera, camera noise, object shadows
    scene_v8.py       wide camera randomisation, held-out viewpoints, camera description
  textures/           photo textures of the real setup (board, desk, backdrop)
  prepare_assets.py   downloads the official SO-101 URDF + meshes, builds the jaw collision meshes
  check_so101.py      sanity checks and preview images of the simulator
dataset/
  record_episodes.py  parallel generation of raw episodes (sharded, resumable)
  repair_raw.py       finds and removes incomplete episodes (e.g. after a full disk)
  export_lerobot.py   raw episodes -> LeRobotDataset (stereo pair; --append-camera for model B)
  generate_v8.sh      the exact v8 generation and export commands
training/
  train_v8.sh         lerobot-train commands for models A and B
evaluation/
  eval_policy.py      closed-loop evaluation in the simulator (success, failure types, breakdowns)
  expert_check_v8.sh  expert sanity check under the v8 servo model
  eval_v8.sh          the five-run v8 evaluation matrix
  diagnostics_v8.sh   the four diagnostic runs of the v8 failure analysis
```

## Setup

Python 3.12, LeRobot 0.6.0 and a PyTorch build matching your NVIDIA driver (see `requirements.txt`).

```bash
conda create -n so101 python=3.12 -y && conda activate so101
conda install -c conda-forge ffmpeg -y
pip install -r requirements.txt      # install the matching torch build first if needed
pip install -e .                     # makes the so101_sim package importable
python simulator/prepare_assets.py --out assets_so101
python simulator/check_so101.py --out check_v8 --scene-preset v8 --width 640 --height 360 --trials 5
```

All commands are run from the repository root (default paths: `assets_so101/`, `datasets/`, `outputs/`).

## Reproducing v8

1. `evaluation/expert_check_v8.sh`: the scripted expert should succeed in about 97 % of scenes.
2. `dataset/generate_v8.sh`: 2000 episodes (~12–14 h with 6 CPU workers), then two exports.
3. `training/train_v8.sh a` and `training/train_v8.sh b`: ~16 h each on an RTX 3090.
4. `evaluation/eval_v8.sh`: five evaluations, each on the same 187 scenes.
5. Optional: `evaluation/diagnostics_v8.sh`.

Every episode and evaluation scene is determined by its random seed, so the scenes are reproducible;
training results vary somewhat between runs.

## Results reported in the research log

| Model | Viewpoints | Success |
|---|---|---|
| A | training range | 25/187 (13 %) |
| B | training range | 37/187 (20 %) |
| A | held out | 17/187 (9 %) |
| B | held out | 20/187 (11 %) |
| B, camera description wrong by 2 cm / 3° | held out | 19/187 (10 %) |

The narrowly trained v7 policy reached 79 %. Diagnostics showed no improvement after 20k training steps,
a large loss of precision even on v7's own scenes, and no effect of the servo model, which points to the
breadth of the camera randomisation as the cause. The v7 policy used in the diagnostic comparison is not
part of this release.

## Notes

- Rendering uses PyBullet's CPU renderer; object shadows are computed geometrically (`scene_v7.py`).
- Units: arm joints in degrees (URDF angles, 0 = middle of the range, as LeRobot's calibration in
  degrees mode), gripper 0 = closed … 100 = open.
- Model B on the real arm needs the camera description computed from a camera-pose calibration
  (`scene_v8.camera_desc_from_real_pose`); the real-arm tools are not part of this release.
