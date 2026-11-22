# Experiment 11: automated calibration of the cell (stage 1)

One fixed ChArUco board, the arm moving by itself, and a joint estimation of
- the wrist camera's mounting on the gripper (camera -> gripper_link),
- the board's pose and the stereo camera's pose in the robot frame (no tape measure),
- the arm's joint zero offsets (shoulder_lift, elbow_flex, wrist_flex; pan from the physical check).
All settings live in one **setup folder** per hardware configuration (`setup_io.py`).

```
setup_io.py              setup folders: create / show; camera devices found by USB port
calib/urdf_fk.py         forward kinematics from the URDF (NumPy); self-test against the simulator's PyBullet
calib/estimate.py        joint estimation (robust least squares, priors, uncertainties), hand-eye initialisation
calib/plan.py            pose planning: safety (limits, heights, transitions), bootstrap, diverse planned poses
calib/session.py         the loop: bootstrap -> planned batches -> re-estimation -> convergence
calib/autocalib.py       command line: --rig sim (PyBullet, known truth) or --rig real (LeRobot + cameras)
camera/charuco_board.py  the ChArUco board (8x6 squares, 15 mm, 11 mm markers)
tests/                   synthetic session and loop with a known truth (no PyBullet needed)
```

## Order of use

1. `python simulator/prepare_assets.py --out assets_so101`
2. Kinematics self-test: `python -m calib.urdf_fk assets_so101/so101_sim.urdf` -> PASSED
3. Synthetic tests with the real URDF (no PyBullet rendering, no hardware):
   `python tests/synthetic_session.py assets_so101/so101_sim.urdf`
   `python tests/synthetic_loop.py assets_so101/so101_sim.urdf` (geometric safety)
   `python tests/synthetic_loop.py assets_so101/so101_sim.urdf --workspace` (simulated teach-in)
4. Setup folder: `python setup_io.py create --dir setups/<name> ...` (see its docstring)
5. **Stage 1, teach the workspace by hand:** `python calib/workspace.py record --setup setups/<name>`
   (torque off after confirmation; cover the board area at all heights down to just above the board, with
   strongly varied wrist_flex and wrist_roll; 'q' + Enter saves); check with `calib/workspace.py show`
6. Simulation check (PyBullet renders the board; uses the taught workspace if present):
   `python calib/autocalib.py --rig sim --setup setups/<name> --out sim_check`
7. Real arm, first without moving: `python calib/autocalib.py --rig real --setup setups/<name> --board-cm 18 --dry-run`
8. **Stage 2, automatic calibration:** `python calib/autocalib.py --rig real --setup setups/<name> --board-cm 18`
   Default mode `replay`: the arm travels along the taught recording (a replay of the hand-taught motion, no
   planning), stops at ~60 recorded poses above the boards (jaws above --clearance-mm), takes the wrist image at
   each, estimates, and returns along the recording. Writes wrist_mount.json, the stereo pose and the joint
   offsets into the setup folder. `--mode planned` runs the older planner-based session.

Safety with a taught workspace: every pose and every move stays within the taught poses (moves follow paths
through recorded neighbours, shortened where a straight segment stays inside, executed as one continuous motion);
joints at least 8 deg inside their limits; the jaw tips and gripper body stay at least --clearance-mm (default 10)
above the lowest jaw point you taught, checked for every pose and along every move; a pose without a checked
route is skipped, never approached directly; moves at
most ~25 deg/s; torque on throughout; the arm returns to its rest pose at the end.

## Stereo camera vs arm: the gripper-object fit

1. `python calib/collect_gripper_object.py --setup setups/<name> --stops 150`: the arm holds a brightly coloured
   object (you insert it at the ready pose, the gripper holds it with a slight squeeze) and stops at poses of the
   taught workspace (clearance raised by --object-mm); raw stereo frame + joint readings at every stop.
2. `python calib/fit_gripper_object.py setups/<name>/gripper_object/<time> --setup setups/<name>`: the object's
   centre in both eyes (colour segmentation around the predicted position, or inside an LLMDet box with
   --detector llmdet), least squares over the stereo pose, the joint offsets and the object's position in the
   gripper; held-out errors in px and mm; comparison with the board-based calibration. --write stores the result.

## Localization with the new tasks (stage 2)

Stereo pair + instruction -> 4 joint angles (JSON), as in exp10, with:
- the printed objects (objects_v11: 7 shapes, white with weight 0.5) and 2-5 objects per scene;
- task types (tasks.py), each scene built so that the instruction refers to exactly one object:
  full ("the red cone"), colour only ("a violet object"), shape only ("the cylinder"),
  superlative ("the biggest cube", "compare the balls ..., choose the smallest ...", volume margin 1.35x);
- the target at each shape's grasp height (cones/pyramids low);
- the stereo camera centred on the setup's calibrated pose (pose_base), rectified 60.9 deg view;
  object sector +-45 deg. Appearance (board, backdrop) is still the v9 one.

```
simulator/so101_sim/env_v11.py   CellEnvV11 (task-driven objects), camera centre from pose_base, rectified HFOV
tasks.py                         task sampler and wording (synonyms: violet, grey, item/thing/piece/body, ...)
common.py                        prompt, labels, make_env for the cell
dataset/generate.py              dataset (rows carry task type, instruction, canonical form, all objects)
training/train.py                Qwen3.5 fine-tuning (unchanged from exp10)
evaluation/evaluate.py           metrics per task type and white/coloured targets, wrong-object rate
evaluation/analyze.py            simulated test splits: miss by direction, scene properties, superlatives
real/record_real.py              real test samples (setup folder: offsets, stereo rectification, taught workspace)
evaluation/analyze_real.py       real evaluation: by task type / shape / object count, miss by direction
run_v11.sh                       smoke test, dataset, training, evaluation
```

Cell geometry measured on the real arm (stored in the setup, used by the simulator):
```
real/probe_table.py      board height for the arm, torque on (table.json -> the simulator's object plane)
real/touch_table.py      the same by hand with the torque off (shows the play; not used for the simulator)
real/compare_sim.py      simulated vs real image from the calibrated camera pose (arm outline must coincide)
real/check_projection.py recorded and predicted points drawn into the real images
```
Scene: with a stereo_background.png in the setup, the simulator draws only the arm, the objects and their
shadows over that real picture (simulator/so101_sim/scene_real.py; warped with the board plane's homography for
randomised cameras). `dataset/generate.py --synthetic-scene` gives the old synthetic board and backdrop.
Image modes (common.IMAGE_MODES, `--image` of dataset/generate.py and real/record_real.py): `full` = the whole
view at 640x360; `crop2x` / `crop15` = the workspace window (arm's base and the board) at 2x / 1.5x the pixel
density (1024x576 / 768x432). The window is rendered directly; real images are cut from the raw camera frames,
which record_real.py always stores, so one real recording serves every mode (`--parquet-only --image crop2x`).
Painted jaw tip: since 16 Nov the last 20 mm of the real fixed jaw are painted black (outside, not the gripping
face); the simulator draws the same (`--tip-paint-mm`, default 20; 0 = the unpainted arm of earlier datasets).
Labels: the grasp point is 2.0 cm above the board for every object (fingertips ~8 mm clear); objects 12-30 cm
from the pan axis. Datasets store their cell in dataset_info.json; evaluate.py uses the same geometry.

Closed loop (second stage): one model call = one small step.
```
closed_loop.py               prompt / answer format, the label-making expert, pose sampler, servo model
dataset/generate_cl.py       stereo pair with the arm on its way + current angles -> change of the angles + gripper
evaluation/evaluate_cl.py    single steps, and whole approaches in the simulator (commands accumulate the changes;
                             shoulder sag and dead bands modelled); --policy expert checks the loop without a model
run_v11cl.sh                 smoke test, dataset, training (from the one-shot model), evaluation
```
The controller adds the model's change to the servo COMMAND (not to the reading): that integrates the servo
sag away, which a one-shot answer cannot.

Real check of a trained model:
```
python real/record_real.py --setup setups/<name> --out data_real_v11          # LeRobot environment
python evaluation/evaluate.py --model outputs/qwen35_08b_v11/final --data data_real_v11 --splits real \
    --out outputs/eval_v11_real.json                                          # model environment
python evaluation/analyze_real.py --eval outputs/eval_v11_real.json
```
