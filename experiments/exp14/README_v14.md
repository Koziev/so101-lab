# Experiment 14: simulator v14

The code is that of experiment 13 with a second simulator version, chosen by `--sim v14` (`record_approach.py`,
`eval_approach.py`, `inspect_policy.py`) or `common.make_env(..., sim="v14")`. The default stays `v11`, so the
exp11–13 runs are reproduced unchanged.

## Changes in v14

**Wrist camera on the arm** (`simulator/so101_sim/wrist_camera.py`, meshes in `simulator/assets/wrist_camera/`).
The physical robot gets two fixed links on `gripper_link`, written into a copy of the URDF next to the original:
- **`wrist_cam_mount`:** the printed head (frame + arm) and base, white, recoloured with the arm.
- **`wrist_cam_board`:** the 30 × 30 × 5 mm board, the 5 mm lens barrel and the thumb nut, black.

How they are placed and used:
- **Placement:** the camera's optical centre and axes come from the setup's `wrist_mount.json` (hand-eye calibration). The board sits behind the lens, the frame on the lens side. The head's rotation about the view axis (a multiple of 90°) and the side the frame faces are chosen so that the bolt at the end of the head's arm is closest to the gripper body. The choice is printed; `EnvConfig.wrist_cam_orientation` overrides it.
- **Contacts:** both links are links of the robot, so contacts with objects count as arm contacts, and contacts with the board count as board contacts (`approach.arm_contact`).
- **Collision shapes:** boxes and cylinders.
- **Unchanged parts:** the kinematics twin keeps the original URDF, so the tip, the goal and the IK are the same as before.

**Arm appearance** (`scene_real.sample_look`, preset `v14`). These values come from one real stereo frame
(22 Nov 2026) and are provisional:
- **Level:** the arm's typical lit face at 145–185 of 255, instead of 215–245.
- **Tint:** the camera's cyan cast on the arm (red × 0.90, ±2 % per channel).
- **Exposure:** 0.85–1.15 for the whole frame, without the occasional 1.1–1.3 boost that clipped the arm to white.
- **Brightness reference:** taken from the white parts only, excluding the black camera board.
- **Unequal eyes:** the real right eye is 8–10 % brighter than the left. The rendered arm and objects get a right-eye factor of 1.04–1.14 (the real backgrounds already have the difference), and each whole eye image varies by ±4 % on top, so the model also sees other and swapped ratios.

## Check

```bash
python tools/check_sim_v14.py --setup setups/2026-11_12v_stereo28mm_wrist --out outputs/check_v14
# with a real frame and its joint readings (real convention):
python tools/check_sim_v14.py --setup ... --real-left left.png --real-q "1.2,-98.0,95.3,70.1,-3.0,1.0"
```

Outputs:
- **`gripper_closeup.png`:** the assembly from six directions.
- **`stereo_*.png`:** v11 against v14 at rest, at the ready pose and at the real pose.
- **`overlay.png`:** the real frame with the simulated edges.

It also prints the camera's distance to the board in each pose, which must not be zero at rest. With a real frame, it prints brightness percentiles of the arm, real against v11 and v14.

**Object set** (`objects_v11.SETS["v14"]`, switched on by `make_env(sim="v14")`):

| Shape | Sizes (measured, widened by 1 mm) | Pose |
|---|---|---|
| cube | edge 27–37 mm | |
| cylinder | diameter 24–36 mm, height 29–36 mm | always upright |
| hex prism | across corners 27–36 mm, height 24–36 mm | always upright |

Colours: white, red, green, blue, yellow and pink. Prompts say "pink" or "magenta" for pink.
- **Source:** the real cubes' front faces in the stereo frame of 22 Nov (both eyes averaged). Each colour is divided by the white cube's, which removes the camera's cast, and scaled to the white albedo 0.93.
- **Camera cast:** the v14 look applies it to the arm and the objects alike.
- **Equal shares:** use `--white-weight 0.1667` to give white the same share as the other five colours.
- **Fluorescent colours:** the real red, green, yellow and pink are fluorescent. Red reaches 255 in the camera image and is clipped at albedo 1.

The object set belongs to the process: `make_env` sets it with the simulator version, so one process should use
one version (the recorder and the evaluation do).

## Recording with v14

```bash
python dataset/record_approach.py --setup $SETUP --out $DATA/data_v14a/raw --sim v14 --num 600 --workers 8 \
    --objects 1,1 --weights full=0.5,colour=0.25,shape=0.25 --white-weight 0.1667 \
    --random-start 0.5 --near-start 0.12 --long-hold 0.4
```

Leave out `--shapes`, or pass `cube,cylinder,hex_prism`: the v14 set has only these three shapes.

## Open

- **Brightness of the objects:** they share the arm's brightness scaling. A real frame with objects and joint readings would show whether white objects need their own level.
- **Arm brightness:** measured from one frame so far; refine with `--real-left`/`--real-q`.

## Commands (experiment 14)

`commands.py` defines five atomic commands, and `dataset/record_commands.py` records them. Each episode is one command:

| Command | Starts from (chain played without rendering) | Expert | Ends when |
|---|---|---|---|
| approach | rest / workspace / near-goal pose | as in v12–13; the wrist roll turns the jaws onto the object's faces (cube, hex prism) | tip 1.5 cm above the object, pointing at its centre |
| grasp | after an approach | shift sideways until the fixed jaw's face is 4–7 mm beside the object → open (55–85 %) → descend along the axis until the fixed jaw's tip is 6–10 mm above the board → close | the jaws are blocked by the object, plus 3 steps |
| lift | 70 % holding (after a grasp), 30 % open and empty (after a release) | tip straight up | tip 10 cm above the board |
| lower | 70 % holding (after a lift), 30 % open and empty | tip straight down | holding: object 4 mm above the board; empty: tip 1.25 cm above it |
| release | 70 % low (after a lower), 30 % lifted (the object drops) | gripper opens, arm still | gripper open |

- **Final hold:** every command ends with a hold of 0.8–1.2 s, or 2.0–2.4 s in 40 % of episodes, as in v12e.
- **Joining commands:** the expert keeps one nominal joint path and one sag correction across the whole chain, so commands join without jumps.
- **Failures:** an episode is kept only if every command of its chain succeeds (`commands.evaluate`).
- **Prompts:**
  - approach and grasp always name the object;
  - lift, lower and release name the held object half of the time, otherwise "it" or a plain gripper command ("lift", "open the gripper", ...).

```bash
SETUP=setups/2026-11_12v_stereo28mm_wrist ./run_v14.sh smoke    # 2 episodes per command + preview videos
./run_v14.sh record      # 100 per command
./run_v14.sh preview
./run_v14.sh export      # LeRobot env
./run_v14.sh train       # 1 epoch
```

`shard_info.json` in each shard counts the failed plans and failed expert runs per command.

### Evaluation

`evaluation/eval_commands.py` (stage `eval`) evaluates every command on the same new scenes at T0 and T1, plus the held-out recording seeds.
- **Start state:** each scene is prepared as in the recorder: the seed's start pose, then the chain leading to the command, played by the expert.
- **Run:** the policy gets the command's prompt and runs for a fixed number of steps per command (approach 150, grasp 160, lift 120, lower 140, release 75).
- **Success:** the recorder's test (`commands.evaluate`). The arm must also come to rest, which is reported separately.
- **Behaviour:** a table records what the arm actually did (open / close / rise / descend / move / still), which shows whether the policy tells the commands apart.
- **Expert baseline:** `--policy expert` runs the same harness with the scripted expert; it should succeed in nearly every scene.
- **Contact log:** each episode's `contacts` lists, per link, the first step and phase of every contact with the board or an object, the number of steps in contact and the deepest penetration. `plan` holds the expert's plan of the evaluated command (for a grasp: `tip_grasp_z`, `raised_m`, `board_clear_m`).

### Clearance of the whole gripper (v14b)

The contact log of the v14a expert showed that the tip, the point every height and distance was planned with, is neither the gripper's lowest point nor its widest: the fixed jaw grazed the board at the end of the grasp descent (8 of 21 grasps), and the jaws or the gripper body touched the object at the end of the approach descent (6 of 28 approaches). The expert now measures the arm's collision shapes at the nominal poses of its plan (`approach.pose_probe`: joints set with `resetJointState`, distances from `getClosestPoints`, joint states restored; no physics step):
- **Grasp height** (`commands._plan_grasp`): the grasp pose is raised until the gripper links, open and closed, keep `GRASP_BOARD_CLEAR` (3 mm) from the board; the grasp is not planned if that puts the tip higher than `GRASP_MIN_DEPTH` (1 cm) below the object's top.
- **Approach goal** (`approach.goal`): a tilt is skipped if at its goal pose any link of the arm is closer than `ARM_TARGET_CLEAR` (3 mm) to the target or the gripper closer than `GRIPPER_BOARD_CLEAR` (4 mm) to the board; the next tilt is tried.
- **Approach path** (`ApproachExpert._clear`): along the descent, the gripper links keep 4 mm from the board and every link 3 mm from the target; along the transit, which is not sag-corrected, 8 mm from the board (`TRANSIT_BOARD_CLEAR`), or no closer than at the start. Every link keeps 8 mm from the other objects. A low start that fails this first rises 4 cm along the vertical (`rise_pose`), then goes on directly or via the ready pose.
- **Rest start** (`start_from_rest`): a scene whose objects are within 5 mm of the folded arm is not used.

The plans differ from v14a only where the old plan would have come too close, so data recorded before this change does not match the new expert.

### Contacts and the sag correction (v14b, second change)

The contact log with signed distances showed two further causes, neither of them the planned geometry:
- **Separated points counted as contacts.** PyBullet's `getContactPoints` also returns points of surfaces that are apart (positive `contactDistance`, inside the contact breaking threshold). Half of the board "contacts" were 0.4–2.3 mm gaps. A contact now counts only with a negative distance (`approach.touching`, used by `arm_contact`, `commands.gripper_board_contact` and the recorder and evaluation loops); a jaw holds the object within 0.5 mm (`commands.HOLD_TOUCH`).
- **Overshoot at the end of descents.** The real touches (0.01–2.2 mm) happen at the end of the approach descent and the grasp / empty lower, although the plan keeps 3–9 mm. During a motion the readings lag the command (command delay, servo tracking), the integral sag correction integrated that lag, and the wound-up correction carried the arm past the goal. The correction now integrates only once the nominal pose has been still for `COMP_STILL_STEPS` (3) steps (`approach.nominal_still`); during a motion it keeps its last value.

Each evaluation record has `clear_mm`: the smallest physical clearance per phase of the gripper to the board and of the arm to the target.

### Clearances of the other commands (v14b, third change)

With contacts counted on penetration and the overshoot gone, the expert reached approach 27/28, grasp 24/27, lift 16/16, lower 13/17, release 17/19. The physical clearance log (`clear_mm`) located the rest:
- **Grasp shift.** At the end of the sideways shift or the opening, the arm came within 0–2 mm of the object, in two cases into it (the object was pushed 4 cm away once). The shift and the opening (half and fully open) are now checked against the object (`GRASP_SHIFT_CLEAR`, 4 mm); if they come closer, the shift target is moved back along the gripper axis by 4–16 mm (`GRASP_SHIFT_RAISE`), which keeps the fixed jaw's offset to the object; the grasp descent starts from there. `plan.shift_back_m` records it.
- **Empty lower.** After a release and a lift, the open gripper came back down onto the object it had released (1–2.7 mm into it, stuck 3–4 cm above the board). Lift and lower now check the nominal path against every object except a held one (`VERTICAL_OBJ_CLEAR`, 3 mm, or no closer than at the start); such a scene is not used for an empty lower.
- **Held lower, then release.** Two lowers left the gripper 0.3–0.6 mm above the board, and the release that followed touched it. A lower now also stops with the gripper links `LOWER_GRIPPER_CLEAR` (4 mm) above the board (physical distance when planned).
- **Transit from a low start.** A workspace start dragged the gripper along the board. A start that does not lean on the board must now gain `TRANSIT_LIFT` (3 mm) of board clearance after `TRANSIT_GRACE` (8) steps; otherwise the transit rises first.

### Sag correction during motion, physical lower targets (v14b, fourth change)

Expert run after the third change: approach 27/28, grasp 25/27, lift 16/16, lower 8/11, release 18/18.
- **Held lower.** Three held lowers ended with the gripper or the object on the board (object 0.1 mm above it or pressed in). The drop was measured on the physical arm but applied to the nominal tip; the correction then brought the arm onto the nominal pose, i.e. lower by the 3-4 mm the arm had not caught up when the lower was planned. The target is now the physical tip minus the drop (`commands._plan_lower`).
- **Empty lower.** The object check of the third change left 1 of 7 empty lowers: after a release and a lift the object stands right below the open jaws. An empty lower now moves its end point sideways, away from the object, by 3-12 mm (`VERTICAL_SIDESTEP`) until the path keeps 3 mm from it (`plan.sidestep_m`).
- **Grasp shift and transit.** Two grasps still hit the object during the shift although the plan kept 4 mm (physical clearance -2.1 and 1.4 mm), and one transit still touched the board. With the correction frozen during motions, the servo dead band (a joint stops several degrees short in one direction of motion) is not compensated while moving, and the gripper leaves its path. The correction now also integrates during a motion, against the nominal pose `COMP_LAG_STEPS` (4) steps earlier (`approach.comp_reference`), which leaves out the command delay and the tracking lag that caused the overshoot. The transit still has no correction.
- **Diagnostic.** `track_mm` per phase: the largest distance between the physical tip and the expert's nominal tip of the step, and the largest drop below it (includes up to a few steps of motion lag while moving).

### Weighted dataset composition (v14c)

`--num-per-command` (`NUM_PER_COMMAND` in run_v14.sh) takes one number for every command, or a total per command:
`approach=2000,grasp=1500,lift=500,lower=500,release=500` (5,000 episodes; approach as many as exp12's v12e). Seeds are mapped to commands by a fixed cycle in proportion to the totals (here 10 seeds: 4 approach, 3 grasp, 1 lift, 1 lower, 1 release), so all commands progress together and an interrupted recording resumes; with equal totals the cycle is the command order, i.e. the same seeds as before. Each command's total is split over the shards.
