# SO-101 Vision-Language Agentic Control (PyBullet)

A CPU-only PyBullet simulation of an SO-101 arm on a table, driven by a
multimodal LLM through an agent harness (OpenCode initially), for the task
**"Grasp and lift the blue cube."** This implements
`so101_opencode_task_specification.md`.

## What works

- Table + SO-101 (from the SO-ARM100 assets) + gripper + three colored cubes,
  front camera, live window.
- Deterministic IK, joint-limit and path-collision checks, runtime collision
  monitoring, emergency stop — independent of the LLM.
- Agent-facing tools (MCP + CLI) with a session image budget (8) and a text-only
  handoff on exhaustion.
- Simulator daemon that outlives agent sessions.
- Simulator-side evaluator + standardized per-trial JSON reports
  (`results/{experiment_id}/{trial_id}.json`) and an aggregate report.

## Architecture

```
  agent (LLM) --read image--> <tool call> 
       |                        |
   OpenCode harness        MCP stdio server (so101/simulator_tools/mcp_server.py)
       |                        |  (thin adapter, dies with the session)
       +------------------------+--> HTTP POST /call
                                     |
                     simulator daemon (scripts/run_simulator.py)
                     - owns PyBullet on ONE thread (sim loop)
                     - tkinter live window (separate thread) shows the
                       observation-camera frame (the same view the LLM sees)
                     - HTTP API (separate thread) enqueues commands
                     - is independent of any agent session (spec 4.2)
```

Keeping the simulator behind a local HTTP API makes it harness-independent:
the MCP server, the CLI (`so101/simulator_tools/cli.py`), and any future
harness (Pi, Goose) are all thin clients.

### Modules

| Path | Responsibility |
|---|---|
| `so101/config.py` | config loading/validation + `configuration_id` hash |
| `so101/simulation/world.py` | PyBullet lifecycle, scene, reset, stepping, shadow collision world, compliant-grasp constraints |
| `so101/robot_model/model.py` | joint/link discovery (FK-verified), limits, EE/gripper, jaw tool point |
| `so101/kinematics/fk.py` | exact URDF forward kinematics (verified vs PyBullet) |
| `so101/kinematics/ik.py` | scipy IK (position + approach axis, multiple candidates) |
| `so101/camera/camera.py` | RGB/depth/seg capture, PNG save |
| `so101/camera/display.py` | tkinter live window |
| `so101/motion_controller/controller.py` | IK + validation + execute, gripper, lift-translate-descend |
| `so101/safety/safety.py` | joint limits, path checks, runtime monitoring, e-stop |
| `so101/tools/core.py` | harness-independent tool service + image budget |
| `so101/simulator_tools/server.py` | daemon (HTTP + sim loop + window) |
| `so101/simulator_tools/mcp_server.py` | MCP stdio server (mcp 2.x) |
| `so101/simulator_tools/cli.py` | CLI client |
| `so101/experiment/trial.py` | evaluator + report (§8.4.6) |
| `scripts/run_simulator.py` | launch the daemon |
| `scripts/aggregate_report.py` | aggregate report (§8.4.7) |
| `scripts/run_scripted_trials.py` | oracle-baseline trials |

## Scene and camera

- Tabletop = plane `z = 0` (table 0.60 × 0.40 × 0.04 m, centre `[0.10, 0, -0.02]`).
- Robot base fixed at `[0.0, 0.12, 0.0]` (`useFixedBase=True` — required in this
  PyBullet build, which defaults to a free base).
- Cubes 4 cm, mass 0.02 kg: blue `[0.28, 0, 0.02]`, red `[0.20, -0.10, 0.02]`,
  green `[0.20, 0.10, 0.02]` (configurable; trials vary them by seed).
- Camera: eye `[0.52, -0.26, 0.42]`, target `[0.16, 0, 0.03]`, up `[0,0,1]`,
  640×480, FOV 60°, **0.591 m** from the workspace centre (within 50–70 cm).

## Agent-facing tool API

Exposed via MCP (`so101_*` or `so101.<tool>` in the harness) and the CLI:

| Tool | Description |
|---|---|
| `get_observation()` | one RGB frame; returns `image_path` (load it to see the scene); respects the budget |
| `get_scene_status()` | running/estop/last action/budget; no object ground truth |
| `move_to_pose(x,y,z)` | move the **gripper jaw point** to world (m); IK + limits + path checks |
| `move_to_named_pose(name)` | `home` or `ready` |
| `set_gripper(command)` | `open` / `close` (close engages the grasp) |
| `wait_steps(steps)` | advance the sim (≤ 6000 steps, 1 ms each) |
| `reset_scene(seed?)` | reset robot + cubes |
| `stop_motion()` | hold current position |
| `emergency_stop(engage)` | engage/clear e-stop |
| `start_task(task, trial_id?, seed?)` | begin a trial (evaluator tracking) |
| `end_task(notes?, failure_hint?)` | finalize + write the §8.4.6 report |

The model estimates cube positions from the camera image; the workspace and
tool-point semantics are stated in the tool descriptions.

## Determinism and safety

- **IK**: exact URDF FK (matches PyBullet to machine precision) + scipy
  `least_squares` on position + gripper approach-axis; multiple candidates are
  generated and the closest one with a collision-free straight joint path is
  executed (avoids sweeping IK branches).
- **Joint limits** checked before execution.
- **Path collision check**: the whole joint path (16 samples) is checked in a
  separate collision-only "shadow" PyBullet client (gravity off, static bodies),
  because this build's `getClosestPoints` only reflects a pose after a step.
  Table base-mount contact is ignored; all moving-link contacts count.
- **Runtime monitoring** every 20 ms during motion: unexpected contacts stop and
  hold the arm. Intended (grasped) objects and the target during grasp are
  exempt.
- **Emergency stop** (agent tool, window button) blocks all motion.
- Low targets are approached **lift–translate–descend** to avoid knocking
  objects over.

## Image budget and session rollover (spec 4.2)

- Max **8 image-bearing observations per agent session** (configurable).
- One RGB image per observation; no depth/seg/debug images to the LLM.
- On exhaustion: returns `image_path: null` plus a **text-only handoff summary**
  and saves state to `results/{experiment_id}/session_<id>_state.json`.
- Each MCP stdio process = one agent session = a fresh budget (the daemon keeps
  running). Restarting the MCP server is the rollover mechanism.

## Evaluation and reports

- The evaluator uses simulator telemetry (object z, gripper contact/attachment,
  retention, verification duration) — the LLM's self-assessment is never the
  sole truth.
- Success = target grasped and retained, lifted ≥ `minimum_lift_height`
  (0.05 m) and held for ≥ `verification_duration` (2 s of sim time).
- Per-trial report at `results/{experiment_id}/{trial_id}.json` (spec §8.4.6);
  aggregate at `results/{experiment_id}/aggregate.{json,md}` (spec §8.4.7).

## How to run

```bash
# 1. Start the daemon (live window on $DISPLAY; add --no-window for headless)
DISPLAY=:1 python3 scripts/run_simulator.py --port 8311 \
    --experiment-id so101_demo --model-id "<model>" --model-endpoint "<endpoint>"

# 2. In the agent harness, the MCP server is launched by opencode.jsonc
#    ("mcp"."servers"."so101"). It connects to http://127.0.0.1:8311/call.

# 3. Or drive it from a shell:
python3 -m so101.simulator_tools.cli status
python3 -m so101.simulator_tools.cli obs
python3 -m so101.simulator_tools.cli move 0.28 0.0 0.12
python3 -m so101.simulator_tools.cli grip close
python3 -m so101.simulator_tools.cli end "done"

# 4. Reports
python3 scripts/aggregate_report.py so101_demo
```

### Tests

```bash
python3 tests/test_m1_scene.py            # scene + camera + determinism
python3 tests/test_m3_motion_safety.py    # motion, safety, grasp/lift (no LLM)
python3 tests/test_m4_tools.py            # daemon + tools + budget + report
python3 tests/test_mcp_smoke.py           # MCP stdio JSON-RPC
```

## Assumptions and limitations (documented simplifications)

- **Gripper (compliant grasp).** The stock URDF gripper is a hollow shell that,
  in this PyBullet build, pushes small cubes away instead of pinching them
  (verified: the jaws never make two-sided contact). To make grasping testable,
  the controller models a firm/compliant grip: when the gripper is closed and
  an object is within `grasp_radius` of the jaw, a rigid constraint attaches it
  (attach happens *before* the jaws move, to avoid pushing). This is a
  documented simplification, not a claim about the real SO-101 gripper.
- **PyBullet build quirks** (Oct 2026 build): non-standard `getJointInfo`
  (17-tuple), `getLinkState` (6 values), `getCameraImage` (5 values, tuples not
  bytes), `calculateInverseKinematics` (8 positional args, unreliable → own IK),
  `TORQUE_CONTROL` is a no-op, `loadURDF` defaults to a free base. These are
  handled in code (`useFixedBase=True`, own IK, position control).
- The table is a slab (no legs); the base mounting mesh dips ~3 mm into the
  tabletop and is excluded from collision checks.
- Self-collision is not checked (the SO-101 self-collision setup is not
  configured in this build).
- `agent_iterations`, token counts, and API request counts are `null` in the
  reports (not measurable reliably from the simulator side).
- The image budget resets if the daemon is restarted (per-session state is
  in-memory).
