# Final Engineering Report — SO-101 Vision-Language Agentic Control

Implements `so101_opencode_task_specification.md`. Simulation-only; no hardware,
no fine-tuning.

## 1. Repository assets reused and modifications

- Reused `TheRobotStudio/SO-ARM100` (cloned to `assets/SO-ARM100`): the SO-101
  URDF `Simulation/SO101/so101_new_calib.urdf` and its STL meshes, plus
  `joints_properties.xml` (STS3215 ≈ 3.35 N·m). No robot model was rebuilt.
- Modifications: loaded with `useFixedBase=True` (this PyBullet build defaults
  to a free base, which makes the whole arm fall). Added a table, cubes, camera,
  a shadow collision world, and a compliant-grasp simplification (see §6).

## 2. Implemented architecture

A long-running **simulator daemon** owns PyBullet on one thread and exposes a
local HTTP API; the **MCP stdio server** and a **CLI** are thin clients. This
keeps the simulator independent of the agent harness and lets it outlive agent
sessions (spec 4.2). A tkinter window shows the same observation-camera frame
the LLM sees. Layer separation matches spec §5: simulation, robot_model,
camera, kinematics, motion_controller, safety, tools, experiment, config.
See `README.md` for the module table.

## 3. Launch and integration

```bash
DISPLAY=:1 python3 scripts/run_simulator.py --port 8311 \
    --experiment-id so101_demo --model-id deepseek-flash \
    --model-endpoint "deepseek (OpenAI-compatible endpoint)"
```
OpenCode launches `python -m so101.simulator_tools.mcp_server` per
`opencode.jsonc` (`mcp.servers.so101`); the tools appear as `so101.*`.

## 4. Agent-facing tool API

`get_observation`, `get_scene_status`, `move_to_pose(x,y,z)`,
`move_to_named_pose(name)`, `set_gripper(open|close)`, `wait_steps(n)`,
`reset_scene(seed?)`, `stop_motion`, `emergency_stop(engage)`,
`start_task(task, trial_id?, seed?)`, `end_task(notes?, failure_hint?)`.
`move_to_pose` targets the gripper jaw point (metres, tabletop `z=0`). Full
descriptions are in `so101/simulator_tools/mcp_server.py`.

## 5. Safety mechanisms

Joint-limit validation; whole-path collision checking in a separate shadow
PyBullet client (16 samples, 5 mm clearance); runtime contact monitoring every
20 ms; emergency stop; bounded wait/move/timeout limits. Safety is enforced
deterministically in the controller, independent of the LLM. Low targets use a
lift–translate–descend approach. Outcomes are verified post-execution from
telemetry.

## 6. Known limitations (documented simplifications)

- **Compliant grasp**: the stock URDF gripper shell does not pinch small cubes
  in this PyBullet build (verified: the jaws never contact both sides), so a
  rigid constraint attaches an object within `grasp_radius` when the gripper is
  closed (engaged before the jaws move). This is a simulator simplification, not
  a claim about the physical gripper.
- Self-collision not checked; table is a slab; base mounting mesh dips ~3 mm
  into the tabletop (excluded from checks).
- Per-trial `agent_iterations`, token and API-request counts are `null`
  (not measurable reliably from the simulator side).
- Image budget resets if the daemon is restarted (per-session state is
  in-memory).
- This PyBullet build is non-standard; the code documents and works around
  its differences (17-tuple `getJointInfo`, 5-value `getCameraImage`, no-op
  `TORQUE_CONTROL`, unreliable built-in IK, free-base default).

## 7. Demonstration result ("Grasp and lift the blue cube")

Run through OpenCode (multimodal): the agent obtained a camera image, identified
the blue cube, moved the gripper above it, closed, lifted, and re-observed.
Example (`results/so101_demo/demo_001.json`): `task_success=true`,
`maximum_lift_height≈0.19 m`, `object_retained=true`,
`verification_duration≈3.3 s`, `collision_count=0`.

## 8. Evaluation results (spec §8.2/§8.4)

11 trials, none excluded (`results/so101_demo/aggregate.json`):

| Harness | Trials | Success | Target ID | Grasp (given attempt) | Lift (given attempt) | Mean images | Collisions |
|---|---|---|---|---|---|---|---|
| OpenCode (LLM) | 3 | 3 (100%) | 100% | 100% | 100% | 1.0 | 0 |
| scripted-oracle (baseline) | 8 | 8 (100%) | 100% | 100% | 100% | 0 | 2 |

Failures are retained in the reports; the aggregate denominators are explicit
(grasp/lift success reported both given-attempt and overall). Trials used seeds
3, 7 and the default scene (varied cube positions); the scripted baseline varied
8 seeds. Each per-trial report follows §8.4.6.

Caveat: the OpenCode trials were orchestrated by the assistant acting as the
agent (via the MCP tools and the CLI); the recorded model id is
`deepseek-flash` per the operator-provided endpoint. The `scripted-oracle`
baseline is not an LLM trial; it demonstrates the controller/evaluator pipeline.

## 9. Verification

- `tests/test_m1_scene.py` — 13/13 (scene, camera, colored cubes, determinism).
- `tests/test_m3_motion_safety.py` — 9/9 (motion, limits, rejection, e-stop,
  grasp+lift, release, reset).
- `tests/test_m4_tools.py` — 16/16 (daemon, tools, budget exhaustion + handoff,
  trial report schema, e-stop).
- `tests/test_mcp_smoke.py` — 4/4 (mcp 2.x stdio: initialize, tools/list,
  tools/call).

## 10. Recommended next steps

- Replace the compliant-grasp simplification with a proper contact gripper once
  a URDF with usable jaw collision geometry (or a compliant contact model) is
  available.
- Run the LLM trials through fully autonomous fresh sessions (one per trial via
  the harness) to quantify success rate/variance across more seeds.
- Capture token/API metrics at the harness level and pass them into `end_task`.
- Enable self-collision and add camera calibration error for a more faithful
  study.
