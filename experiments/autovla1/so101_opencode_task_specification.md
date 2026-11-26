# Task Specification: Vision-Language Agentic Control of an SO-101 Robotic Arm in PyBullet

## 1. Objective

Implement an agent-based system in which a **TUI-controlled agent harness**
(orchestrator) connects a multimodal LLM to a simulated SO-101 robotic arm
through natural-language instructions, camera images, and bounded simulator tools.

OpenCode is the initial harness. The implementation should keep the simulator
interface sufficiently independent of the harness that alternative harnesses
(e.g. Pi Coding Agent or Goose) can be evaluated without changing the simulator,
robot controller, task, or evaluation logic.

The system must test the following hypothesis:

> A general-purpose multimodal LLM, accessed through an
> OpenAI-compatible API and orchestrated by the selected agent harness, can perform basic
> object-manipulation tasks with an SO-101 arm without additional
> fine-tuning, using visual observations and text instructions to plan
> and execute actions.

A representative instruction is:

> "Grasp and lift the blue cube."

The agent must perceive the scene, identify the target object, determine
an appropriate manipulation strategy, operate the simulated arm, observe
the outcome, and recover or report failure when necessary.

This is a **simulation-only research prototype**. Do not implement
control of a physical robot or send commands to real hardware.

## 2. Experimental setup and constraints

### 2.1 Host machine

-   Operating system: Linux (use the current environment).
-   CPU-only machine; no GPU is available.
-   RAM: 32 GB.
-   An initial agent harness (OpenCode) is already installed and configured.
-   The initial harness already has access to the selected LLM.
-   The LLM is exposed through an OpenAI-compatible API.
-   The API/model supports text and image inputs and the tool-calling
    behavior required by the selected harness.
-   The initial harness is OpenCode; the simulator interface and evaluation
    pipeline should also permit later trials with alternative harnesses.
-   PyBullet is already installed in the current Python environment.

Do not spend time installing or configuring another LLM, changing the
model endpoint, or provisioning GPU support. Reuse the existing model,
endpoint, and environment when testing alternative harnesses.

### 2.2 Robot and simulation resources

Use **PyBullet** as the physics engine.

The SO-101 simulation resources are available in:

-   Repository: https://github.com/TheRobotStudio/SO-ARM100
-   Simulation directory:
    https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation

Inspect the repository before implementing the robot model. Reuse its
SO-101/SO-ARM100-compatible URDF, meshes, configuration, and other
relevant assets where practical. Do not recreate the robot model from
scratch unless the supplied resources prove unusable.

The simulator must contain:

-   A table with a clearly defined tabletop surface.
-   An SO-101 arm mounted in a plausible position on the table.
-   A gripper that can open and close.
-   Several colored cubes placed on the tabletop in front of the arm.
-   A single front-facing camera viewing the tabletop and the arm
    workspace.

The camera should be positioned approximately **50--70 cm from the
workspace**, with a view that makes the robot, tabletop, and target
cubes visible. Choose practical camera parameters (position,
orientation, field of view, resolution) and document them. The exact
distance may be adjusted to obtain a useful image.

### 2.3 Initial object set

Use simple, visually distinguishable objects for the first experiment:

-   Blue cube.
-   Red cube.
-   Green cube.
-   Optionally, additional cubes or other simple objects after the basic
    experiment works.

Use contrasting colors and a table material that makes object
segmentation and visual identification reasonably straightforward. Keep
object dimensions, mass, friction, and initial positions configurable.

The initial scene should be deterministic or reproducible through a
configurable random seed. It should be possible to reset the scene and
restore the initial robot and object states.

### 2.4 Headless operation and visual evidence

The complete system must operate in a **headless, terminal-only
environment**, including remote SSH sessions without a desktop session,
X server, display server, or GUI forwarding.

- Run PyBullet in `DIRECT` mode. Do not start PyBullet in `GUI` mode.
- Do not create windows, GUI widgets, live camera displays, or interactive
  visualization windows.
- Do not require a human to watch a live simulation stream.
- Camera observations must still be captured programmatically and supplied
  as RGB images to the multimodal LLM through the agent harness.
- Record every grasping attempt as an MP4 video, as specified in section
  7. The video is the persistent visual evidence for later inspection.
- Provide terminal logs and machine-readable reports for monitoring,
  debugging, and evaluation.

Do not add a GUI fallback. A headless run must be the normal and tested
execution path, not an optional mode.

## 3. Functional requirements

### 3.1 Simulation initialization

Implement a standalone simulation module responsible for:

1.  Starting PyBullet in headless `DIRECT` mode. GUI mode must not be
    used or required.
2.  Loading the table and the SO-101 model.
3.  Identifying the arm's actuated joints, gripper joints, end-effector
    link, and relevant fixed links.
4.  Configuring joint limits and reasonable motor/control parameters.
5.  Adding the colored cubes.
6.  Creating and configuring the camera.
7.  Providing a reset operation that restores a known initial scene.
8.  Providing a simulation stepping mechanism that can be called by the
    controller.

Do not assume joint indices from memory or hard-code them without
checking the loaded URDF. Resolve joints and links by name where
possible, and validate that all required joints and links have been
found.

Log the resolved joint names, indices, limits, and end-effector link at
startup.

### 3.2 Camera and observations

Implement a camera interface that can:

-   Capture an RGB image from the configured front-facing camera.
-   Optionally capture a depth image and segmentation mask for debugging
    and internal geometric reasoning.
-   Return the RGB image in a format that can be passed to the
    multimodal LLM through the selected agent harness.
-   Save selected observations to disk for experiment reproducibility.
-   Provide timestamps or step identifiers so that observations can be
    associated with actions.

The LLM's primary visual input should be the **camera RGB image**, not
privileged simulator state. Do not expose exact object positions, object
identities, ground-truth segmentation, or robot joint states to the LLM
by default.

Depth images, segmentation masks, object coordinates, and other
simulator ground truth may be used internally by deterministic control
and safety components. Keep these channels separate from the LLM
observation interface. If an experiment uses privileged information,
record that fact explicitly.

### 3.3 Agent interface

The selected agent harness must be able to interact with the simulator through a small,
documented set of tools or callable Python functions.

Prefer a clean interface with operations such as:

-   `get_observation()`: capture and return the current camera image.
-   `get_scene_status()`: return a concise status report, such as
    whether the simulation is running and whether the robot is in a
    fault or emergency-stop state. Do not include hidden object ground
    truth in the default agent-facing response.
-   `move_to_pose(...)`: request a Cartesian end-effector target pose.
-   `move_to_named_pose(...)`: move to a predefined safe pose, such as
    `home` or `pre_grasp`.
-   `set_gripper(...)`: open or close the gripper, or set a supported
    opening value.
-   `wait_steps(...)`: advance the simulation for a bounded number of
    steps.
-   `reset_scene()`: restore the initial experiment state.
-   `stop_motion()`: stop or safely hold the robot.

These are conceptual operations; adapt names and signatures to the
implementation. Document the final tool/API schema and provide examples.

**Important:** The LLM should issue high-level, bounded requests. It
should not directly write arbitrary joint torques, set arbitrary joint
positions, modify the physics world, or bypass the safety layer. The
deterministic controller should translate validated Cartesian requests
into joint-space commands.

Use the selected harness's supported Python execution, custom-tool, or MCP
mechanism to expose the simulator interface. Keep the simulator API independent
of harness-specific message formats wherever practical. Keep
the interface simple enough for the agent to discover and use reliably.
Do not assume undocumented harness capabilities; inspect the installed
setup and use its supported mechanisms.

### 3.4 Motion control

Implement a deterministic motion-control layer for the SO-101.

At minimum, it should support:

-   Moving the end effector to a requested Cartesian position and, where
    feasible, orientation.
-   Opening and closing the gripper.
-   Moving through a sequence of waypoints.
-   Returning to a safe/home pose.
-   Stopping or holding motion.
-   Reporting whether a requested movement succeeded, failed, or was
    rejected.

Use inverse kinematics supported by PyBullet or another lightweight
deterministic method compatible with the existing environment. Respect
joint limits and validate the IK result before execution.

Motion should be executed incrementally rather than by instantly
teleporting the robot to the target. Use position control or another
stable control method, with bounded target changes and simulation
stepping.

The initial implementation should favor reliability and observability
over speed. Avoid aggressive acceleration, large instantaneous
movements, and unstable control gains.

The controller should return useful status information, including:

-   Whether a target pose was reachable.
-   Whether the target was rejected by a safety check.
-   Whether the movement completed within the allowed tolerance and
    timeout.
-   Whether a collision or other fault was detected.

Do not report success merely because a command was accepted. Verify the
resulting state after execution.

### 3.5 Safety and collision avoidance

Safety is a core requirement, even though the experiment runs only in
simulation.

The robot must not collide with:

-   The table or its supporting structure.
-   Cubes or other scene objects, except for intended contact during
    grasping.
-   Its own links, where self-collision checking is supported and
    configured.

The intended grasp contact between the gripper and the selected target
cube is allowed. Contact between other robot links and the cube is not
an intended part of the manipulation.

Implement a safety layer that validates proposed movements before and
during execution. At minimum:

1.  Check joint limits before accepting a target.
2.  Check the proposed path for collisions, not just the final pose.
3.  Check the robot's state during execution.
4.  Stop or hold motion if an unexpected collision or unsafe condition
    is detected.
5.  Reject targets that cannot be reached safely.
6.  Bound movement duration, waypoint count, and requested simulation
    steps.
7.  Provide an emergency-stop operation callable by the agent and the
    human operator.

Use PyBullet collision queries and collision geometry wherever possible.
Configure collision filtering carefully so that intended gripper-object
contact can be distinguished from forbidden contacts. Document any
limitations in the collision model.

Do not rely exclusively on the LLM to avoid collisions. **Safety
enforcement must be deterministic and independent of the LLM's
reasoning.** The LLM may propose an action, but the controller must
validate it before execution.

If a safe path cannot be found, do not execute an unsafe approximation.
Return a clear error and let the agent re-observe or choose another
strategy.

## 4. Agent behavior and manipulation workflow

The agent should follow an observe--plan--act--observe loop. It must use
fresh camera observations to ground its decisions and verify task
completion.

A representative workflow for "Grasp and lift the blue cube" is:

1.  **Observe:** Request a camera image.
2.  **Interpret:** Identify the blue cube and estimate its location and
    spatial relationship to the robot. Distinguish it from other cubes.
3.  **Plan:** Choose a safe approach direction, pre-grasp pose, grasp
    pose, gripper action, and lift direction.
4.  **Validate:** Submit movement requests to the deterministic
    controller, which performs IK, joint-limit, and collision checks.
5.  **Approach:** Move to a safe pre-grasp pose above or beside the
    target.
6.  **Descend:** Move toward the grasp pose using short, validated
    movements.
7.  **Grasp:** Close the gripper using the configured grasp procedure.
8.  **Verify grasp:** Obtain a fresh camera observation and assess
    whether the cube appears to be held. Where useful, the simulator may
    also provide an internal grasp/contact check to the controller, but
    do not silently expose this privileged signal to the LLM.
9.  **Lift:** Move the grasped cube upward by a configurable distance
    while maintaining a safe trajectory.
10. **Verify completion:** Obtain another camera observation and
    determine whether the cube was lifted and remains held.
11. **Report:** State whether the task succeeded, failed, or could not
    be verified. Include a concise explanation.

The workflow is an example, not a hard-coded sequence that must be used
for every task. The agent should be able to adapt based on visual
observations and action results.

### 4.1 Closed-loop behavior

Do not let the agent generate a long sequence of unverified servo
commands and execute it blindly.

Use short action segments with observations between meaningful stages.
Re-observe after approaching the object, after closing the gripper, and
after lifting. If the scene differs from expectations, the agent should
stop, inspect the new image, and re-plan.

Set explicit limits on:

-   Number of agent/tool iterations per task.
-   Number of movement requests per task.
-   Maximum duration of a movement request.
-   Maximum simulation steps per wait operation.
-   Overall task timeout.

These limits should be configurable. When a limit is reached, stop
safely and return a clear status rather than continuing indefinitely.

### 4.2 Image Budget and Conversation Management

Image inputs are a limited resource. The model endpoint may impose a maximum
number of images per API request. In addition, an agent harness may retain
images from earlier tool interactions and include them in later requests.

The agent and harness integration must therefore minimize unnecessary image
transmission and provide an explicit session-management policy.

#### Image handling rules

1. **One image per observation.** Each agent-facing observation must contain
   exactly one RGB camera image. Do not attach depth maps, segmentation masks,
   debug renderings, or additional camera views to the same observation unless
   a separately documented experiment explicitly requires them.
2. **Avoid redundant observations.** Request a new image only when it is
   needed to make a decision or verify a meaningful action. Reuse the latest
   observation when no meaningful scene change has occurred.
3. **Observe at meaningful milestones.** Prefer observations at task
   initialization, after approach, after closing the gripper, after lifting,
   and after recovery from an unexpected situation.
4. **Do not repeatedly inspect unchanged scenes.** If the latest observation
   is sufficient to choose a bounded action, proceed rather than requesting
   another image solely for reassurance.
5. **Avoid unnecessary image-file inspection.** Do not load saved RGB, depth,
   or segmentation files into the model conversation unless inspection of
   those particular files is required.

#### Session image budget

Use the following conservative initial limits, configurable for subsequent
experiments:

- Maximum of **8 image-bearing observations per agent session**.
- Maximum of **1 RGB image per observation**.
- The harness integration must prevent the accumulated conversation history
  from exceeding the configured endpoint image limit. Counting only newly
  requested observations is insufficient if earlier images remain in history.

When the session image budget is exhausted:

1. Stop requesting additional camera observations.
2. Save the experiment state, action history, relevant observation paths, and
   current trial progress to disk.
3. Produce a concise text-only handoff summary containing the task, current
   simulator status, target object, completed actions, outstanding actions,
   latest known result, and relevant failures.
4. Start a new agent session and continue using the handoff summary.
5. Obtain a fresh camera observation in the new session before making
   perception-dependent decisions.

Do not assume that context compaction removes old image attachments or resets
the accumulated image count. Session rollover must be implemented or verified
at the harness level. The simulator must remain independently running so that
starting a new agent session does not reset the robot or scene.

#### Failure handling

If the harness or model endpoint reports an image-count or image-payload error:

1. Stop the current agent workflow; do not repeatedly retry the same
   conversation history.
2. Preserve simulator state, logs, and the failed session identifier.
3. Record the failure as a harness or API failure, as appropriate.
4. Start a fresh agent session with a text-only handoff summary.
5. Resume only after verifying that a single fresh RGB observation can be
   transmitted successfully.

The image budget applies to the complete request history actually sent to the
model, not merely to the number of images in the latest tool result.

### 4.3 Natural-language task input

Provide a simple way for a human to submit a task in natural language.
Initially, a command-line prompt is sufficient.

Examples:

-   "Grasp and lift the blue cube."
-   "Move the red cube to the left side of the table."
-   "Pick up the green cube and place it near the blue cube."

Only the first task is required for the initial end-to-end milestone.
The other examples illustrate possible extensions and should not delay
implementation of the basic experiment.

The agent must ask for clarification if a task is ambiguous in a way
that prevents safe execution. It must not invent a target object when it
cannot identify the requested object reliably.

## 5. Software architecture

Use a modular architecture. A suggested separation is:

  -----------------------------------------------------------------------
  Component                           Responsibility
  ----------------------------------- -----------------------------------
  `simulation`                        PyBullet lifecycle,
                                      robot/table/object loading,
                                      stepping, reset

  `robot_model`                       Joint/link discovery, limits,
                                      end-effector and gripper
                                      configuration

  `camera`                            RGB/depth capture, image
                                      conversion, frame timestamps

  `kinematics`                        IK and Cartesian pose handling

  `motion_controller`                 Waypoints, interpolation, motor
                                      commands, completion checks

  `safety`                            Joint-limit validation, collision
                                      checks, runtime stop conditions

  `simulator_tools`                   Small agent-facing API exposed to
                                      the selected harness

  `agent_runner`                      Task submission, observation/action
                                      loop, iteration limits

  `logging`                           Structured action, observation,
                                      error, and experiment logs

  `config`                            Camera, robot, objects, controller,
                                      safety, and experiment parameters
  -----------------------------------------------------------------------

The exact file layout may differ, but keep simulation, control, safety,
and agent orchestration logically separate. Avoid a monolithic script.

Use the current Python/Conda environment and its already installed
dependencies.

**Do not install, upgrade, downgrade, or remove Python packages or other
software autonomously.** Before implementation, inspect the active Python
interpreter and check whether all required Python packages and video
encoding tools are already available.

If any additional Python package is required, stop implementation at that
point and ask the user to install it in the intended environment. State
the exact package name, why it is needed, and (if known) the compatible
version. Do not run `pip install`, `conda install`, or equivalent package
installation commands, even if installation appears to be user-local.

Prefer functionality available in the existing environment. For MP4
recording, check for an already available encoder such as the `ffmpeg`
executable and/or an already installed Python video-writing package. Do
not install an encoder or package. If MP4 encoding cannot be implemented
with the tools already available, pause and ask the user how to proceed.

Do not install or modify system-level software. Continue only after the
user confirms that any required dependency has been installed and verified.

## 6. Configuration

Put tunable parameters in a configuration file rather than scattering
constants through the code.

At minimum, configure:

-   Paths to the SO-101 URDF and mesh assets.
-   Physics time step and gravity.
-   Table dimensions and pose.
-   Robot base pose.
-   Cube dimensions, masses, colors, and initial positions.
-   Camera position, orientation, resolution, and field of view.
-   Joint control gains and movement speed limits.
-   IK settings and tolerances.
-   Gripper open/close targets.
-   Safe/home pose.
-   Collision-checking settings.
-   Task and tool-call timeouts.
-   Maximum agent iterations and action limits.
-   Random seed.
-   Video recording frame rate, codec, pixel format, and quality/bitrate
    settings, using options supported by the encoder already available.
-   Video output subdirectory and timestamp convention.

Validate configuration values at startup and provide clear errors for
invalid paths or inconsistent parameters.

## 7. Logging and experiment reproducibility

Record enough information to reproduce and diagnose each run.

For every task, log:

-   Experiment/run identifier and timestamp.
-   Initial scene configuration and random seed.
-   Natural-language task instruction.
-   Images actually supplied to the LLM (or stable paths to saved images),
    including the number of images in each model request where measurable.
-   Agent decisions and tool requests, subject to the selected harness's
    logging facilities.
-   Controller validation results.
-   Executed waypoints or target poses.
-   Joint states and relevant collision events for internal debugging.
-   Gripper commands and grasp-verification results.
-   Final task status and failure reason, if any.
-   For every grasping attempt: unique attempt identifier; attempt start
    and end timestamps; video file path; final filename; objective
    `success`/`failure` label; and any recording/encoding errors.

### 7.1 MP4 recording for every grasping attempt

Record **each individual grasping attempt** as a separate MP4 video. If a
trial includes retries, each retry must have its own recording. A trial
with no actual grasp attempt does not need an attempt video; record that
no attempt was made in the trial report.

- Store videos under
  `results/{experiment_id}/{trial_id}/videos/`.
- Capture frames from the configured PyBullet camera using the same
  viewpoint as the RGB observations supplied to the LLM.
- Record continuously from immediately before the attempt's first
  manipulation action through the attempt's final verification or
  termination. Include the approach, descent, gripper closing, lift,
  verification, and any recovery actions belonging to that attempt.
- Record at a configurable, documented frame rate. Use a streaming
  encoder/writer where possible; do not accumulate an unbounded sequence
  of full-resolution frames in RAM.
- Do not open a window or display the recording. Recording must work in
  headless `DIRECT` mode over a terminal-only SSH session.
- Use UTC timestamps in filenames and include an attempt identifier to
  avoid collisions. The final filename must include both the timestamp
  and the objective outcome label, for example:
  `20261123T075501.123456Z_attempt_001_success.mp4`
  or
  `20261123T075501.123456Z_attempt_002_failure.mp4`.
- The timestamp is the attempt start time. Finalize/rename the file only
  after the simulator-side evaluator has determined the attempt outcome.
  Use `success` only when the attempt meets the objective grasp-and-lift
  success criteria. Use `failure` for unsuccessful, aborted, timed-out,
  or unverified attempts. Do not use the LLM's self-assessment as the
  filename label.
- Write to a temporary filename while recording, then atomically rename
  to the final timestamp-and-label filename when recording is finalized.
  Handle normal shutdown and interruption so that a recoverable partial
  recording is finalized with a `failure` label and clearly marked as
  incomplete in the trial report. Never label an incomplete recording
  `success`.
- Validate that each MP4 file exists, is non-empty, and can be opened or
  probed by an available tool. Record validation status and any error in
  the trial report. Do not silently omit a failed recording.
- If video encoding is unavailable with the software already installed,
  stop and ask the user before installing or adding any dependency.

Keep the distinction between **agent-visible observations** and
**privileged simulator/debug information** explicit in the logs.

Avoid logging API keys, credentials, or other secrets. Do not hard-code
the LLM API key in source files.

## 8. Evaluation protocol

The first goal is a reliable end-to-end demonstration, followed by
repeatable trials.

The harness is an experimental variable. When comparing OpenCode, Pi, Goose,
or other harnesses, keep the model identifier and endpoint, task specification,
simulator and robot configuration, camera settings, safety controller, tool
semantics, initial scene, random seeds, and task timeout constant wherever
possible. Record unavoidable differences (for example, harness-specific tool
schemas or context policies) rather than silently treating the conditions as
identical.

Use the same simulator-side evaluator and report generator for every harness.
Do not let each harness independently define or calculate task success.

### 8.1 Initial milestone

Demonstrate that the agent can:

1.  Start the simulator in headless PyBullet `DIRECT` mode from a
    terminal-only session.
2.  Receive the instruction "Grasp and lift the blue cube."
3.  Obtain a camera image through its tool interface without opening a
    display window.
4.  Identify the blue cube from the image.
5.  Plan and execute a safe grasp-and-lift attempt.
6.  Record the complete attempt as an MP4 file in the designated videos
    subdirectory.
7.  Use new observations to verify the outcome.
8.  Finalize the video filename with a timestamp and objective success
    label, and report the result accurately.

The LLM must use its existing general-purpose capabilities. Do not
fine-tune, train, or otherwise adapt model weights for this experiment.
Prompt engineering and tool descriptions are allowed, but record their
versions so that trials are comparable.

### 8.2 Repeatability

After the initial demonstration works, run multiple trials. Reset the
simulation between trials and vary cube positions within a predefined
reachable workspace. Optionally vary the positions of non-target cubes
as distractors.

Record at least:

-   Task success rate.
-   Correct target identification rate.
-   Successful grasp rate.
-   Successful lift rate.
-   Collision/unsafe-motion count.
-   Number of agent iterations and tool calls.
-   Task completion time.
-   Failure category.

Define success objectively. For the initial grasp-and-lift task, success
should require that the target cube is visibly lifted above the tabletop
by a configured minimum height and remains held for a configured
verification interval. Do not count merely touching or pushing the cube
as success.

Report both aggregate results and per-trial outcomes. Do not hide failed
trials or exclude them without documenting the reason.

### 8.3 Ablations and extensions

Do not implement these before the basic system is operational. They may
be useful for later experiments:

-   Fixed versus randomized object positions.
-   One target cube versus multiple distractor cubes.
-   RGB-only agent observations versus optional depth-assisted internal
    control.
-   Different camera viewpoints.
-   Different prompts or tool descriptions.
-   Different LLMs, while keeping the task and evaluation protocol
    fixed.
-   Pick-and-place rather than grasp-and-lift.


### 8.4 Standardized Trial Reporting

Every manipulation trial must produce a standardized machine-readable report.
The report schema must be independent of the agent harness so that results
from OpenCode, Pi, Goose, and other harnesses can be compared directly.

The report must be generated by shared experiment/evaluation infrastructure
using simulator telemetry and recorded events wherever possible. The LLM's
self-assessment must not be the sole source of truth for task success.

#### 8.4.1 Trial identification and configuration

Each report must contain:

- `experiment_id`: identifier of the overall experiment.
- `trial_id`: unique trial identifier.
- `timestamp`: trial start time in ISO 8601 format, including timezone.
- `harness.name`: agent harness name (e.g. OpenCode, Pi, Goose).
- `harness.version`: harness version, or `null` if unavailable.
- `model.id`: exact model identifier.
- `model.endpoint`: provider identifier, excluding credentials.
- `task`: exact natural-language instruction.
- `random_seed`: simulator random seed.
- `initial_scene`: initial object positions and relevant scene configuration,
  or a path/reference to a versioned scene record.
- `configuration_id`: identifier or hash of simulator, camera, controller,
  safety, and task configuration.
- `recordings`: an array with one entry for every grasping attempt. Each
  entry must include `attempt_id`, `start_timestamp` (UTC ISO 8601),
  `end_timestamp` (UTC ISO 8601, or `null` if unavailable), `video_path`,
  `filename`, `outcome_label` (`success` or `failure`),
  `recording_complete`, and `video_validation_status`. A trial with no
  grasp attempts must contain an empty array.

The filename must contain the attempt start timestamp and its objective
outcome label, for example
`20261123T075501.123456Z_attempt_001_success.mp4`. The label must agree
with the simulator-side evaluation for that attempt. An interrupted or
unverified attempt must be labelled `failure`.

#### 8.4.2 Manipulation outcome

Report these outcomes independently:

- `target_identified`: whether the agent correctly identified the requested
  object.
- `approach_completed`: whether the robot reached the pre-grasp configuration.
- `grasp_attempted`: whether a grasp attempt was actually executed.
- `grasp_success`: whether the target was grasped and retained.
- `lift_attempted`: whether a lift was actually attempted.
- `lift_success`: whether the target met the objective lift criteria.
- `task_success`: whether the complete task met the objective success criteria.

Use `true`, `false`, or `null` where appropriate. `null` means the outcome
could not be established or the corresponding stage was not reached. Do not
treat an unattempted operation as a failed operation. Include an explanation
for `null` values in the failure description or trial notes.

#### 8.4.3 Objective grasp and lift verification

Determine grasp and lift outcomes using deterministic simulator telemetry
whenever possible.

A successful grasp requires:

1. Contact between the target cube and the intended gripper contact surfaces.
2. Evidence that the cube is retained by the gripper after closing.
3. Evidence that the cube is not merely resting on the tabletop or being
   pushed by the gripper.

A successful lift requires:

1. The target cube's vertical displacement relative to its initial position
   is at least the configured `minimum_lift_height`.
2. The cube remains above that threshold for at least the configured
   `verification_duration`.
3. The cube remains retained by the gripper throughout the verification
   interval.

Record these measurements:

- `initial_cube_position`: target cube position before manipulation.
- `final_cube_position`: target cube position at the end of the trial.
- `maximum_lift_height`: maximum vertical displacement relative to the
  initial position.
- `minimum_lift_height`: configured minimum vertical displacement required
  for success.
- `verification_duration`: actual duration for which the lift condition was
  maintained.
- `grasp_contact_verified`: whether intended gripper contact was detected.
- `object_retained`: whether the cube remained mechanically held during
  verification.

If a reliable internal grasp check is unavailable, mark the relevant fields
as unknown (`null`) rather than assuming success from the agent's description.
Privileged simulator telemetry may be used by the evaluator but must not be
exposed to the LLM during the experiment unless a separately documented
ablation explicitly permits it.

The exact contact and retention tests must be implemented and documented by
the simulator. Do not claim that contact alone proves a successful grasp.

#### 8.4.4 Efficiency and resource consumption

Record:

- `duration_seconds`: total trial duration.
- `agent_iterations`: number of agent reasoning/turn iterations, where
  available.
- `tool_calls`: total number of tool calls.
- `observation_count`: number of camera observations requested.
- `image_count`: number of images actually transmitted to the model.
- `input_tokens`: total input tokens, if reported reliably by the provider.
- `output_tokens`: total output tokens, if reported reliably by the provider.
- `api_requests`: number of model API requests.
- `api_errors`: number of failed model API requests.

Use `null` when a metric cannot be measured reliably. Do not estimate missing
token counts. Define counting conventions consistently across harnesses
(e.g. whether failed tool calls count as tool calls).

#### 8.4.5 Safety and failure reporting

Record:

- `collision_count`: number of detected collision events, using a documented
  counting convention.
- `unsafe_action_attempts`: number of proposed actions rejected by the safety
  layer.
- `emergency_stop_triggered`: whether emergency stop was activated.
- `failure_category`: primary standardized failure category.
- `failure_description`: concise explanation.
- `secondary_causes`: zero or more additional failure categories.

Use one primary failure category from this list:

- `none`: task completed successfully.
- `target_not_identified`: target object could not be identified reliably.
- `target_misidentified`: incorrect object selected.
- `planning_failure`: no suitable manipulation strategy was produced.
- `ik_failure`: inverse kinematics could not produce a valid target.
- `safety_rejection`: movement rejected by the safety layer.
- `collision`: unexpected collision occurred.
- `approach_failure`: pre-grasp or approach movement failed.
- `grasp_failure`: target was not successfully grasped.
- `lift_failure`: grasp succeeded but lifting failed.
- `object_dropped`: target was released or dropped during manipulation.
- `verification_failure`: outcome could not be reliably verified.
- `agent_error`: agent generated invalid actions or otherwise failed.
- `harness_error`: orchestration, context management, or tool execution failed.
- `api_error`: model endpoint returned an error.
- `timeout`: task exceeded its allowed duration.
- `other`: failure not covered by the categories above.

Use `none` only when `task_success` is `true`. Record secondary causes when
multiple failures occurred. For interrupted or incomplete trials, select the
most informative primary category and explain the interruption.

#### 8.4.6 Machine-readable per-trial report

Save each report as:

`results/{experiment_id}/{trial_id}.json`

Use this structure (the values are illustrative, not experimental results):

```json
{
  "experiment_id": "so101_harness_ablation",
  "trial_id": "trial_001",
  "timestamp": "2026-11-21T12:00:00Z",
  "harness": {
    "name": "OpenCode",
    "version": null
  },
  "model": {
    "id": "markoff/qwen3.8-flash-next-hibrid48",
    "endpoint": "talos"
  },
  "task": "Grasp and lift the blue cube.",
  "random_seed": 42,
  "initial_scene": {
    "blue_cube_position": [0.30, 0.10, 0.025]
  },
  "configuration_id": "baseline_v1",
  "recordings": [
    {
      "attempt_id": "attempt_001",
      "start_timestamp": "2026-11-21T12:00:05.123456Z",
      "end_timestamp": "2026-11-21T12:00:50.000000Z",
      "video_path": "results/so101_harness_ablation/trial_001/videos/20261121T120005.123456Z_attempt_001_success.mp4",
      "filename": "20261121T120005.123456Z_attempt_001_success.mp4",
      "outcome_label": "success",
      "recording_complete": true,
      "video_validation_status": "valid"
    }
  ],
  "outcome": {
    "target_identified": true,
    "approach_completed": true,
    "grasp_attempted": true,
    "grasp_success": true,
    "lift_attempted": true,
    "lift_success": true,
    "task_success": true
  },
  "verification": {
    "initial_cube_position": [0.30, 0.10, 0.025],
    "final_cube_position": [0.30, 0.10, 0.125],
    "maximum_lift_height": 0.10,
    "minimum_lift_height": 0.05,
    "verification_duration": 2.0,
    "grasp_contact_verified": true,
    "object_retained": true
  },
  "efficiency": {
    "duration_seconds": 45.2,
    "agent_iterations": 12,
    "tool_calls": 18,
    "observation_count": 4,
    "image_count": 4,
    "input_tokens": null,
    "output_tokens": null,
    "api_requests": 12,
    "api_errors": 0
  },
  "safety": {
    "collision_count": 0,
    "unsafe_action_attempts": 0,
    "emergency_stop_triggered": false
  },
  "failure": {
    "failure_category": "none",
    "failure_description": null,
    "secondary_causes": []
  }
}
```

#### 8.4.7 Aggregate comparison report

After completing a series of trials, produce an aggregate report grouped by
harness. At minimum, include:

- Number of trials, including incomplete and interrupted trials.
- Number and percentage of successful trials.
- Target identification accuracy.
- Grasp success rate.
- Lift success rate.
- Mean and median task completion time.
- Mean number of tool calls.
- Mean number of transmitted images.
- Mean input/output token consumption, when available.
- Total collision events and unsafe action attempts.
- Failure category distribution.

Define denominators explicitly. For example, report both grasp success among
trials where a grasp was attempted and overall grasp completion across all
trials. Do the same for lift success. Do not silently exclude trials in which
the harness or API failed.

Report per-trial outcomes alongside aggregate metrics. Do not exclude failed
or incomplete trials without documenting the reason.

Use identical trial seeds and initial configurations across harnesses
whenever possible. If exact replay is impossible, document the differences.

## 9. Implementation plan

Work incrementally. Keep each milestone runnable and testable.

### Milestone 0 --- Environment and dependency preflight

-   Confirm the active Python interpreter and environment.
-   Verify that PyBullet and the other currently required Python
    dependencies are already available in that environment.
-   Check whether an MP4 encoder is already available, for example the
    `ffmpeg` executable or an already installed Python video-writing
    package.
-   Do not install, upgrade, downgrade, or remove any software.
-   If any additional Python package is required, stop and ask the user
    to install it in the intended environment. Report the exact package,
    reason, and compatible version if known. Do not proceed with
    implementation that depends on the missing package until the user
    confirms installation.

**Deliverable:** A terminal preflight report listing the active Python
interpreter, relevant installed dependencies, available video encoder,
and any dependency that requires user action.

### Milestone 1 --- Inspect resources and establish the simulation

-   Inspect the supplied Simulation directory and identify the correct
    SO-101 assets and setup instructions.
-   Verify that the assets and meshes resolve correctly.
-   Load the table and robot in PyBullet.
-   Discover joints and links by name.
-   Confirm that the gripper and arm can be controlled.
-   Add colored cubes and a reset operation.

**Deliverable:** A standalone script that launches a stable SO-101
tabletop scene.

### Milestone 2 --- Headless camera capture and video recording

-   Configure the front-facing camera at the required approximate
    distance.
-   Implement RGB capture and image conversion in headless mode.
-   Confirm programmatically (using saved frames and/or image metadata)
    that all relevant objects and the robot workspace are visible.
-   Implement a reusable observation function that returns the RGB image
    for the multimodal LLM.
-   Implement MP4 recording for every grasping attempt, using the same
    configured camera viewpoint.
-   Check that an available video encoder can produce playable MP4 files.
-   Save sample frames for debugging when useful; do not display them in
    a GUI.

**Deliverable:** A headless scene with a reusable observation function
and verified MP4 recording for a grasping attempt.

### Milestone 3 --- Deterministic motion and safety

-   Implement joint discovery, IK, Cartesian movement, waypoint
    interpolation, and gripper control.
-   Add joint-limit and path-collision checks.
-   Add runtime collision monitoring and emergency stop.
-   Test home, pre-grasp, approach, and lift movements without the LLM.
-   Verify that forbidden contacts are detected and stop motion.

**Deliverable:** A safe, deterministic controller with a documented
Python API.

### Milestone 4 --- Agent-harness integration

-   Expose the observation, movement, gripper, reset, status, and stop
    operations to the selected harness.
-   Ensure image observations are actually passed to the multimodal LLM,
    not merely described as text.
-   Document the tool interface for the agent.
-   Test a short observe--act--observe interaction.

**Deliverable:** The selected harness can inspect the scene and issue
validated simulator actions.

### Milestone 5 --- End-to-end task

-   Implement the agent loop for natural-language tasks.
-   Run "Grasp and lift the blue cube."
-   Verify grasp and lift from fresh observations.
-   Add bounded retries and safe failure handling.
-   Record each grasping attempt to its own MP4 file.
-   Finalize each recording with a timestamp and objective success label.
-   Verify that the complete run works from a terminal-only SSH session,
    without opening any GUI or display window.

**Deliverable:** A complete, reproducible, headless demonstration of the
target task, with an MP4 recording for every grasping attempt.

### Milestone 6 --- Evaluation

-   Add repeatable trial initialization and configurable object
    positions.
-   Run multiple trials.
-   Calculate and report the metrics defined above.
-   Preserve per-trial logs and representative observations.

**Deliverable:** An evaluation summary that supports or rejects the
hypothesis based on observed results.

## 10. Acceptance criteria

The implementation is considered complete for the initial experiment
when all of the following are satisfied:

-   [ ] The SO-101 model is loaded from the supplied simulation
    resources.
-   [ ] The simulation runs on the CPU-only host within the available 32
    GB RAM and uses PyBullet `DIRECT` mode.
-   [ ] The complete workflow runs in a terminal-only remote SSH session
    without a GUI, display server, or X forwarding.
-   [ ] A table, arm, gripper, and colored cubes are present.
-   [ ] A front-facing camera provides images of the workspace from
    approximately 50--70 cm.
-   [ ] The same RGB observation can be passed to the multimodal LLM
    through the selected agent harness without opening a display window.
-   [ ] Every grasping attempt is recorded to an MP4 file in the
    designated videos subdirectory.
-   [ ] Every MP4 filename contains the attempt timestamp and objective
    success label (`success` or `failure`).
-   [ ] Each trial report references every grasp-attempt recording and
    records its validation status.
-   [ ] Missing Python dependencies cause the agent to pause and ask the
    user for installation; the agent never installs packages itself.
-   [ ] The agent can submit natural-language tasks.
-   [ ] The agent can request bounded Cartesian movements and gripper
    operations through a documented interface.
-   [ ] IK, joint limits, trajectory validation, and collision
    monitoring are handled deterministically.
-   [ ] Unsafe movements are rejected or stopped independently of the
    LLM.
-   [ ] The agent can attempt to grasp and lift the requested blue cube.
-   [ ] The result is checked using fresh observations and reported
    honestly.
-   [ ] The scene can be reset and trials can be repeated.
-   [ ] Logs are sufficient to diagnose failures and calculate
    evaluation metrics.
-   [ ] No fine-tuning or weight updates are performed.
-   [ ] Every trial produces the standardized machine-readable report defined in section 8.4.
-   [ ] The simulator-side evaluator and report generator are shared across harnesses.
-   [ ] Harness name/version and relevant API, image, and context-management failures are recorded.

## 11. Important implementation guidance

1.  **Inspect before coding.** First examine the repository's Simulation
    directory, URDF, meshes, and existing scripts. Reuse working
    resources rather than duplicating them.
2.  **Validate each layer independently.** Do not start by building the
    entire agent loop. First prove that the robot loads, the camera
    works, and deterministic motion is stable.
3.  **Keep the LLM out of the low-level control loop.** The LLM should
    reason about goals and observations. Deterministic code should
    perform IK, interpolation, collision checking, and motor control.
4.  **Treat images as first-class inputs.** Verify the complete image
    path from PyBullet capture through the selected agent harness to the model. A
    text-only approximation of the image does not satisfy the
    experiment.
5.  **Avoid privileged-state leakage.** The agent should identify
    objects from images. Do not provide simulator object coordinates or
    ground-truth labels to the agent unless explicitly running a
    separately documented ablation.
6.  **Use closed-loop execution.** Observe after meaningful actions and
    adapt when the scene differs from expectations.
7.  **Fail safely.** If perception is uncertain, IK fails, or a safe
    path is unavailable, stop and report the problem instead of guessing
    or bypassing checks.
8.  **Keep the implementation CPU-friendly.** PyBullet and the local
    orchestration code must run without GPU dependencies. Image
    processing should remain lightweight.
9.  **Document assumptions.** Record any simplifications in the robot
    model, gripper, camera calibration, collision geometry, or grasp
    verification.
10. **Headless by design.** Do not add GUI windows, live display, or
    desktop dependencies. Use terminal logs, saved observations, MP4
    attempt recordings, and machine-readable reports for monitoring.
11. **Respect the dependency gate.** Do not install Python packages or
    other software. If a required Python package is missing, pause and
    ask the user to install it in the active environment before
    proceeding.
12. **Do not overbuild.** Prioritize a robust blue-cube grasp-and-lift
    demonstration. Defer complex planning frameworks, additional
    perception models, and unrelated features until the baseline works.

## 12. Final report expected from the agent harness

When implementation is complete, provide a concise engineering report
containing:

-   Repository assets reused and any modifications made.
-   Implemented architecture and file layout.
-   How to launch the simulator and agent-harness integration.
-   How to run and reset the experiment.
-   The harness-independent agent-facing tool/API definitions.
-   Safety mechanisms and known limitations.
-   A demonstration result for the blue-cube task.
-   MP4 recording paths for every grasping attempt, with timestamped
    success/failure filenames.
-   Evaluation results, if repeatable trials have been completed.
-   Any missing dependencies for which implementation was paused pending
    user installation.
-   Known bugs, incomplete requirements, and recommended next steps.

Do not claim that a requirement is implemented unless it has been
tested. Clearly distinguish implemented and verified functionality from
planned or partially working functionality.
