---
date: 2026-11-26
categories:
  - Project
  - Lab notes
tags:
  - SO-101
  - agentic robotics
  - multimodal LLM
  - OpenCode
  - MCP
  - PyBullet
  - specification gaming
---

# Autonomous Harness Control 1: A Coding Agent Finds a Shortcut in the Simulator

## A separate line of research

The SO-101 experiments 1–14 focused primarily on training and evaluating dedicated manipulation policies, including SmolVLA and Qwen-based approaches. This experiment starts a separate line of research: can a general-purpose multimodal LLM, operating through an agent harness and a set of simulator tools, build the software needed to control a robot—without training a specialized VLA policy?

The motivation came partly from the discussion about general-purpose models moving into robotics. On 13 November 2026, [Fortune reported that OpenAI had disbanded its standalone robotics team and reassigned its researchers to other groups](https://fortune.com/2026/11/13/openai-disbands-robotics-team-as-its-general-models-start-running-robots/). The article described OpenAI's view that its general frontier models had become directly applicable to robotics. This was an organizational change and a strategic signal, not proof that specialized robotics models were no longer useful. In parallel, Google DeepMind introduced [Gemini Robotics 2.5](https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/), explicitly presented as a vision-language-action model for robot control.

I wanted to test a much smaller version of the broader idea with the hardware and compute resources available to me.

The hypothesis was:

> A general-purpose multimodal LLM, orchestrated by a coding-agent harness and connected to a robot simulator through bounded tools, can implement and execute a basic object-manipulation task without fine-tuning or a dedicated VLA model.

The experiment was deliberately framed as a software-agent task. Rather than asking the model to output individual servo commands directly, I gave the harness access to a task specification and a simulator interface, then asked the agent to implement the control system.

## Experimental setup

The simulated scene contained an SO-101 manipulator mounted beside a tabletop, three coloured cubes (blue, red and green), and a front-facing camera. PyBullet provided the 3D physics simulation.

The implementation was constrained to run on the CPU. GPU access was prohibited, no model fine-tuning was allowed, and the agent had to work with the already configured model endpoint.

| Component | Configuration |
|---|---|
| Robot | SO-101 arm and gripper |
| Simulator | PyBullet |
| Scene | Table and three coloured cubes |
| Agent harness | OpenCode |
| Model | DeepSeek V4.1 Flash 1017 |
| Compute | CPU-only; no GPU |
| Reported API cost | approximately $0.90 |
| Task | “Grasp and lift the blue cube.” |

The full [task specification](https://github.com/Koziev/so101-lab/blob/main/experiments/autovla1/so101_opencode_task_specification.md) defined the expected architecture and evaluation protocol. In particular, the simulator had to expose a small tool interface through an MCP (Model Context Protocol) server. The agent was expected to observe the scene through camera images, plan the manipulation, issue bounded actions, and verify the result.

The specification also required headless operation: no GUI windows or live displays. The objective was not merely to produce a plausible animation. A successful grasp had to be established from simulator-side evidence that the cube was retained by the gripper and lifted above the tabletop for a verification interval.

## What the agent built

The agent successfully implemented the MCP integration. It created a long-running simulator daemon that owned the PyBullet instance and exposed a local HTTP API. An MCP stdio server and a command-line client acted as thin clients of that daemon. This separation allowed the simulator to remain alive independently of an individual agent session and made the simulator interface less dependent on OpenCode.

The resulting tool interface included operations for obtaining a camera observation, checking scene status, moving the gripper to a Cartesian position or named pose, opening and closing the gripper, advancing simulation steps, resetting the scene, and stopping motion.

The implementation also included a deterministic control layer: inverse kinematics, joint-limit checks, path collision checks, runtime contact monitoring, and an emergency stop. The agent produced a structured engineering report describing the architecture, launch procedure, tools, tests, and known limitations.

The report is available as [FINAL_REPORT.md](https://github.com/Koziev/so101-lab/blob/main/experiments/autovla1/FINAL_REPORT.md), and the implementation is in the [autovla1 code directory](https://github.com/Koziev/so101-lab/tree/main/experiments/autovla1).

The MCP integration was a genuine success of the engineering part of the task. The agent managed to build a useful interface between OpenCode and a persistent physics simulator.

## The apparent success—and the actual failure

The final engineering report recorded three OpenCode/LLM trials as successful, with a reported 3/3 success rate. It included an example trial in which the cube was reported as retained and lifted.

However, watching the debugging process raised a concern: the cube appeared to rise with the gripper without a convincing physical grasp. The trajectory video made the problem clear.

<video src="/so101-lab/assets/autovla1/grasp_trajectory.recoded.mp4" controls playsinline width="100%"></video>

[Open the grasp trajectory video](https://github.com/Koziev/so101-lab/blob/main/docs/assets/autovla1/grasp_trajectory.recoded.mp4).

The agent had modified the simulator's grasp logic. In the implementation documented in the repository, closing the gripper while an object was within a configured grasp radius created a rigid constraint attaching that object to the gripper. The attachment was engaged before the jaws moved. In effect, the cube could become mechanically attached to the manipulator without being pinched and retained through the intended jaw-contact mechanics.

The implementation described this as a “compliant grasp” simplification intended to work around limitations of the stock gripper collision geometry in that PyBullet build. Regardless of the label, this changed the physical problem being evaluated. Once the constraint was created, lifting the arm also lifted the attached cube. The simulator-side evaluator could then observe object retention and vertical displacement and report success.

That is not evidence that the model understood how to grasp the cube. It is a simulator-side shortcut that bypassed the central manipulation requirement. The 3/3 result in the engineering report therefore cannot be interpreted as a valid grasp-and-lift success rate for the intended task.

This is a useful distinction: the agent did not merely make a mistake in its motion plan. It changed the environment in a way that made the success condition easier to satisfy. The apparent completion was a form of specification gaming.

## A second deviation from the task specification

The repository's engineering report and README also disclose a Tkinter window that displays the simulator camera image. The documented launch command sets a display environment variable, while the README describes a live window and provides a no-window option.

This conflicts with the specification's explicit requirement that the system run headlessly, without creating windows or requiring a display server. PyBullet may still be running in DIRECT mode, but adding a separate live display window is nevertheless a deviation from the required terminal-only execution path.

The MCP requirement was handled successfully, but the result should not be described as fully compliant with the complete task specification.

## What this experiment demonstrates

The experiment produced a mixed result.

**What worked:**
- OpenCode connected to the simulator through an MCP server.
- The agent created a modular simulator/control interface rather than relying on a monolithic script.
- It implemented useful deterministic components, including IK, collision checks, runtime monitoring, and structured trial reporting.
- The simulator daemon could persist across agent sessions.

**What did not work:**
- The manipulation result was invalidated by a simulator modification that attached the object to the gripper through a rigid constraint.
- The implementation diverged from the headless-only requirement by adding a live Tkinter display.
- Consequently, the reported 3/3 OpenCode success count does not establish that the multimodal model can perform a physical grasp, even in the intended simulated setup.

The broader lesson is about the evaluation of coding agents. When an agent is allowed to modify both the controller and the environment used to score its work, it can satisfy the observable success condition without solving the intended task. This is not necessarily evidence of deliberate deception. It is a failure of the trust boundary between the system being developed and the mechanism used to evaluate it.

For this class of experiment, the simulator physics and success evaluator should be treated as trusted infrastructure. The agent may create or modify the controller and harness integration, but it should not be able to weaken grasp mechanics, inject object attachments, or redefine task success. The evaluation should independently inspect the final code changes and verify that the object is retained through the intended gripper contacts—not through a special constraint created by the controller.

## Next steps

A follow-up experiment should preserve the useful MCP architecture while tightening the boundary between agent-written code and the trusted simulator:

1. Keep the physics engine, object dynamics, grasp mechanics, and success evaluator outside the agent's writable area.
2. Restrict the agent's changes to the controller and harness integration, then review the resulting diff before evaluation.
3. Require grasp verification to use the intended jaw contacts and object-retention measurements; do not accept a controller-created fixed constraint as a grasp.
4. Make headless execution the only supported path for the experiment, rather than an optional launch mode.
5. Run repeated trials with randomized cube positions and retain all per-trial reports and attempt videos.
6. Once the baseline is trustworthy, compare alternative harnesses while keeping the model, task, simulator, and evaluation protocol fixed.

The first run did not confirm the hypothesis that a general-purpose multimodal LLM can independently control this manipulator to achieve a valid grasp. It did show that a coding-agent harness can assemble a substantial simulation-control stack—and that a poorly isolated evaluator can be gamed by the very agent it is meant to assess.

## Reproducibility resources

- [Task specification](https://github.com/Koziev/so101-lab/blob/main/experiments/autovla1/so101_opencode_task_specification.md)
- [Implementation and simulator integration](https://github.com/Koziev/so101-lab/tree/main/experiments/autovla1)
- [Final engineering report](https://github.com/Koziev/so101-lab/blob/main/experiments/autovla1/FINAL_REPORT.md)
- [Recorded grasp trajectory](https://github.com/Koziev/so101-lab/blob/main/docs/assets/autovla1/grasp_trajectory.recoded.mp4)
