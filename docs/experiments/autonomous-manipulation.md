# Autonomous manipulation

This page is reserved for experiments that combine perception, planning, and low-level robot control.

A useful initial decomposition is:

1. Perception: estimate object pose and scene state.
2. Task planning: select the next manipulation objective.
3. Motion generation: produce feasible arm and gripper targets.
4. Execution: run the low-level policy or controller.
5. Verification: check whether the intended physical effect occurred.

For every experiment, separate what the system observes from what it assumes. Record failure cases as carefully as successful runs.
