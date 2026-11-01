# Teleoperation

Document the control interface used to move the physical follower arm.

Previous experiments included a terminal-based joint jog controller and a teach-and-repeat script that records manually guided joint positions and plays them back.

For each revision, record:

- Joint naming and mapping to servo IDs.
- Units and sign conventions.
- Joint limits and how they were established.
- Update frequency and motion interpolation.
- Safety behavior when communication is lost.
- Playback speed and transition handling.

Do not treat playback limits or joint ranges as safe until they have been checked against the physical mechanism.
