# Servo configuration and calibration

This page records the calibration procedure for the SO-101 follower. Keep a copy of the calibration file locally and commit only files that are safe and useful to share.

## Recorded setup

- Servos: six STS3215 units (the tested build used a 1:345 ratio).
- Controller: FE-URT-2.
- Calibration tool: LeRobot calibration workflow.
- Example Linux serial device: `/dev/ttyACM1` (may differ on another boot or machine).

A calibration file was saved during the experiments at:

`~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json`

This is a machine-local path, not a file included in this repository.

## Recommended record

For every calibration session, record the LeRobot version, robot configuration, serial port, servo IDs, calibration filename, joint limits, and any manual adjustments.

## Verification

After calibration, test each joint individually at low speed and within conservative limits. Verify direction, neutral position, and mechanical clearance before attempting coordinated movement.
