---
date: 2026-11-01
categories:
  - Project
  - Lab notes
tags:
  - SO-101
  - documentation
---

# Starting SO-101 Lab

This repository is the home for my experiments with the SO-101 robotic arm. I want to document the entire process: printing and assembling the hardware, configuring the servos, writing small control utilities, building simulations, and eventually exploring more autonomous manipulation.

## Why keep a lab notebook?

Robotics experiments involve many small details that are easy to lose: which servo variant was used, what voltage was tested, which calibration file belongs to which build, what changed between software versions, and why a particular approach was abandoned.

The goal is to capture those details alongside the code, CAD exports, diagrams, and photographs. Successful results matter, but so do failed experiments and the measurements that explain them.

## Initial platform

The current physical platform is an SO-101 follower arm with six STS3215 serial bus servos and an FE-URT-2 controller. The software work uses LeRobot, with PyBullet for selected simulation experiments.

These are notes about my particular build; hardware variants and software versions should always be checked before reproducing a procedure.

## What comes next

- Organize the printing and assembly notes.
- Document the controller, servo configuration, and calibration procedure.
- Publish the small utilities used for teleoperation and teach-and-repeat experiments.
- Add simulation models and compare their kinematics with the physical arm.

This is the starting point. I'll update the documentation as experiments are repeated and verified.
