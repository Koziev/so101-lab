---
date: 2026-11-06
categories:
  - Project
  - Lab notes
tags:
  - SO-101
  - SmolVLA
  - PyBullet
  - LeRobot
  - experiment-v9
---

# Experiment v9: how much camera randomisation can a small dataset afford?

## Concept

Experiment v8 showed that very broad camera randomisation destroyed the positioning precision of a SmolVLA grasping policy. v9 asked the narrower question: around the *real* camera position, how does the width of the randomisation trade off precision at that position against robustness to small camera displacements? To keep iterations short, every variant was trained on only 200 demonstrations.

## Setup

- **Scene and task:** the v8 simulator scene (plywood board, photo textures, single object 12–24 cm in front of the robot), the measured servo model of the 7.4 V arm (direction-dependent dead band, command delay), and an arm rendered only in light-grey to white shades, as both real arms are printed in white PLA.
- **Camera:** the stereo camera drawn from a normal distribution around the measured pose of the real camera (52.5 cm in front of the pan axis, 31.2 cm high, 38.7° down, 3.6 mm lenses, about 44° horizontal field of view). One width factor *s* scales all standard deviations; at *s* = 1 they are 3 cm (distance), 2.5 cm (height), 3° (pitch), 4° (azimuth), 2° (yaw) and 1.5° (roll), cut off at ±2.5σ. The field of view varies by only ±0.5° for every *s*.
- **Data and policy:** 200 episodes per value of *s* (same seeds), the scripted expert with servo-lag compensation (100 % success in a 45-scene check). SmolVLA fine-tuned from `lerobot/smolvla_base` for 6,000 steps (about 2.5 h on an RTX 3090), stereo images and joint angles only.
- **Evaluation:** every policy on three test sets with cameras drawn at *T* = 0 (exactly the real pose), *T* = 1 and *T* = 2, 100 identical scenes each, re-planning every 10 steps.

## Results

| Trained with | *T* = 0 (real pose) | *T* = 1 | *T* = 2 |
|---|---|---|---|
| *s* = 0 (fixed camera) | **42 %** (2.7 cm) | 12 % (5.0 cm) | 2 % (5.7 cm) |
| *s* = 1 | 24 % (3.6 cm) | **19 %** (4.2 cm) | 10 % (4.7 cm) |
| *s* = 2 | 24 % (3.7 cm) | 10 % (4.7 cm) | 10 % (4.6 cm) |

Success rate, with the median distance between gripper and object at closing in brackets. Comparisons are paired on identical scenes.

- The fixed-camera policy was the best at its own camera position, but lost most of its skill when the camera moved by about one standard deviation (42 % → 12 %).
- Randomisation made the policy flatter but lower: *s* = 1 lost significantly at the real pose (23 scenes won by *s* = 0, 5 by *s* = 1, p = 0.001) and gained significantly only at *T* = 2 (p = 0.02). Averaged over the three test sets, *s* = 0 and *s* = 1 were equal (19 % vs 18 %).
- Widening further did not help: *s* = 2 was no better than *s* = 1 anywhere and worse at *T* = 1 (p = 0.035).
- Policies trained with the servo model failed almost completely with ideal servos: they learned to command beyond the target to compensate for the lag.

## Conclusions

- A policy trained with a fixed camera is precise but fragile: a few centimetres or degrees of real camera displacement remove most of its success.
- With 200 demonstrations, any camera randomisation spreads the data too thin; the dataset size, not the randomisation width, was the binding constraint.
- Learning both robust perception and precise control from a few hundred closed-loop demonstrations is expensive. This motivated experiment 10, which separates the perception part ("where is the object?") and trains it on single images, which are far cheaper to generate than demonstrations.