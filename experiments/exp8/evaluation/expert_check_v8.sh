#!/usr/bin/env bash
# Sanity check before generating data: the scripted expert under the v8 servo model (~97 % expected).
set -euo pipefail
mkdir -p outputs
python evaluation/eval_policy.py --policy expert --scene-preset v8 --width 640 --height 360 \
  --held-out-limit 0 --new-scenes 100 --results outputs/eval_expert_v8.json
