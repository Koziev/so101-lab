"""
Label the arrangement of every recorded real sample, one key per sample (resumable).

    python real/label_arrangements.py --data data_real_v2

Shows the left image of each sample with its target description. Keys:
  1 = single object          2 = pair, different colours      3 = pair, same colour (cube + ball)
  4 = three objects, all different colours     5 = three objects, same colour     6 = other
  b = back one sample        q = save and quit
Writes <data>/arrangements.json ({sample index: label}); already labelled samples are skipped.
"""
import argparse
import glob
import json
import os

import cv2

LABELS = {ord("1"): "single", ord("2"): "pair_diff_color", ord("3"): "pair_same_color",
          ord("4"): "triple_diff_color", ord("5"): "triple_same_color", ord("6"): "other"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_real_v2")
    ap.add_argument("--all", action="store_true", help="show already labelled samples too")
    args = ap.parse_args()
    out = os.path.join(args.data, "arrangements.json")
    labels = json.load(open(out)) if os.path.exists(out) else {}
    files = sorted(glob.glob(os.path.join(args.data, "samples", "*.json")), key=lambda p: int(os.path.basename(p)[:-5]))
    todo = [f for f in files if args.all or os.path.basename(f)[:-5] not in labels]
    print(f"{len(files)} samples, {len(todo)} to label")
    i = 0
    while 0 <= i < len(todo):
        meta = json.load(open(todo[i]))
        idx = str(meta["index"])
        img = cv2.imread(todo[i][:-5] + "_left.png")
        text = f"#{idx}  target: {meta['target']}  objects: {meta.get('n_objects', '?')}  current: {labels.get(idx, '-')}"
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, "1 single 2 pair-diff 3 pair-same 4 triple-diff 5 triple-same 6 other  b back  q quit",
                    (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        cv2.imshow("label arrangement", cv2.resize(img, None, fx=1.5, fy=1.5))
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            i = max(0, i - 1)
            continue
        if key in LABELS:
            labels[idx] = LABELS[key]
            json.dump(labels, open(out, "w"), indent=1)
            i += 1
    json.dump(labels, open(out, "w"), indent=1)
    cv2.destroyAllWindows()
    counts = {}
    for v in labels.values():
        counts[v] = counts.get(v, 0) + 1
    print(f"saved {out}: {counts}")


if __name__ == "__main__":
    main()
