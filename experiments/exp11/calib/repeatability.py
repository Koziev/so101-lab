"""
Repeatability of the arm, from the revisited stops of a session (autocalib --revisit-every):
for every pose visited more than once, compare
  - the joint readings of the visits (what the servos report), and
  - the camera's pose relative to the board, from the wrist image (what physically happened).
The camera motion predicted from the reading differences (kinematics) is subtracted from the observed one; what
remains is motion the readings do not report: play, compliance, settling. Reported in mm and degrees, and as
the equivalent wrist-image shift in pixels.

    python calib/repeatability.py setups/<name>/calibration/<time> --setup setups/<name>
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "camera"))
import charuco_board  # noqa: E402
from calib.estimate import inv, pnp, rot_angle_deg  # noqa: E402
from calib.urdf_fk import UrdfChain  # noqa: E402
from setup_io import Setup  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--setup", required=True)
    ap.add_argument("--urdf", default=os.path.join(ROOT, "assets_so101", "so101_sim.urdf"))
    ap.add_argument("--boards", default="2xA4")
    args = ap.parse_args()
    setup = Setup(args.setup)
    w = setup.load("wrist_calib.json")
    K, D = np.array(w["K"]), np.array(w["dist"])
    mount = setup.load("wrist_mount.json")
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = np.array(mount["R_cam2gripper"]), np.array(mount["t_cam2gripper_m"])
    d = np.asarray(setup.offsets()[:5])
    fk = UrdfChain(args.urdf, "gripper_link")
    boards = charuco_board.make_board_set(args.boards)
    groups = defaultdict(list)
    for qf in sorted(glob.glob(os.path.join(args.session, "images", "*_q.json"))):
        info = json.load(open(qf))
        if "pose_index" not in info:
            continue
        img = cv2.imread(qf[:-len("_q.json")] + "_wrist.png", cv2.IMREAD_GRAYSCALE)
        for b, obj, im in charuco_board.detect_set(boards, img, 8):
            C = pnp(obj, im, K, D)                                   # board -> camera
            if C is not None:
                groups[(info["pose_index"], b)].append((os.path.basename(qf)[:3], np.asarray(info["q_real"])[:5], C))
    rows = []
    for (pi, b), visits in sorted(groups.items()):
        if len(visits) < 2:
            continue
        name0, q0, C0 = visits[0]
        for name, q, C in visits[1:]:
            dq = q - q0
            # camera pose in the board frame, observed vs predicted from the readings (same board, same mounting)
            cam_obs0, cam_obs = inv(C0), inv(C)
            G0, G = fk.pose(np.radians(q0 + d)), fk.pose(np.radians(q + d))
            pred = inv(G0 @ X) @ (G @ X)                             # camera motion implied by the readings
            obs = inv(cam_obs0) @ cam_obs
            # obs: camera motion in the first camera's frame (board fixed): T_c0<-c = inv(T_b<-c0) T_b<-c
            res = inv(pred) @ obs
            dist = float(np.linalg.norm(C[:3, 3]))
            rows.append({"pose": pi, "board": boards[b][0], "visits": f"{name0}->{name}",
                         "dq_max_deg": float(np.max(np.abs(dq))),
                         "obs_mm": float(1000 * np.linalg.norm(obs[:3, 3])), "obs_deg": rot_angle_deg(obs[:3, :3]),
                         "unexplained_mm": float(1000 * np.linalg.norm(res[:3, 3])),
                         "unexplained_deg": rot_angle_deg(res[:3, :3]),
                         "pixels": float(K[0, 0] * (math.radians(rot_angle_deg(res[:3, :3]))
                                                    + np.linalg.norm(res[:3, 3]) / max(dist, 0.05)))})
    if not rows:
        raise SystemExit("no revisited stops with a board view in this session (run with --revisit-every)")
    print(f"{'pose':>5} {'brd':>3} {'visits':>9} {'readings differ':>16} {'camera moved':>16} {'not explained by readings':>28}")
    for r in rows:
        print(f"{r['pose']:5d} {r['board']:>3} {r['visits']:>9} {r['dq_max_deg']:12.2f} deg "
              f"{r['obs_mm']:6.2f} mm {r['obs_deg']:5.2f} deg {r['unexplained_mm']:8.2f} mm {r['unexplained_deg']:5.2f} deg "
              f"(~{r['pixels']:4.1f} px)")
    um, ud, px = (np.array([r[k] for r in rows]) for k in ("unexplained_mm", "unexplained_deg", "pixels"))
    print(f"\n{len(rows)} revisits: unexplained camera motion median {np.median(um):.2f} mm / {np.median(ud):.2f} deg "
          f"(~{np.median(px):.1f} px in the wrist image), max {um.max():.2f} mm / {ud.max():.2f} deg")
    print("compare with the calibration's median per-view error: if similar, play/compliance sets the floor")


if __name__ == "__main__":
    main()
