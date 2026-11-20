"""Synthetic calibration session with a known truth (toy URDF or the real one): checks that the estimator
recovers the wrist mounting, the board pose, the stereo pose and the joint offsets."""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calib.estimate import CellCalibration, inv, rot_angle_deg, to_T  # noqa: E402
from calib.urdf_fk import UrdfChain  # noqa: E402
from calib.gripper_axes import realistic_wrist_camera  # noqa: E402


def look_at(eye, target, roll_deg=0.0):
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.c_[x, y, z]
    a = math.radians(roll_deg)
    R = R @ np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    T = np.eye(4); T[:3, :3], T[:3, 3] = R, eye
    return T


def run(urdf, n_poses=25, noise_px=0.3, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    fk = UrdfChain(urdf, "gripper_link")
    Kw = np.array([[1291.0, 0, 622], [0, 1291, 361], [0, 0, 1]]); Dw = np.array([-0.15, 0.29, 0, 0, 0])
    Ks = np.array([[734.0, 0, 623], [0, 734, 369], [0, 0, 1]]); Ds = np.array([-0.005, -0.044, 0, 0, 0.01])
    # truth
    X_true, _src = realistic_wrist_camera(urdf)       # beside the fixed jaw, ~50 deg off the approach axis
    Z_true = to_T([0, 0, math.radians(8)], [0.20, 0.01, 0.0])                # board A on the table, turned 8 deg
    Z_B = to_T([0, 0, math.radians(-5)], [0.20, -0.13, 0.0])                   # board B beside it
    S_true = look_at(np.array([0.53, 0.03, 0.34]), np.array([0.18, 0.0, 0.0]), roll_deg=-2.4)
    d_true = np.array([0.5, 2.0, -3.0, 4.0, 0.0])                              # simulator = real + d
    grid = np.array([[i * 0.015, j * 0.015, 0.0] for j in range(5) for i in range(7)]) + [0.015, 0.015, 0]
    cal = CellCalibration(fk, (Kw, Dw), (Ks, Ds), d_prior=np.array([0.5, 0, 0, 0, 0]))

    def observe(T_board_cam, K, D, size=(1280, 720)):
        pts_cam = (T_board_cam[:3, :3] @ grid.T).T + T_board_cam[:3, 3]
        if (pts_cam[:, 2] < 0.05).any():
            return None, None
        r, t = cv2.Rodrigues(T_board_cam[:3, :3])[0], T_board_cam[:3, 3]
        img = cv2.projectPoints(grid, r, t, K, D)[0].reshape(-1, 2)
        ok = (img[:, 0] > 20) & (img[:, 0] < size[0] * 0.8) & (img[:, 1] > 20) & (img[:, 1] < size[1] - 20)
        if ok.sum() < 12:
            return None, None
        return grid[ok], img[ok] + rng.normal(0, noise_px, (ok.sum(), 2))

    tries = 0
    while len(cal.samples) < n_poses and tries < 20000:
        tries += 1
        q_true = np.array([rng.uniform(-30, 30), rng.uniform(-20, 50), rng.uniform(-60, 30), rng.uniform(-20, 90),
                           rng.uniform(-90, 90)])
        G = fk.pose(np.radians(q_true))
        if G[2, 3] < 0.08:
            continue
        for bi, Zb in enumerate((Z_true, Z_B)):
            obj, img = observe(inv(G @ X_true) @ Zb, Kw, Dw)
            if obj is not None:
                cal.add_sample(q_true - d_true, obj, img, board=bi)          # the arm REPORTS q_true - d
    for _ in range(3):
        for bi, Zb in enumerate((Z_true, Z_B)):
            obj, img = observe(inv(S_true) @ Zb, Ks, Ds)
            if obj is not None:
                cal.add_stereo(obj, img, board=bi)
    res = cal.solve()
    eX = 1000 * np.linalg.norm(res["X"][:3, 3] - X_true[:3, 3]); rX = rot_angle_deg(X_true[:3, :3].T @ res["X"][:3, :3])
    eS = 1000 * np.linalg.norm(res["S"][:3, 3] - S_true[:3, 3]); rS = rot_angle_deg(S_true[:3, :3].T @ res["S"][:3, :3])
    eZ = 1000 * np.linalg.norm(res["Z"][0][:3, 3] - Z_true[:3, 3])
    eZB = 1000 * np.linalg.norm(res["Z"][1][:3, 3] - Z_B[:3, 3]) if 1 in res["Z"] else float("nan")
    if verbose:
        print(f"{len(cal.samples)} wrist board views ({tries} candidates), {len(cal.stereo_obs)} stereo board views, noise {noise_px} px")
        print(f"  wrist mounting: position error {eX:.2f} mm, rotation error {rX:.3f} deg  (std {res['std']['X_pos_mm']:.2f} mm, {res['std']['X_rot_deg']:.3f} deg)")
        print(f"  board poses:    A {eZ:.2f} mm, B {eZB:.2f} mm  (views per board {res['views_per_board']})")
        print(f"  stereo pose:    position error {eS:.2f} mm, rotation error {rS:.3f} deg")
        print(f"  joint offsets:  true {d_true[1:4]}  estimated {np.round(res['d'][1:4], 2)}  std {np.round(list(res['std']['d_deg'].values()), 2)}")
        print(f"  residuals: wrist {res['wrist_rms_px']:.2f} px, stereo {res['stereo_rms_px']:.2f} px")
    return eX, rX, eS, rS, res["d"] - d_true


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "assets_so101/so101_sim.urdf")
