"""The full automated loop (bootstrap -> planned batches -> convergence) against a synthetic rig."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calib.estimate import CellCalibration, inv, rot_angle_deg  # noqa: E402
from calib.plan import Planner  # noqa: E402
from calib.session import run_session  # noqa: E402
from calib.urdf_fk import UrdfChain  # noqa: E402
from calib.gripper_axes import realistic_wrist_camera  # noqa: E402
from tests.synthetic_session import look_at, to_T  # noqa: E402
import math  # noqa: E402
import cv2  # noqa: E402


class SyntheticRig:
    def __init__(self, fk, X, Z, S, d, Kw, Dw, Ks, Ds, grid, noise=0.3, seed=1):
        self.fk, self.X, self.Z, self.S, self.d = fk, X, Z, S, d
        self.Kw, self.Dw, self.Ks, self.Ds, self.grid = Kw, Dw, Ks, Ds, grid
        self.rng = np.random.default_rng(seed)
        self.noise, self.gripper_pct = noise, 25.0
        self.q_true = np.array([0, -80, 80, 60, 0.0])          # folded "rest"
        self.rest_pose = np.r_[self.q_true - d, 25.0]
        self.moves = 0

    def read_joints(self):
        return np.r_[self.q_true - self.d, self.gripper_pct]

    def move_to(self, q_real):
        self.q_true = np.asarray(q_real, float)[:5] + self.d
        self.moves += 1

    def move_path(self, waypoints):
        pl = getattr(self, "planner", None)
        if pl is not None and pl.jaw_points is not None:
            prev, lowest = self.q_true.copy(), 9.0
            for w in waypoints:                                  # sample every segment as the arm travels it
                nxt = np.asarray(w, float)[:5] + self.d          # true angles = real command + true offsets
                for t in np.linspace(0, 1, 15):
                    lowest = min(lowest, pl.jaw_height(prev + (nxt - prev) * t))
                prev = nxt
            self.motion_minima = getattr(self, "motion_minima", []) + [lowest]
            self.min_jaw = min(getattr(self, "min_jaw", 9.0), lowest)
        self.q_true = np.asarray(waypoints[-1], float)[:5] + self.d
        self.moves += 1
        self.waypoints = getattr(self, "waypoints", 0) + len(waypoints)

    def _obs(self, T, K, D, usable):
        pts = (T[:3, :3] @ self.grid.T).T + T[:3, 3]
        if (pts[:, 2] < 0.04).any():
            return None
        img = cv2.projectPoints(self.grid, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, D)[0].reshape(-1, 2)
        ok = (img[:, 0] > 15) & (img[:, 0] < 1265) & (img[:, 1] > 15) & (img[:, 1] < 705) & usable(img[:, 0], img[:, 1])
        if ok.sum() < 10:
            return None
        return self.grid[ok], img[ok] + self.rng.normal(0, self.noise, (int(ok.sum()), 2))

    def observe(self):
        G = self.fk.pose(np.radians(self.q_true))
        jaws = lambda x, y: (x < 1024) & ~((x < 256) & (y > 504))
        wrist, stereo = [], []
        for b, Zb in enumerate(self.Z):
            w = self._obs(inv(G @ self.X) @ Zb, self.Kw, self.Dw, jaws)
            st = self._obs(inv(self.S) @ Zb, self.Ks, self.Ds, lambda x, y: np.ones_like(x, bool))
            if w is not None:
                wrist.append((b, *w))
            if st is not None:
                stereo.append((b, *st))
        return {"q": self.read_joints(), "wrist": wrist, "stereo": stereo}


def teach_by_hand(planner, d, steps=4000, step_deg=3.0, seed=5):
    """A simulated teach-in: a random walk through the (geometrically) safe region, 3 deg per step, recording a
    pose whenever it differs from all stored ones by >= 3 deg (as calib/workspace.py record does)."""
    rng = np.random.default_rng(seed)
    q = planner.random_safe(1)[0]
    pts = [q - d]
    direction = rng.normal(size=5)
    for _ in range(steps):
        direction = 0.8 * direction + 0.6 * rng.normal(size=5)
        step = direction / np.max(np.abs(direction)) * step_deg
        nq = q + step
        if planner.safe(nq):
            q = nq
            if np.min(np.max(np.abs(np.asarray(pts) - (q - d)), axis=1)) >= 3.0:
                pts.append(q - d)
        else:
            direction = -direction
    return np.asarray(pts)


def main(urdf):
    fk = UrdfChain(urdf, "gripper_link")
    Kw = np.array([[1291.0, 0, 622], [0, 1291, 361], [0, 0, 1]]); Dw = np.array([-0.15, 0.29, 0, 0, 0])
    Ks = np.array([[734.0, 0, 623], [0, 734, 369], [0, 0, 1]]); Ds = np.array([-0.005, -0.044, 0, 0, 0.01])
    X, _src = realistic_wrist_camera(urdf)       # beside the fixed jaw, ~50 deg off the approach axis
    Z = [to_T([0, 0, math.radians(8)], [0.17, -0.05, 0.0]), to_T([0, 0, math.radians(-4)], [0.17, 0.08, 0.0])]
    S = look_at(np.array([0.53, 0.03, 0.34]), np.array([0.18, 0.0, 0.0]), roll_deg=-2.4)
    d = np.array([0.5, 2.0, -3.0, 4.0, 0.0])
    grid = np.array([[i * 0.015, j * 0.015, 0.0] for j in range(5) for i in range(7)]) + [0.015, 0.015, 0]
    rig = SyntheticRig(fk, X, Z, S, d, Kw, Dw, Ks, Ds, grid)
    planner = Planner(fk, Kw, Dw, board_points={0: grid, 1: grid}, seed=2)
    if "--workspace" in sys.argv:
        from calib.workspace import Workspace
        taught = teach_by_hand(planner, d, steps=4000)                       # real readings, like the recorder
        ws = Workspace(taught)
        print(f"  simulated teach-in: {len(taught)} poses")
        planner = Planner(fk, Kw, Dw, board_points={0: grid, 1: grid}, seed=2, workspace=ws, d=np.r_[0.5, 0, 0, 0, 0])
        from calib.gripper_axes import jaw_points
        pts, _ = jaw_points(urdf)
        lowest = planner.set_jaw_rule(pts, [q + planner.d for q in ws.P], 0.010)
        ws_kept = ws.filtered(lambda p: planner.jaw_height(p + planner.d) >= planner.min_jaw_z)
        print(f"  jaw rule: lowest taught jaw point {100 * lowest:.1f} cm, kept {len(ws_kept.P)} of {len(ws.P)} poses")
        planner.ws_raw, planner.ws = ws, ws_kept
        rig.safe_pose_sim = max((q + planner.d for q in ws_kept.P), key=lambda q: planner.jaw_height(q))
        rig.planner = planner
        low = min(ws.P, key=lambda p: planner.jaw_height(p + planner.d))      # a LOW rest pose, like the folded arm
        rig.q_true, rig.rest_pose = low + d, np.r_[low, 25.0]
        print(f"  rest pose jaw height {100 * planner.jaw_height(low + planner.d):.1f} cm "
              f"(clearance rule needs >= {100 * planner.min_jaw_z:.1f} cm)")
    else:
        rig.safe_pose_sim = max(planner.random_safe(400), key=lambda q: fk.pose(np.radians(q))[2, 3])
    cal = CellCalibration(fk, (Kw, Dw), (Ks, Ds), d_prior=np.array([0.5, 0, 0, 0, 0]))
    if "--replay" in sys.argv:
        from calib.gripper_axes import jaw_points
        from calib.session import choose_stops, run_replay
        P = teach_by_hand(planner, d, steps=4000)                         # recording order, real readings
        pl = Planner(fk, Kw, Dw, board_points={0: grid, 1: grid}, seed=2)
        pts, _ = jaw_points(urdf)
        pl.set_jaw_rule(pts, [q + d for q in P], 0.010)
        centre = np.mean([Zb[:3, :3] @ grid.mean(axis=0) + Zb[:3, 3] for Zb in Z], axis=0)
        keep = lambda i: pl.jaw_height(P[i] + d) >= pl.min_jaw_z and np.linalg.norm(
            fk.pose(np.radians(P[i] + d))[:2, 3] - centre[:2]) <= 0.15
        stops = choose_stops(P, 60, keep)
        rig.q_true, rig.rest_pose, rig.planner = P[0] + d, np.r_[P[0], 25.0], pl
        res = run_replay(rig, cal, P, stops, jaw_ok=lambda q: pl.jaw_height(np.asarray(q)[:5] + d) >= pl.min_jaw_z,
                         log=lambda m: None if m.startswith("  stop") else print("  " + m))
        mm = rig.motion_minima
        print(f"  {len(P)} recorded poses, {len(stops)} stops, {rig.moves} motions ({rig.waypoints} waypoints)")
        print(f"  lowest jaw point: between stops {100 * min(mm[1:-2]):.2f} cm (limit {100 * pl.min_jaw_z:.2f} cm); "
              f"leaving rest {100 * mm[0]:.2f} cm, returning {100 * min(mm[-2:]):.2f} cm "
              f"(lowest recorded {100 * min(pl.jaw_height(q + d) for q in P):.2f} cm)")
        print(f"mount error {1000 * np.linalg.norm(res['X'][:3, 3] - X[:3, 3]):.2f} mm / "
              f"{rot_angle_deg(X[:3, :3].T @ res['X'][:3, :3]):.3f} deg; stereo error "
              f"{1000 * np.linalg.norm(res['S'][:3, 3] - S[:3, 3]):.2f} mm; offsets {np.round(res['d'][1:4], 2)} (true {d[1:4]})")
        return
    centre = np.mean([Zb[:3, :3] @ grid.mean(axis=0) + Zb[:3, 3] for Zb in Z], axis=0)   # rough position of the boards:
    guess = centre + np.array([0.02, -0.02, 0.0])                         # known to a few cm, as on the real cell
    res = run_session(rig, cal, planner, board_centre=guess, log=lambda m: print("  " + m))
    if getattr(rig, "planner", None) is not None and rig.planner.jaw_points is not None:
        mm = rig.motion_minima
        print(f"lowest jaw point (TRUE offsets): calibration moves {100 * min(mm[1:-1]):.2f} cm (allowed >= "
              f"{100 * rig.planner.min_jaw_z:.2f} cm); leaving rest {100 * mm[0]:.2f} cm, returning {100 * mm[-1]:.2f} cm "
              f"(taught paths, rest pose itself {100 * rig.planner.jaw_height(rig.rest_pose[:5] + rig.planner.d):.2f} cm)")
    print(f"\n{res['poses']} wrist views from {res['visited']} poses, {rig.moves} motions "
          f"({getattr(rig, 'waypoints', rig.moves)} waypoints in total)")
    print(f"mount error {1000 * np.linalg.norm(res['X'][:3, 3] - X[:3, 3]):.2f} mm / "
          f"{rot_angle_deg(X[:3, :3].T @ res['X'][:3, :3]):.3f} deg; stereo error "
          f"{1000 * np.linalg.norm(res['S'][:3, 3] - S[:3, 3]):.2f} mm / {rot_angle_deg(S[:3, :3].T @ res['S'][:3, :3]):.3f} deg; "
          f"boards {', '.join(f'{1000 * np.linalg.norm(res[chr(90)][b][:3, 3] - Z[b][:3, 3]):.2f} mm' for b in res[chr(90)])}; "
          f"offsets {np.round(res['d'][1:4], 2)} (true {d[1:4]})")


if __name__ == "__main__":
    main(next((a for a in sys.argv[1:] if not a.startswith("--")), "assets_so101/so101_sim.urdf"))
