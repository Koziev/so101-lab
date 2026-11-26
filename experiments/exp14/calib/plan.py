"""
Pose planning for the automated calibration. All angles here are in the SIMULATOR convention (degrees).

Safety: joints at least `margin_deg` inside their URDF limits; the gripper_link origin at least `min_grip_z`
above the table (the jaws reach ~10 cm beyond it), the wrist above `min_wrist_z`; a transition between two
poses is checked along the straight joint-space path.
Bootstrap (mounting unknown): random safe poses looking forward/down over the board area; the runner keeps the
ones where the board is detected. Planned (mounting roughly known): candidates predicted to show >= `min_corners`
board corners in the usable part of the wrist image, at a usable distance and angle; chosen greedily to be as
different as possible (camera orientation first) from the poses already collected.
"""
import math

import cv2
import numpy as np

from calib.estimate import inv, rot_angle_deg


class Planner:
    def __init__(self, fk, Kw, Dw, image_size=(1280, 720), margin_deg=8.0, min_grip_z=0.15, min_wrist_z=0.12,
                 board_points=None, usable=None, seed=0, workspace=None, d=None):
        """workspace: calib.workspace.Workspace of taught poses (REAL readings); d: offsets (simulator = real + d)
        used to convert. With a workspace, safety = joint limits + staying within the taught region."""
        self.ws, self.d = workspace, (np.zeros(5) if d is None else np.asarray(d, float)[:5])
        self.jaw_points = None                                   # gripper points (gripper_link frame) for the height rule
        self.min_jaw_z = None                                    # lowest allowed height of those points (base frame)
        self.fk, self.Kw, self.Dw, self.size = fk, Kw, Dw, image_size
        self.lo = np.degrees(fk.lower[:5]) + margin_deg
        self.hi = np.degrees(fk.upper[:5]) - margin_deg
        self.min_grip_z, self.min_wrist_z = min_grip_z, min_wrist_z
        self.board_points = board_points
        # usable image region: the jaws occupy the right edge and the lower left corner of the wrist view
        self.usable = usable or (lambda x, y: (x < 0.8 * image_size[0]) & ~((x < 0.2 * image_size[0]) & (y > 0.7 * image_size[1])))
        self.rng = np.random.default_rng(seed)
        self._wrist_index = len(fk.chain) - 2 if len(fk.chain) >= 2 else 0

    # safety ---------------------------------------------------------------------------------------------
    def safe(self, q):
        q = np.asarray(q, float)
        if (q[:5] < self.lo).any() or (q[:5] > self.hi).any():
            return False
        if self.jaw_points is not None and self.min_jaw_z is not None and self.jaw_height(q) < self.min_jaw_z:
            return False
        if self.ws is not None:
            return self.ws.contains(q[:5] - self.d)
        T = self.fk.pose(np.radians(q[:5]))
        if T[2, 3] < self.min_grip_z:
            return False
        Tw = self._partial_pose(np.radians(q[:5]))                 # wrist (link before the roll joint)
        return Tw[2, 3] >= self.min_wrist_z

    def _partial_pose(self, q):
        T, k = np.eye(4), 0
        for j in self.fk.chain[:-1]:
            T = T @ j["T"]
            if j["type"] in ("revolute", "continuous"):
                from calib.urdf_fk import axis_angle, homog
                T = T @ homog(axis_angle(j["axis"], q[k]), np.zeros(3))
                k += 1
        return T

    def jaw_height(self, q_sim):
        """Lowest height (base frame) of the gripper's jaw tips and body for joint angles q (simulator degrees)."""
        T = self.fk.pose(np.radians(np.asarray(q_sim, float)[:5]))
        return float(np.min((T[:3, :3] @ self.jaw_points.T).T[:, 2] + T[2, 3]))

    def set_jaw_rule(self, jaw_points, taught_sim, clearance_m):
        """Height rule: every pose (and every point of every move) keeps the lowest jaw point at least
        `clearance_m` above the lowest jaw point among the taught poses."""
        self.jaw_points = np.asarray(jaw_points, float)
        lowest = min(self.jaw_height(q) for q in taught_sim)
        self.min_jaw_z = lowest + clearance_m
        return lowest

    def path_safe(self, q_from, q_to, steps=20):
        return all(self.safe(q_from + (q_to - q_from) * s) for s in np.linspace(0, 1, steps))

    def route(self, q_from, q_to):
        """Waypoints (simulator convention) for a safe move: direct if the straight path is safe; else through
        recorded neighbours in the taught workspace; None if neither works."""
        q_from, q_to = np.asarray(q_from, float), np.asarray(q_to, float)
        if self.path_safe(q_from, q_to):
            return [q_to]
        if self.ws is not None:
            way = self.ws.path(q_from[:5] - self.d, q_to[:5] - self.d)
            if way is not None:
                way = [w + self.d for w in way[1:]] + [q_to]
                prev = q_from
                for w in way:                       # steps between taught neighbours: limits + jaw height only
                    if not self.segment_ok(prev, w):
                        return None
                    prev = w
                return self._shortcut(q_from, way)
        return None

    def route_taught(self, q_from, q_to):
        """Route for leaving / returning to the rest pose: a path through the RAW taught poses (no clearance
        rule: the rest pose is low by nature, and every taught pose was made by hand), checked against the joint
        limits and the raw workspace tolerance only. None if not connected."""
        ws = getattr(self, "ws_raw", None) or self.ws
        if ws is None:
            return [np.asarray(q_to, float)]
        way = ws.path(np.asarray(q_from, float)[:5] - self.d, np.asarray(q_to, float)[:5] - self.d)
        if way is None:
            return None
        way = [w + self.d for w in way[1:]] + [np.asarray(q_to, float)]
        # no joint-limit check here: taught poses were made by hand and are physically valid even where the
        # URDF model's limits are narrower than the real arm's (e.g. the folded rest pose)
        out, cur, i = [], np.asarray(q_from, float), 0                       # shorten where a straight segment
        while i < len(way):                                                  # stays within the raw taught region
            j = len(way) - 1
            while j > i and not all(ws.contains(cur[:5] + (way[j][:5] - cur[:5]) * t - self.d)
                                    for t in np.linspace(0, 1, 20)):
                j -= 1
            out.append(way[j])
            cur, i = way[j], j + 1
        return out

    def segment_ok(self, q_from, q_to, steps=12):
        """A short step between neighbouring taught poses: joint limits and the jaw-height rule along it."""
        for t in np.linspace(0, 1, steps):
            q = q_from + (np.asarray(q_to) - q_from) * t
            if (q[:5] < self.lo).any() or (q[:5] > self.hi).any():
                return False
            if self.jaw_points is not None and self.min_jaw_z is not None and self.jaw_height(q) < self.min_jaw_z:
                return False
        return True

    def _shortcut(self, start, way):
        """Drop waypoints: from each point jump to the farthest later one reachable by a safe straight segment
        (the next waypoint itself is always allowed: that step was checked in route())."""
        out, cur, i = [], np.asarray(start, float), 0
        while i < len(way):
            j = len(way) - 1
            while j > i and not self.path_safe(cur, way[j]):
                j -= 1
            out.append(way[j])
            cur, i = way[j], j + 1
        return out

    # prediction -----------------------------------------------------------------------------------------
    def predict(self, q, X, Z):
        """Predicted wrist view: (corners in the usable region, distance m, viewing angle deg); with several
        boards (Z a dict, board_points a dict) the best-seen board counts."""
        if isinstance(Z, dict):
            res = [self._predict_one(q, X, Z[b], self.board_points[b] if isinstance(self.board_points, dict)
                                     else self.board_points) for b in Z]
            return max(res, key=lambda r: r[0])
        return self._predict_one(q, X, Z, self.board_points)

    def _predict_one(self, q, X, Z, board_points):
        T_cam = self.fk.pose(np.radians(q[:5])) @ X
        T_bc = inv(T_cam) @ Z
        pts = (T_bc[:3, :3] @ board_points.T).T + T_bc[:3, 3]
        if (pts[:, 2] < 0.04).any():
            return 0, 0.0, 90.0
        img = cv2.projectPoints(board_points, cv2.Rodrigues(T_bc[:3, :3])[0], T_bc[:3, 3], self.Kw, self.Dw)[0].reshape(-1, 2)
        x, y = img[:, 0], img[:, 1]
        inside = (x > 15) & (x < self.size[0] - 15) & (y > 15) & (y < self.size[1] - 15) & self.usable(x, y)
        normal_cam = T_bc[:3, :3] @ np.array([0, 0, 1.0])
        centre = pts.mean(axis=0)
        angle = math.degrees(math.acos(abs(float(normal_cam @ centre / np.linalg.norm(centre)))))
        return int(inside.sum()), float(np.linalg.norm(centre)), angle

    # candidates -----------------------------------------------------------------------------------------
    def random_safe(self, n, around=None, spread=None):
        """Safe joint vectors: from the taught workspace if there is one; else uniform in a forward/down-looking
        box, or around `around` (+- spread deg)."""
        if self.ws is not None:
            a = None if around is None else np.asarray(around, float)[:5] - self.d
            qs = [q + self.d for q in self.ws.sample(n, self.rng, around=a, spread=spread)]
            return [q for q in qs if self.safe(q)]
        out, tries = [], 0
        # wide search box (simulator degrees); the safety checks (heights, joint limits) do the filtering.
        # For reference, the SO-101 ready pose is about [0, -47, 42, 95, 0].
        box_lo = np.array([-40, -75, -75, -20, -100.0])
        box_hi = np.array([40, 60, 75, 105, 100.0])
        while len(out) < n and tries < 200 * n:
            tries += 1
            q = (np.asarray(around) + self.rng.uniform(-1, 1, 5) * spread) if around is not None else \
                self.rng.uniform(box_lo, box_hi)
            if self.safe(q):
                out.append(q)
        return out

    # systematic bootstrap ---------------------------------------------------------------------------
    def pan_axis_xy(self):
        return self.fk.chain[0]["T"][:2, 3].copy() if self.fk.chain else np.zeros(2)

    def forward(self):
        """Robot's forward direction (unit, horizontal) and pan-axis position in the base frame. Forward = the
        horizontal direction of the farthest reach with shoulder_pan = 0 (does not assume what the zero pose
        looks like)."""
        pan = self.pan_axis_xy()
        best, best_r = None, -1.0
        for lift in np.linspace(self.lo[1], self.hi[1], 15):
            for elbow in np.linspace(self.lo[2], self.hi[2], 15):
                tip = self.fk.pose(np.radians([0, lift, elbow, 0, 0]))[:2, 3] - pan
                r = float(np.linalg.norm(tip))
                if r > best_r:
                    best, best_r = tip, r
        return np.r_[best / best_r, 0.0], np.r_[pan, 0.0]

    def board_centre_guess(self, distance_m):
        f, pan = self.forward()
        return pan + f * distance_m

    def _place_gripper(self, target, flex, roll):
        """shoulder_pan, shoulder_lift and elbow_flex that put the gripper_link origin at `target` (wrist_flex,
        wrist_roll fixed); None if not reachable within 1 cm."""
        from scipy.optimize import least_squares
        rel = target[:2] - self.pan_axis_xy()
        heading = math.degrees(math.atan2(rel[1], rel[0]))
        lo, hi = self.lo[:3], self.hi[:3]
        best = None
        for dpan in (0, 90, -90, 180):                             # the pan zero may point anywhere
            for le in ((0, 0), (30, -30), (-30, 30), (45, 20), (-45, -20)):
                x0 = np.clip(np.array([heading + dpan, le[0], le[1]], float), lo + 1e-3, hi - 1e-3)
                f = lambda x: self.fk.pose(np.radians([x[0], x[1], x[2], flex, roll]))[:3, 3] - target
                r = least_squares(f, x0, bounds=(lo, hi))
                if best is None or r.cost < best.cost:
                    best = r
                if best.cost < 1e-8:
                    break
        if best is None or np.sqrt(2 * best.cost) > 0.01:
            return None
        return np.array([best.x[0], best.x[1], best.x[2], flex, roll])

    def bootstrap_from_workspace(self, board_centre, radius=0.12, n=80, report=print):
        """Taught poses with the gripper above the board area, chosen to differ as much as possible in their
        wrist angles (farthest-point sampling on wrist_flex / wrist_roll)."""
        c = np.asarray(board_centre, float)
        cand = [q + self.d for q in self.ws.P
                if np.linalg.norm(self.fk.pose(np.radians(q + self.d))[:2, 3] - c[:2]) <= radius]
        if not cand:
            report(f"no taught pose has the gripper within {100 * radius:.0f} cm of the board centre")
            return []
        cand = np.asarray(cand)
        chosen = [int(self.rng.integers(len(cand)))]
        w = np.array([0.5, 0.5, 0.5, 1.0, 1.0])                        # wrist angles matter most
        dmin = np.max(np.abs(cand - cand[chosen[0]]) * w, axis=1)
        while len(chosen) < min(n, len(cand)):
            i = int(np.argmax(dmin))
            chosen.append(i)
            dmin = np.minimum(dmin, np.max(np.abs(cand - cand[i]) * w, axis=1))
        report(f"bootstrap: {len(chosen)} taught poses above the board (of {len(cand)} within {100 * radius:.0f} cm)")
        return [cand[i] for i in chosen]

    def bootstrap_poses(self, board_centre, heights=(0.22, 0.27), flex_step=20.0, roll_step=45.0, back=0.06,
                        report=print):
        if self.ws is not None:
            return self.bootstrap_from_workspace(board_centre, report=report)
        """Gripper held above (and slightly towards the robot from) the board centre, its orientation swept over a
        grid of wrist_flex x wrist_roll: whatever the camera's direction on the gripper, some orientations look
        at the board. Safe, reachable poses only."""
        c = np.asarray(board_centre, float)
        towards_robot = self.pan_axis_xy() - c[:2]
        towards_robot /= max(1e-9, np.linalg.norm(towards_robot))
        out, unreachable, unsafe = [], 0, 0
        for h in heights:
            target = np.r_[c[:2] + towards_robot * back, h]
            for flex in np.arange(self.lo[3], self.hi[3] + 1e-6, flex_step):
                for roll in np.arange(max(self.lo[4], -135), min(self.hi[4], 135) + 1e-6, roll_step):
                    q = self._place_gripper(target, flex, roll)
                    if q is None:
                        unreachable += 1
                    elif not self.safe(q):
                        unsafe += 1
                    else:
                        out.append(q)
        if report:
            report(f"sweep: {len(out)} usable poses, {unreachable} unreachable, {unsafe} unsafe "
                   f"(gripper targets at heights {heights} m above {np.round(c, 3)})")
        return out

    def plan(self, X, Z, collected, k, min_corners=15, dist=(0.10, 0.35), max_angle=55.0, n_candidates=3000):
        """k new poses predicted to see the board well, as different as possible from `collected`."""
        cands = []
        for q in self.random_safe(n_candidates):
            n, d, a = self.predict(q, X, Z)
            if n >= min_corners and dist[0] <= d <= dist[1] and a <= max_angle:
                cands.append(q)
        if not cands:
            return []
        cam = lambda q: self.fk.pose(np.radians(q[:5])) @ X
        chosen_T = [cam(q) for q in collected]
        chosen = []
        for _ in range(k):
            best, best_score = None, -1.0
            for q in cands:
                T = cam(q)
                if not chosen_T:
                    score = 1.0
                else:
                    score = min(rot_angle_deg(Tc[:3, :3].T @ T[:3, :3]) + 100 * np.linalg.norm(Tc[:3, 3] - T[:3, 3])
                                for Tc in chosen_T)
                if score > best_score:
                    best, best_score = q, score
            chosen.append(best)
            chosen_T.append(cam(best))
            cands = [q for q in cands if q is not best]
        return chosen

    def order(self, start, poses):
        """Nearest-neighbour order in joint space (shorter, gentler moves)."""
        todo, out, cur = list(poses), [], np.asarray(start, float)
        while todo:
            i = int(np.argmin([np.abs(q - cur).sum() for q in todo]))
            cur = todo.pop(i)
            out.append(cur)
        return out
