"""
SO-101 kinematics in a private DIRECT PyBullet client (so planning never disturbs the
physics world).

Joint angles are the URDF angles of so101_new_calib.urdf, where 0 rad is the middle of
each joint's range, the same convention as LeRobot's calibration. No sim<->real angle
conversion is applied anywhere; only radians<->degrees and the gripper 0..100 scale.

Everything about the gripper (TCP offset, approach axis, jaw closing axis, which end of
the gripper range is "open") is derived from the URDF at start-up rather than hardcoded,
so check_so101.py can print it for verification.
"""
import numpy as np
import pybullet
from pybullet_utils import bullet_client

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT = "gripper"
ALL_JOINTS = ARM_JOINTS + [GRIPPER_JOINT]
GRIPPER_LINK = "gripper_link"
TCP_LINK = "gripper_frame_link"
JAW_LINK = "moving_jaw_so101_v1_link"
SHOULDER_LINK = "shoulder_link"
BACK_DIST = 0.05  # second point on the approach axis, used to constrain gripper direction


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def name_maps(pb, body):
    """Joint name -> index and child-link name -> index. Never rely on file order."""
    joints, links = {}, {}
    for i in range(pb.getNumJoints(body)):
        info = pb.getJointInfo(body, i)
        joints[info[1].decode()] = i
        links[info[12].decode()] = i
    return joints, links


class SO101Kinematics:
    def __init__(self, urdf_path, rng=None):
        self.pb = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        self.body = self.pb.loadURDF(urdf_path, useFixedBase=True)
        self.joint_idx, self.link_idx = name_maps(self.pb, self.body)
        missing = [n for n in ALL_JOINTS if n not in self.joint_idx]
        if missing:
            raise RuntimeError(f"URDF lacks joints {missing}; is this so101_new_calib.urdf?")
        self.q_idx = [self.joint_idx[n] for n in ALL_JOINTS]
        info = [self.pb.getJointInfo(self.body, i) for i in self.q_idx]
        self.lower = np.array([x[8] for x in info])
        self.upper = np.array([x[9] for x in info])
        self.g_link = self.link_idx[GRIPPER_LINK]
        self.tcp_link = self.link_idx[TCP_LINK]
        self.jaw_link = self.link_idx[JAW_LINK]
        self.base_offset = np.zeros(3)  # world position of the robot base; set by the env

        # --- gripper geometry in the gripper_link frame (pose independent) ---
        self.set_q(np.zeros(6))
        g_pos, g_rot = self.frame(self.g_link)
        t_pos, _ = self.frame(self.tcp_link)
        self.tcp_local = g_rot.T @ (t_pos - g_pos)
        # wrist_roll rotates about gripper_link z; fingers point toward the TCP.
        self.approach_local = np.array([0.0, 0.0, np.sign(self.tcp_local[2]) or -1.0])
        self.back_local = self.tcp_local - BACK_DIST * self.approach_local
        _, j_rot = self.frame(self.jaw_link)
        self.hinge_local = unit(g_rot.T @ j_rot[:, 2])  # gripper joint axis is jaw-link z
        self.closing_local = unit(np.cross(self.hinge_local, self.approach_local))

        # Open = the gripper limit where the moving jaw is farther from the TCP.
        dists = []
        for g in (self.lower[5], self.upper[5]):
            q = np.zeros(6)
            q[5] = g
            self.set_q(q)
            jaw_com = np.array(self.pb.getLinkState(self.body, self.jaw_link, computeForwardKinematics=True)[0])
            dists.append(np.linalg.norm(jaw_com - self.frame(self.tcp_link)[0]))
        if dists[1] >= dists[0]:
            self.gripper_closed, self.gripper_open = self.lower[5], self.upper[5]
        else:
            self.gripper_closed, self.gripper_open = self.upper[5], self.lower[5]

        self.set_q(np.zeros(6))
        self.base_bottom_z = self.pb.getAABB(self.body, -1)[0][2]
        self.pan_axis_local = self.frame(self.link_idx[SHOULDER_LINK])[0][:2].copy()
        self.forward_yaw = self._detect_forward_yaw(rng or np.random.default_rng(0))
        self._probe_gripper()

    # ------------------------------------------------------------------ FK
    def set_q(self, q):
        for k, idx in enumerate(self.q_idx[:len(q)]):
            self.pb.resetJointState(self.body, idx, float(q[k]))

    def frame(self, link):
        s = self.pb.getLinkState(self.body, link, computeForwardKinematics=True)
        rot = np.array(self.pb.getMatrixFromQuaternion(s[5])).reshape(3, 3)
        return np.array(s[4]) + self.base_offset, rot

    def points(self, q, local=None):
        """Grasp point, back point and gripper rotation for joint vector q (5 or 6 values).
        `local` is the point in the gripper_link frame (default: URDF tool point)."""
        self.set_q(q)
        g_pos, g_rot = self.frame(self.g_link)
        loc = self.tcp_local if local is None else np.asarray(local, float)
        back = loc - BACK_DIST * self.approach_local
        return g_pos + g_rot @ loc, g_pos + g_rot @ back, g_rot

    def min_gripper_z(self, q6):
        """Lowest point (AABB, conservative) of gripper body and moving jaw, world z."""
        self.set_q(q6)
        z = min(self.pb.getAABB(self.body, l)[0][2] for l in (self.g_link, self.jaw_link))
        return z + self.base_offset[2]

    def pose_error(self, q5, tcp_target, approach, local=None):
        tcp, back, _ = self.points(q5, local)
        pos_err = np.linalg.norm(tcp - tcp_target)
        ang = np.degrees(np.arccos(np.clip(unit(tcp - back) @ unit(approach), -1, 1)))
        return pos_err, ang

    # ------------------------------------------------------------------ gripper scale
    def gripper_pct_to_rad(self, pct):
        """LeRobot convention: 0 = closed, 100 = open (mapped linearly onto URDF limits)."""
        return self.gripper_closed + np.clip(pct, 0, 100) / 100.0 * (self.gripper_open - self.gripper_closed)

    def gripper_rad_to_pct(self, rad):
        return 100.0 * (rad - self.gripper_closed) / (self.gripper_open - self.gripper_closed)

    # ------------------------------------------------------------------ IK
    def solve(self, tcp_target, approach_dir, roll, seeds, local=None, w_dir=0.6, max_iter=150,
              pos_tol=0.003, ang_tol_deg=8.0):
        """
        Levenberg-Marquardt IK over (pan, lift, elbow, wrist_flex) with wrist_roll fixed.
        Constrains the TCP position and the gripper's approach direction (via a second
        point on the approach axis), which is what a 5-DoF arm can actually achieve.
        Returns (q5, ok, pos_err_m, ang_err_deg).
        """
        tcp_target = np.asarray(tcp_target, float)
        a = unit(approach_dir)
        back_target = tcp_target - a * BACK_DIST
        lo, hi = self.lower[:4], self.upper[:4]

        def resid(q4):
            tcp, back, _ = self.points(np.r_[q4, roll], local)
            return np.r_[tcp - tcp_target, w_dir * (back - back_target)]

        best = None
        for seed in seeds:
            q = np.clip(np.asarray(seed, float)[:4], lo, hi)
            r = resid(q)
            cost, lam = r @ r, 1e-3
            for _ in range(max_iter):
                J = np.empty((6, 4))
                for j in range(4):
                    dq = np.zeros(4)
                    dq[j] = 1e-5
                    J[:, j] = (resid(q + dq) - r) / 1e-5
                H, g = J.T @ J, J.T @ r
                step = -np.linalg.solve(H + lam * (np.diag(np.diag(H)) + 1e-6 * np.eye(4)), g)
                q_new = np.clip(q + step, lo, hi)
                r_new = resid(q_new)
                c_new = r_new @ r_new
                if c_new < cost:
                    q, r, cost = q_new, r_new, c_new
                    lam = max(lam * 0.3, 1e-7)
                    if cost < 1e-10:
                        break
                else:
                    lam *= 10
                    if lam > 1e6:
                        break
            q5 = np.r_[q, roll]
            pe, ae = self.pose_error(q5, tcp_target, a, local)
            if best is None or pe + 1e-3 * ae < best[1] + 1e-3 * best[2]:
                best = (q5, pe, ae)
        q5, pe, ae = best
        return q5, (pe < pos_tol and ae < ang_tol_deg), pe, ae

    def best_roll(self, q5, yaw_target, symmetry_deg=90.0, n=721):
        """wrist_roll that aligns the jaw closing axis with yaw_target (mod symmetry)."""
        sym = np.radians(symmetry_deg)
        best_r, best_c = 0.0, np.inf
        for r in np.linspace(self.lower[4], self.upper[4], n):
            _, _, rot = self.points(np.r_[q5[:4], r])
            c = rot @ self.closing_local
            d = (np.arctan2(c[1], c[0]) - yaw_target) % sym
            cost = min(d, sym - d) + 0.05 * abs(r)  # mild preference for small roll
            if cost < best_c:
                best_r, best_c = r, cost
        return best_r

    def _detect_forward_yaw(self, rng, n=4000):
        """Direction in which the arm reaches the table with shoulder_pan = 0."""
        vecs = []
        for _ in range(n):
            q = rng.uniform(self.lower[:5], self.upper[:5])
            q[0] = 0.0
            tcp, _, _ = self.points(q)
            v = tcp[:2] - self.pan_axis_local
            if 0.0 < tcp[2] - self.base_bottom_z < 0.06 and np.linalg.norm(v) > 0.08:
                vecs.append(unit(v))
        if not vecs:
            return 0.0
        m = np.mean(vecs, axis=0)
        return float(np.arctan2(m[1], m[0]))

    # ------------------------------------------------------------------ jaw gap
    def _probe_gripper(self, res=0.001, half_span=0.06, depths=np.arange(-0.05, 0.0201, 0.002),
                       pcts=(30, 50, 70, 100)):
        """
        Measure the jaw gap from collision geometry. On lines parallel to the closing axis
        (through the tool point shifted by `depth` along the approach axis) a tiny sphere
        tests which link occupies each point. A free run bounded by gripper_link on one side
        and the moving jaw on the other is the gap.
        Coordinates: depth > 0 means further toward the fingertips than the URDF tool point;
        u is the offset along closing_local from that line's origin.
        """
        pb = self.pb
        probe = pb.createMultiBody(0, pb.createCollisionShape(pybullet.GEOM_SPHERE, radius=res / 2), -1, [0, 0, 5])
        us = np.arange(-half_span, half_span + 1e-9, res)
        free = -99

        def labels(d):
            g_pos, g_rot = self.frame(self.g_link)
            base = g_pos + g_rot @ (self.tcp_local + self.approach_local * d)
            c = g_rot @ self.closing_local
            lab = np.full(len(us), free)
            for i, u in enumerate(us):
                pb.resetBasePositionAndOrientation(probe, (base + c * u).tolist(), [0, 0, 0, 1])
                cps = pb.getClosestPoints(probe, self.body, 0.0)
                if cps:
                    lab[i] = cps[0][4]
            return lab

        def gap(lab):
            best, i, n = None, 0, len(lab)
            while i < n:
                if lab[i] != free:
                    i += 1
                    continue
                j = i
                while j < n and lab[j] == free:
                    j += 1
                if i > 0 and j < n and {lab[i - 1], lab[j]} == {self.g_link, self.jaw_link}:
                    lo, hi = us[i] - res / 2, us[j - 1] + res / 2  # faces of the free run
                    g = (lo, hi) if lab[i - 1] == self.g_link else (hi, lo)  # (fixed face, moving face)
                    if best is None or abs(g[1] - g[0]) > abs(best[1] - best[0]):
                        best = g
                i = j
            return best

        table = {}  # (pct, depth) -> (u_fixed, u_moving)
        for pct in pcts:
            q = np.zeros(6)
            q[5] = self.gripper_pct_to_rad(pct)
            self.set_q(q)
            for d in depths:
                g = gap(labels(d))
                if g is not None:
                    table[(pct, round(float(d), 4))] = g
        pb.removeBody(probe)
        self.set_q(np.zeros(6))
        if not table:
            raise RuntimeError("No jaw gap found by the gripper probe; check the collision meshes "
                               "(run prepare_assets.py) and the joint names.")

        ds = sorted({d for (_, d) in table})
        self.probe_depths = np.array(ds)
        self.probe_fixed_u = np.array([np.median([g[0] for (p, d), g in table.items() if d == dd]) for dd in ds])
        signs = [np.sign(g[1] - g[0]) for g in table.values()]
        self.grasp_sign = float(np.sign(np.sum(signs)) or 1.0)  # direction fixed jaw -> moving jaw
        self.tip_depth = float(max(ds))
        d_ref = self.tip_depth - 0.008
        self.width_pcts, self.widths = [], []
        for pct in pcts:
            cands = [(abs(d - d_ref), abs(g[1] - g[0])) for (p, d), g in table.items() if p == pct]
            if cands:
                self.width_pcts.append(pct)
                self.widths.append(min(cands)[1])
        self.probe_table = table

    def grasp_local(self, size, margin=0.004, clearance=0.003):
        """Point (gripper_link frame) to place at the object centre: object just clear of the
        fixed jaw's inner face, fingertips reaching below the centre but above the table."""
        insert = float(np.clip(size / 2 - clearance, 0.004, 0.015))
        d = self.tip_depth - insert
        u_f = float(np.interp(d, self.probe_depths, self.probe_fixed_u))
        return (self.tcp_local + self.approach_local * d
                + self.closing_local * (u_f + self.grasp_sign * (size / 2 + margin)))

    def min_open_pct(self, size, clearance=0.012):
        """Smallest probed gripper opening whose gap fits the object, or None."""
        for pct, w in zip(self.width_pcts, self.widths):
            if w >= size + clearance:
                return float(pct)
        return None
