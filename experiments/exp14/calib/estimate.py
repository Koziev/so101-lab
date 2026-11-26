"""
Joint estimation of the cell geometry from observations of ONE fixed ChArUco board:

  X  wrist camera -> gripper_link          (the camera's mounting on the gripper)
  Z  board -> robot base                    (where the board lies)
  S  stereo left eye -> robot base          (the stereo camera's pose; raw left-eye images)
  d  joint zero offsets (deg), simulator angle = real reading + d

Observations: for every arm pose, the joint readings and the ChArUco corners seen by the wrist camera; and,
for the static stereo camera, the corners seen in its raw left-eye image (one or several frames).
Model: wrist pixel = project(K_w, D_w, inv(G(q + d) X) Z p), stereo pixel = project(K_s, D_s, inv(S) Z p),
with G from the URDF chain. Gauge: shoulder_pan's offset is not observable from these data (turning the whole
world about the pan axis changes nothing) and is fixed to a prior (physical check); wrist_roll's offset is
absorbed by X and fixed to 0. A weak prior keeps the board horizontal (it lies on the table).
"""
import math

import cv2
import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


# ------------------------------------------------------------------------------------------------ helpers
def to_T(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float).reshape(3))[0]
    T[:3, 3] = np.asarray(tvec, float).reshape(3)
    return T


def from_T(T):
    return cv2.Rodrigues(T[:3, :3])[0].reshape(3), T[:3, 3].copy()


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def rot_angle_deg(R):
    return math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def average_pose(Ts):
    """Mean of rigid transforms: chordal rotation mean + mean translation."""
    U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.mean([T[:3, 3] for T in Ts], axis=0)
    return T


def pnp(obj, img, K, D):
    ok, r, t = cv2.solvePnP(np.asarray(obj, np.float64), np.asarray(img, np.float64).reshape(-1, 1, 2), K, D,
                            flags=cv2.SOLVEPNP_ITERATIVE)
    return to_T(r, t) if ok else None


def hand_eye(Gs, Cs):
    """AX = XB, A = inv(G_j) G_i, B = C_j inv(C_i) (Park & Martin over all pairs). G: gripper->base, C: board->cam."""
    As, Bs = [], []
    for i in range(len(Gs)):
        for j in range(i + 1, len(Gs)):
            As.append(inv(Gs[j]) @ Gs[i])
            Bs.append(Cs[j] @ inv(Cs[i]))
    al = np.array([cv2.Rodrigues(A[:3, :3])[0].reshape(3) for A in As])
    be = np.array([cv2.Rodrigues(B[:3, :3])[0].reshape(3) for B in Bs])
    U, _, Vt = np.linalg.svd(be.T @ al)
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    M = np.vstack([A[:3, :3] - np.eye(3) for A in As])
    v = np.concatenate([R @ B[:3, 3] - A[:3, 3] for A, B in zip(As, Bs)])
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, np.linalg.lstsq(M, v, rcond=None)[0]
    return X


# ------------------------------------------------------------------------------------------------ estimator
class CellCalibration:
    """
    fk          calib.urdf_fk.UrdfChain (base -> gripper_link)
    wrist       (K, D) of the wrist camera;  stereo (K, D) of the raw left eye
    d_prior     prior offsets (deg, 5 joints, simulator = real + d); pan's entry is kept fixed
    free        joints whose offsets are estimated (default lift, elbow, wrist_flex)
    Several boards (each rigid, with its own pose Z_b) are supported: observations carry a board index.
    """

    def __init__(self, fk, wrist, stereo, d_prior=None, free=("shoulder_lift", "elbow_flex", "wrist_flex"),
                 board_level_deg=2.0, offset_prior_deg=10.0, free_intrinsics=False, sag=()):
        self.fk, (self.Kw, self.Dw), (self.Ks, self.Ds) = fk, wrist, stereo
        self.Kw, self.Dw = np.asarray(self.Kw, float), np.asarray(self.Dw, float).ravel()
        # optionally estimate the wrist intrinsics too: f (fx = fy), cx, cy, k1, k2
        self.free_intrinsics = free_intrinsics
        self.Kw0, self.Dw0 = self.Kw.copy(), self.Dw.copy()
        # optional gravity sag: joint j deflects by c_j * (moment arm of the gripper end about joint j), deg per m
        self.sag = [JOINTS.index(j) for j in sag]
        self.c = np.zeros(5)
        self.d_prior = np.zeros(5) if d_prior is None else np.asarray(d_prior, float)
        self.free = [JOINTS.index(j) for j in free]
        self.board_level = math.radians(board_level_deg)
        self.offset_prior = offset_prior_deg
        self.samples, self.stereo_obs = [], []

    # observations ---------------------------------------------------------------------------------------
    def add_sample(self, q_real_deg, obj, img, board=0, name=""):
        self.samples.append({"q": np.asarray(q_real_deg, float)[:5], "obj": np.asarray(obj, np.float64),
                             "img": np.asarray(img, np.float64).reshape(-1, 2), "b": int(board), "name": name})

    def per_view_rms(self, res):
        """RMS reprojection error (px) of every wrist view under the solution `res`."""
        return np.array([float(np.sqrt(np.mean(self._sample_err(s, res["X"], res["Z"], res["d"]) ** 2)))
                         for s in self.samples])

    def add_stereo(self, obj, img, board=0):
        self.stereo_obs.append({"obj": np.asarray(obj, np.float64), "img": np.asarray(img, np.float64).reshape(-1, 2),
                                "b": int(board)})

    @property
    def boards(self):
        return sorted({o["b"] for o in self.samples} | {o["b"] for o in self.stereo_obs})

    def G(self, q_real_deg, d, c=None):
        q = np.asarray(q_real_deg, float) + d
        c = self.c if c is None else c
        if self.sag:
            from calib.urdf_fk import gravity_moments
            q = q + c * gravity_moments(self.fk, np.radians(q))
        return self.fk.pose(np.radians(q))

    # parameters: X | Z for each board in self.boards | S | free offsets ----------------------------------------
    def pack(self, X, Z, S, d):
        Zs = [Z[b] for b in self.boards]
        x = np.concatenate([*from_T(X), *[v for z in Zs for v in from_T(z)], *from_T(S), d[self.free],
                            self.c[self.sag]])
        if self.free_intrinsics:
            f = 0.5 * (self.Kw[0, 0] + self.Kw[1, 1])
            x = np.concatenate([x, [f, self.Kw[0, 2], self.Kw[1, 2], self.Dw[0], self.Dw[1]]])
        return x

    def _intrinsics(self, p):
        if not self.free_intrinsics:
            return self.Kw, self.Dw
        f, cx, cy, k1, k2 = p[-5:]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        D = self.Dw0.copy()
        D[0], D[1] = k1, k2
        return K, D

    def unpack(self, p):
        nb = len(self.boards)
        X = to_T(p[0:3], p[3:6])
        Z = {b: to_T(p[6 + 6 * k: 9 + 6 * k], p[9 + 6 * k: 12 + 6 * k]) for k, b in enumerate(self.boards)}
        o = 6 + 6 * nb
        S = to_T(p[o:o + 3], p[o + 3:o + 6])
        d = self.d_prior.copy()
        d[self.free] = p[o + 6:o + 6 + len(self.free)]
        if self.sag:
            c = np.zeros(5)
            c[self.sag] = p[o + 6 + len(self.free):o + 6 + len(self.free) + len(self.sag)]
            self.c = c
        return X, Z, S, d

    # initial guess --------------------------------------------------------------------------------------
    def initialise(self):
        obs = []
        for s in self.samples:
            C = pnp(s["obj"], s["img"], self.Kw, self.Dw)
            if C is not None:
                obs.append((s["b"], self.G(s["q"], self.d_prior), C))
        if len(obs) < 4:
            raise ValueError(f"only {len(obs)} usable wrist views; at least 4 are needed")
        # hand-eye from pose pairs that saw the SAME board
        best = max({b for b, _, _ in obs}, key=lambda b: sum(1 for o in obs if o[0] == b))
        pairsG, pairsC = [], []
        for b in {b for b, _, _ in obs}:
            Gb = [G for bb, G, _ in obs if bb == b]
            Cb = [C for bb, _, C in obs if bb == b]
            if len(Gb) >= 2:
                pairsG.append(Gb)
                pairsC.append(Cb)
        X = hand_eye_groups(pairsG, pairsC)
        Z = {}
        for b in {b for b, _, _ in obs}:
            Z[b] = average_pose([G @ X @ C for bb, G, C in obs if bb == b])
        S = None
        stereo_T = [(o["b"], pnp(o["obj"], o["img"], self.Ks, self.Ds)) for o in self.stereo_obs]
        Ss = [Z[b] @ inv(T) for b, T in stereo_T if T is not None and b in Z]
        S = average_pose(Ss) if Ss else np.eye(4)                   # placeholder without stereo observations
        for b, T in stereo_T:                                          # boards seen only by the stereo camera
            if T is not None and b not in Z and Ss:
                Z[b] = S @ T
        for b in self.boards:
            Z.setdefault(b, Z[best])
        return X, Z, S, self.d_prior.copy()

    # residuals ------------------------------------------------------------------------------------------
    def residuals(self, p):
        X, Z, S, d = self.unpack(p)
        Kw, Dw = self._intrinsics(p)
        res = []
        for s in self.samples:
            T = inv(self.G(s["q"], d) @ X) @ Z[s["b"]]                # board -> wrist camera
            r, t = from_T(T)
            proj, _ = cv2.projectPoints(s["obj"], r, t, Kw, Dw)
            res.append((proj.reshape(-1, 2) - s["img"]).ravel())
        for o in self.stereo_obs:
            r, t = from_T(inv(S) @ Z[o["b"]])
            proj, _ = cv2.projectPoints(o["obj"], r, t, self.Ks, self.Ds)
            res.append((proj.reshape(-1, 2) - o["img"]).ravel())
        for b in self.boards:
            n = Z[b][:3, 2]                                            # each board lies flat on the table
            res.append(np.array([n[0], n[1]]) / self.board_level)
        res.append((d[self.free] - self.d_prior[self.free]) / self.offset_prior)
        if not self.stereo_obs:                       # nothing determines S: hold it at its start value
            o = 6 + 6 * len(self.boards)
            res.append((p[o:o + 6] - self._s_anchor) * 1000.0)
        if self.sag:                                  # weak prior: sag of the order of a few deg per 10 cm
            res.append(self.c[self.sag] / 100.0)
        if self.free_intrinsics:                      # weak priors around the separate intrinsic calibration
            f0 = 0.5 * (self.Kw0[0, 0] + self.Kw0[1, 1])
            res.append(np.array([(p[-5] - f0) / 100.0, (p[-4] - self.Kw0[0, 2]) / 50.0, (p[-3] - self.Kw0[1, 2]) / 50.0,
                                 (p[-2] - self.Dw0[0]) / 0.5, (p[-1] - self.Dw0[1]) / 1.0]))
        return np.concatenate(res)

    def solve(self, init=None):
        from scipy.optimize import least_squares
        X0, Z0, S0, d0 = init or self.initialise()
        Z0 = dict(Z0)
        for b in self.boards:                                          # a board seen for the first time
            if b not in Z0:
                Z0[b] = self.initialise()[1].get(b, next(iter(Z0.values())))
        x0 = self.pack(X0, Z0, S0, d0)
        o = 6 + 6 * len(self.boards)
        self._s_anchor = x0[o:o + 6].copy()
        sol = least_squares(self.residuals, x0, loss="soft_l1", f_scale=2.0, x_scale="jac", max_nfev=400)
        X, Z, S, d = self.unpack(sol.x)
        if self.free_intrinsics:
            self.Kw, self.Dw = self._intrinsics(sol.x)
        J = sol.jac
        m, k = J.shape
        sigma2 = float(np.sum(sol.fun ** 2) / max(1, m - k))
        try:
            std = np.sqrt(np.clip(np.diag(np.linalg.inv(J.T @ J) * sigma2), 0, None))
        except np.linalg.LinAlgError:
            std = np.full(k, np.nan)
        nb = len(self.boards)
        o = 6 + 6 * nb
        wr = np.concatenate([self._sample_err(s, X, Z, d) for s in self.samples])
        st = np.concatenate([self._stereo_err(ob, Z, S) for ob in self.stereo_obs]) if self.stereo_obs else np.array([])
        intr = None
        if self.free_intrinsics:
            intr = {"f_px": float(self.Kw[0, 0]), "cx": float(self.Kw[0, 2]), "cy": float(self.Kw[1, 2]),
                    "k1": float(self.Dw[0]), "k2": float(self.Dw[1]),
                    "std": {"f_px": float(std[-5]), "cx": float(std[-4]), "cy": float(std[-3])}}
        sag = None
        if self.sag:
            o2 = o + 6 + len(self.free)
            sag = {JOINTS[j]: {"deg_per_m": float(self.c[j]), "std": float(std[o2 + k])} for k, j in enumerate(self.sag)}
        return {"X": X, "Z": Z, "S": S, "d": d, "success": bool(sol.success), "boards": self.boards,
                "wrist_intrinsics": intr, "sag": sag,
                "wrist_rms_px": float(np.sqrt(np.mean(wr ** 2))),
                "stereo_rms_px": float(np.sqrt(np.mean(st ** 2))) if st.size else None,
                "views_per_board": {b: sum(1 for s in self.samples if s["b"] == b) for b in self.boards},
                "std": {"X_rot_deg": float(math.degrees(np.linalg.norm(std[0:3]))), "X_pos_mm": float(1000 * np.linalg.norm(std[3:6])),
                        "Z_pos_mm": {b: float(1000 * np.linalg.norm(std[9 + 6 * k:12 + 6 * k])) for k, b in enumerate(self.boards)},
                        "S_rot_deg": float(math.degrees(np.linalg.norm(std[o:o + 3]))) if self.stereo_obs else None,
                        "S_pos_mm": float(1000 * np.linalg.norm(std[o + 3:o + 6])) if self.stereo_obs else None,
                        "d_deg": dict(zip([JOINTS[i] for i in self.free], std[o + 6:o + 6 + len(self.free)].tolist()))},
                "x": sol.x}

    def _sample_err(self, s, X, Z, d):
        r, t = from_T(inv(self.G(s["q"], d) @ X) @ Z[s["b"]])
        proj, _ = cv2.projectPoints(s["obj"], r, t, self.Kw, self.Dw)
        return np.linalg.norm(proj.reshape(-1, 2) - s["img"], axis=1)

    def _stereo_err(self, o, Z, S):
        r, t = from_T(inv(S) @ Z[o["b"]])
        proj, _ = cv2.projectPoints(o["obj"], r, t, self.Ks, self.Ds)
        return np.linalg.norm(proj.reshape(-1, 2) - o["img"], axis=1)


def hand_eye_groups(Gs_groups, Cs_groups):
    """hand_eye() over several boards: AX = XB pairs are formed only within the same board."""
    As, Bs = [], []
    for Gs, Cs in zip(Gs_groups, Cs_groups):
        for i in range(len(Gs)):
            for j in range(i + 1, len(Gs)):
                As.append(inv(Gs[j]) @ Gs[i])
                Bs.append(Cs[j] @ inv(Cs[i]))
    al = np.array([cv2.Rodrigues(A[:3, :3])[0].reshape(3) for A in As])
    be = np.array([cv2.Rodrigues(B[:3, :3])[0].reshape(3) for B in Bs])
    U, _, Vt = np.linalg.svd(be.T @ al)
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    M = np.vstack([A[:3, :3] - np.eye(3) for A in As])
    v = np.concatenate([R @ B[:3, 3] - A[:3, 3] for A, B in zip(As, Bs)])
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, np.linalg.lstsq(M, v, rcond=None)[0]
    return X


def converged(prev, cur, pos_mm=1.0, rot_deg=0.2, off_deg=0.3):
    """True if X, S and the offsets changed by less than the tolerances AND their uncertainties are below them."""
    if prev is None:
        return False
    dX = 1000 * np.linalg.norm(cur["X"][:3, 3] - prev["X"][:3, 3])
    dXr = rot_angle_deg(prev["X"][:3, :3].T @ cur["X"][:3, :3])
    dS = 1000 * np.linalg.norm(cur["S"][:3, 3] - prev["S"][:3, 3]) if cur["std"]["S_pos_mm"] is not None else 0.0
    dd = float(np.max(np.abs(cur["d"] - prev["d"])))
    s = cur["std"]
    return (dX < pos_mm and dXr < rot_deg and dS < pos_mm and dd < off_deg and s["X_pos_mm"] < pos_mm
            and s["X_rot_deg"] < rot_deg and (s["S_pos_mm"] is None or s["S_pos_mm"] < 2 * pos_mm)
            and max(s["d_deg"].values(), default=0) < off_deg)
