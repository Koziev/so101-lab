"""
Automated calibration of the cell: wrist-camera mounting, board pose, stereo camera pose, joint zero offsets.

    # validation in simulation (PyBullet renders the board; known truth, injected offsets):
    python calib/autocalib.py --rig sim --setup setups/<name> --urdf assets_so101/so101_sim.urdf --out sim_check
    # the real cell (ChArUco board lying still in front of the robot, arm folded at rest):
    python calib/autocalib.py --rig real --setup setups/<name> --port /dev/ttyACM0

Real run: the arm moves by itself (torque on, slow moves, every transition checked for height and joint
limits) through bootstrap poses and then planned poses, captures the wrist image and the raw stereo frame in
each, and stops when the estimates converge. Keep a hand near the power switch for the first runs.
Results go into the setup folder: wrist_mount.json, the stereo pose (stereo_calib.json: pose_base),
the joint offsets (arm.json), and calibration/<time>/ with the observations, images and report.json.
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "camera"))
import charuco_board  # noqa: E402
from calib.estimate import CellCalibration, inv, rot_angle_deg, to_T  # noqa: E402
from calib.plan import Planner  # noqa: E402
from calib.session import run_session  # noqa: E402
from calib.urdf_fk import UrdfChain  # noqa: E402
from setup_io import JOINTS, Setup  # noqa: E402


def board_set(name):
    """[(label, board)] and {index: all inner corners (board frame, m)}."""
    boards = charuco_board.make_board_set(name)
    grids = {k: np.asarray(b.getChessboardCorners(), np.float64).reshape(-1, 3) for k, (_, b) in enumerate(boards)}
    return boards, grids


def detect(boards, gray, min_corners):
    """[(board index, obj, img)] for every board seen with >= min_corners corners."""
    return [(k, o.astype(np.float64), i.reshape(-1, 2).astype(np.float64))
            for k, o, i in charuco_board.detect_set(boards, gray, min_corners)]


# ------------------------------------------------------------------------------------------------ real rig
class RealRig:
    def __init__(self, setup, port, out_dir, boards, min_corners=8, settle_s=1.0, still_px=1.5):
        self.settle_s, self.still_px = settle_s, still_px
        self.tag = None                                   # stop number / recorded pose index of the next observation
        from calib.lerobot_helpers import make_robot, move_to, read_joints
        self._move, self._read = move_to, read_joints
        self.robot = make_robot(port, setup.info["arm_id"], 20.0)
        self.boards = boards
        self.min_corners, self.out_dir, self.n = min_corners, out_dir, 0
        self.wrist = self._open(setup.device("wrist"), 1280, 720)
        self.stereo = self._open(setup.device("stereo"), 2560, 720)
        self.rest_pose = self._read(self.robot)
        self.gripper_pct = 25.0
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    @staticmethod
    def _open(idx, w, h):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    @staticmethod
    def _grab(cap):
        for _ in range(4):
            cap.read()
        ok, f = cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return f

    def read_joints(self):
        return self._read(self.robot)

    def move_to(self, q_real):
        cur = self._read(self.robot)
        dist = float(np.max(np.abs(np.asarray(q_real)[:5] - cur[:5])))
        self._move(self.robot, np.asarray(q_real, float), seconds=max(1.5, dist / 25.0))     # <= 25 deg/s
        self._wait_still()

    def move_path(self, waypoints):
        """Through the waypoints without stopping (each segment <= 25 deg/s), then wait until the camera is still."""
        for w in waypoints:
            cur = self._read(self.robot)
            dist = float(np.max(np.abs(np.asarray(w)[:5] - cur[:5])))
            self._move(self.robot, np.asarray(w, float), seconds=max(0.4, dist / 25.0), settle=False)
        self._wait_still()

    def _wait_still(self, timeout=4.0):
        """Minimum pause, then until two consecutive wrist frames (small, grey) differ by < still_px on average:
        the camera on its bracket keeps swinging for a moment after the arm stops."""
        time.sleep(self.settle_s)
        t0, prev = time.time(), None
        while time.time() - t0 < timeout:
            ok, f = self.wrist.read()
            if not ok:
                break
            g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
            if prev is not None and float(np.mean(np.abs(g - prev))) < self.still_px:
                return
            prev = g

    def observe(self):
        q = np.mean([self._read(self.robot) for _ in range(5)], axis=0)
        wf, sf = self._grab(self.wrist), self._grab(self.stereo)
        stem = os.path.join(self.out_dir, "images", f"{self.n:03d}")
        cv2.imwrite(stem + "_wrist.png", wf)
        cv2.imwrite(stem + "_stereo.png", sf)
        json.dump({"q_real": q.tolist(), "phase": "stereo" if self.n < 3 else "stop", **(self.tag or {})},
                  open(stem + "_q.json", "w"))
        self.n += 1
        left = sf[:, : sf.shape[1] // 2]
        return {"q": q, "wrist": detect(self.boards, cv2.cvtColor(wf, cv2.COLOR_BGR2GRAY), self.min_corners),
                "stereo": detect(self.boards, cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), self.min_corners)}

    def close(self):
        self.robot.disconnect()
        self.wrist.release()
        self.stereo.release()


# ------------------------------------------------------------------------------------------------ PyBullet rig
class SimRig:
    """Kinematics from PyBullet (robot in one client), rendering of the board (second client) from a wrist
    camera at G X_true and a stereo left eye at S_true. Readings are q_true - d_true. Cameras are ideal
    pinholes (centred principal point, no distortion) with the focal lengths of the setup's cameras."""

    def __init__(self, urdf, boards, grids, fw, fs, truth, out_dir, min_corners=10, size_w=(1280, 720), size_s=(1280, 720)):
        import pybullet as p
        self.p, self.boards, self.grids, self.truth = p, boards, grids, truth
        self.min_corners, self.out_dir, self.n = min_corners, out_dir, 0
        self.kin = p.connect(p.DIRECT)
        self.robot = p.loadURDF(urdf, [0, 0, 0], useFixedBase=True, physicsClientId=self.kin)
        names = {p.getJointInfo(self.robot, i, physicsClientId=self.kin)[1].decode(): i
                 for i in range(p.getNumJoints(self.robot, physicsClientId=self.kin))}
        self.jidx = [names[j] for j in JOINTS[:5]]
        self.glink = [i for i in range(p.getNumJoints(self.robot, physicsClientId=self.kin))
                      if p.getJointInfo(self.robot, i, physicsClientId=self.kin)[12].decode() == "gripper_link"][0]
        self.scene = p.connect(p.DIRECT)
        self.size_w, self.size_s = size_w, size_s
        self.Kw = np.array([[fw, 0, size_w[0] / 2], [0, fw, size_w[1] / 2], [0, 0, 1.0]])
        self.Ks = np.array([[fs, 0, size_s[0] / 2], [0, fs, size_s[1] / 2], [0, 0, 1.0]])
        self.gripper_pct = 25.0
        self.q_true = np.array([0, -90, 90, 60, 0.0])
        self.rest_pose = np.r_[self.q_true - truth["d"], 25.0]
        self._build_scene()
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    def _build_scene(self):
        p, sc = self.p, self.scene
        p.createMultiBody(0, p.createCollisionShape(p.GEOM_PLANE, physicsClientId=sc),
                          p.createVisualShape(p.GEOM_BOX, halfExtents=[1, 1, 0.001], rgbaColor=[0.85, 0.75, 0.6, 1],
                                              physicsClientId=sc), [0, 0, -0.001], physicsClientId=sc)
        for k in range(len(self.boards)):
            self._add_board(k)

    def _add_board(self, k):
        p, sc = self.p, self.scene
        board = self.boards[k][1]
        sq = board.getSquareLength()
        nx, ny = board.getChessboardSize()
        W, H = nx * sq, ny * sq
        img = board.generateImage((nx * 100, ny * 100), marginSize=0)
        tex_path = os.path.join(self.out_dir, f"board_texture_{k}.png")
        cv2.imwrite(tex_path, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        R, t = self.truth["Z"][k][:3, :3], self.truth["Z"][k][:3, 3]
        quat = _quat(R)
        verts = [[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]]
        idx = [0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2]                     # both windings: visible from either side
        best = None
        for flip_u in (False, True):                                   # find the texture mapping that matches
            for flip_v in (False, True):
                uvs = [[(1 - u) if flip_u else u, (1 - v) if flip_v else v] for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))]
                vs = p.createVisualShape(p.GEOM_MESH, vertices=verts, indices=idx, uvs=uvs,
                                         normals=[[0, 0, -1]] * 4, physicsClientId=sc)   # face up: board z points down
                body = p.createMultiBody(0, -1, vs, t.tolist(), quat, physicsClientId=sc)
                p.changeVisualShape(body, -1, textureUniqueId=p.loadTexture(tex_path, physicsClientId=sc),
                                    rgbaColor=[1, 1, 1, 1], physicsClientId=sc)
                err = self._check_mapping(k)
                if err is not None and (best is None or err < best[0]):
                    if best is not None:
                        p.removeBody(best[1], physicsClientId=sc)
                    best = (err, body)
                else:
                    p.removeBody(body, physicsClientId=sc)
        if best is None or best[0] > 2.0:
            raise RuntimeError(f"could not render board {k} with a consistent texture mapping")
        print(f"sim board {k}: texture mapping verified (corner error {best[0]:.2f} px)")

    def _check_mapping(self, k):
        Z = self.truth["Z"][k]
        c = Z[:3, :3] @ self.grids[k].mean(axis=0) + Z[:3, 3]
        T = _look_at(c + np.array([0.0, 0.0, 0.40]), c, up_hint=np.array([1.0, 0, 0]))
        img = self._render(T, self.Ks, self.size_s)
        r = [x for x in detect(self.boards, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 12) if x[0] == k]
        if not r:
            return None
        _, obj, pix = r[0]
        Tbc = inv(T) @ Z
        proj = cv2.projectPoints(obj, cv2.Rodrigues(Tbc[:3, :3])[0], Tbc[:3, 3], self.Ks, np.zeros(5))[0].reshape(-1, 2)
        return float(np.median(np.linalg.norm(proj - pix, axis=1)))

    def _render(self, T_cam, K, size):
        p = self.p
        eye, z, y = T_cam[:3, 3], T_cam[:3, 2], T_cam[:3, 1]
        view = p.computeViewMatrix(eye.tolist(), (eye + z).tolist(), (-y).tolist(), physicsClientId=self.scene)
        fov = 2 * math.degrees(math.atan(size[1] / 2 / K[1, 1]))
        proj = p.computeProjectionMatrixFOV(fov, size[0] / size[1], 0.02, 3.0, physicsClientId=self.scene)
        _, _, rgba, _, _ = p.getCameraImage(size[0], size[1], view, proj, renderer=p.ER_TINY_RENDERER,
                                            physicsClientId=self.scene)
        return cv2.cvtColor(np.reshape(np.asarray(rgba, np.uint8), (size[1], size[0], 4))[:, :, :3], cv2.COLOR_RGB2BGR)

    def _gripper(self):
        p = self.p
        for i, j in zip(self.jidx, self.q_true):
            p.resetJointState(self.robot, i, math.radians(j), physicsClientId=self.kin)
        st = p.getLinkState(self.robot, self.glink, computeForwardKinematics=True, physicsClientId=self.kin)
        T = np.eye(4)
        T[:3, :3] = np.array(p.getMatrixFromQuaternion(st[5])).reshape(3, 3)
        T[:3, 3] = st[4]
        return T

    def read_joints(self):
        return np.r_[self.q_true - self.truth["d"], self.gripper_pct]

    def move_to(self, q_real):
        self.q_true = np.asarray(q_real, float)[:5] + self.truth["d"]

    def observe(self):
        G = self._gripper()
        wf = self._render(G @ self.truth["X"], self.Kw, self.size_w)
        sf = self._render(self.truth["S"], self.Ks, self.size_s)
        stem = os.path.join(self.out_dir, "images", f"{self.n:03d}")
        cv2.imwrite(stem + "_wrist.png", wf)
        if self.n == 0:
            cv2.imwrite(stem + "_stereo.png", sf)
        self.n += 1
        jaws = wf.copy()
        jaws[:, int(0.8 * self.size_w[0]):] = 128                      # the real jaws hide these regions
        jaws[int(0.7 * self.size_w[1]):, : int(0.2 * self.size_w[0])] = 128
        return {"q": self.read_joints(), "wrist": detect(self.boards, cv2.cvtColor(jaws, cv2.COLOR_BGR2GRAY), self.min_corners),
                "stereo": detect(self.boards, cv2.cvtColor(sf, cv2.COLOR_BGR2GRAY), self.min_corners)}

    def close(self):
        self.p.disconnect(self.kin)
        self.p.disconnect(self.scene)


def _quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return [x, y, z, w]


def _look_at(eye, target, roll_deg=0.0, up_hint=np.array([0, 0, 1.0])):
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up_hint)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    a = math.radians(roll_deg)
    R = np.c_[x, y, z] @ np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, eye
    return T


def sim_truth(seed=0, urdf=None):
    """A plausible cell: board 17 cm in front (face up, turned 8 deg), stereo camera 49 cm away and 34 cm high,
    wrist camera beside the fixed jaw looking ~50 deg off the gripper axis, joint offsets of a few degrees."""
    rng = np.random.default_rng(seed)
    from calib.gripper_axes import realistic_wrist_camera
    X, src = realistic_wrist_camera(urdf)
    print(f"sim truth: wrist camera defined from the gripper axes ({src})")
    a = math.radians(8)
    Rb = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1.0]]) @ np.array([[math.cos(a), -math.sin(a), 0],
                                                                     [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Z = {}
    for k, (y, yaw) in enumerate(((0.21, 3.0), (-0.01, -4.0))):          # board A to the left, B to the right
        a = math.radians(yaw)
        Rk = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1.0]]) @ np.array([[math.cos(a), -math.sin(a), 0],
                                                                         [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        Z[k] = np.eye(4)
        Z[k][:3, :3], Z[k][:3, 3] = Rk, [0.34, y, 0.0005]
    S = _look_at(np.array([0.53, 0.03, 0.34]), np.array([0.18, 0.0, 0.0]), roll_deg=-2.4)
    d = np.r_[0.5, rng.uniform(-4, 4, 3), 0.0]
    return {"X": X, "Z": Z, "S": S, "d": d}


# ------------------------------------------------------------------------------------------------ main
def intrinsics_from_views(views, image_size, drop_factor=2.5, min_views=12):
    """Wrist intrinsics from the session's own wrist views (board points only, no joint readings): the
    calibrate_wrist.py model (square pixels, no tangential distortion, k1 + k2), views far above the median
    error dropped. views: [(obj Nx3, img Nx2)], one planar board per entry. Returns a wrist_calib dict or None."""
    W, H = image_size
    objs = [np.asarray(o, np.float32).reshape(-1, 3) for o, _ in views]
    imgs = [np.asarray(i, np.float32).reshape(-1, 1, 2) for _, i in views]
    if len(objs) < min_views:
        return None
    flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3 | cv2.CALIB_USE_INTRINSIC_GUESS
    K0 = np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1.0]])

    def run(o, i):
        rms, K, D, rv, tv = cv2.calibrateCamera(o, i, (W, H), K0.copy(), None, flags=flags)
        e = [float(np.sqrt(np.mean(np.sum((cv2.projectPoints(ob, r, t, K, D)[0] - im) ** 2, axis=2))))
             for ob, im, r, t in zip(o, i, rv, tv)]
        return rms, K, D, e

    rms, K, D, err = run(objs, imgs)
    keep = [k for k, e in enumerate(err) if e <= drop_factor * np.median(err)]
    if len(keep) < len(err) and len(keep) >= min_views:
        objs, imgs = [objs[k] for k in keep], [imgs[k] for k in keep]
        rms, K, D, err = run(objs, imgs)
    cover = np.zeros((4, 4), int)
    for i in imgs:
        for x, y in i.reshape(-1, 2):
            cover[min(3, int(4 * y / H)), min(3, int(4 * x / W))] += 1
    return {"rms_px": float(rms), "K": K.tolist(), "dist": D.ravel().tolist(), "image_size": [W, H],
            "hfov_deg": float(2 * np.degrees(np.arctan(W / 2 / K[0, 0]))),
            "vfov_deg": float(2 * np.degrees(np.arctan(H / 2 / K[1, 1]))), "views": len(objs),
            "model": "square pixels, k1+k2", "coverage_4x4": cover.tolist(),
            "source": "automated calibration session (wrist views of the boards)"}


def recompute(args, setup, fk, boards, wrist, stereo, offsets):
    """The calibration from a session folder's saved images + joint readings, with the current detection."""
    import glob
    Ks, Ds = np.array(stereo["left"]["K"]), np.array(stereo["left"]["dist"])
    qfiles = sorted(glob.glob(os.path.join(args.from_session, "images", "*_q.json")))
    if not qfiles:
        raise SystemExit("no *_q.json joint readings in this session (recorded before this version): rerun the session")
    # pass 1: wrist detections (for automatic intrinsics)
    det_views, size = [], None
    for qf in qfiles:
        info = json.load(open(qf))
        if info.get("phase") == "stereo":
            continue
        wf = cv2.imread(qf[:-len("_q.json")] + "_wrist.png")
        if wf is None:
            continue
        size = (wf.shape[1], wf.shape[0])
        for b, obj, img in detect(boards, cv2.cvtColor(wf, cv2.COLOR_BGR2GRAY), 8):
            det_views.append((obj, img))
    wrist = choose_intrinsics(args, setup, wrist, det_views, size)
    Kw, Dw = np.array(wrist["K"]), np.array(wrist["dist"])
    cal = CellCalibration(fk, (Kw, Dw), (Ks, Ds), d_prior=np.r_[offsets[0], 0, 0, 0, 0],
                          free_intrinsics=args.free_intrinsics,
                          sag=("shoulder_lift", "elbow_flex", "wrist_flex") if args.sag else ())
    views = 0
    for qf in qfiles:
        info = json.load(open(qf))
        stem = qf[:-len("_q.json")]
        sf = cv2.imread(stem + "_stereo.png")
        if info.get("phase") == "stereo" and sf is not None:
            left = cv2.cvtColor(sf[:, : sf.shape[1] // 2], cv2.COLOR_BGR2GRAY)
            for b, obj, img in detect(boards, left, 10):
                cal.add_stereo(obj, img, board=b)
            continue
        wf = cv2.imread(stem + "_wrist.png")
        hits = detect(boards, cv2.cvtColor(wf, cv2.COLOR_BGR2GRAY), 8) if wf is not None else []
        for b, obj, img in hits:
            cal.add_sample(np.asarray(info["q_real"])[:5], obj, img, board=b, name=os.path.basename(stem))
        views += bool(hits)
        print(f"  {os.path.basename(stem)}: " + (", ".join(f"board {boards[b][0]} {len(o)} points" for b, o, _ in hits)
                                                  if hits else "no board"))
    print(f"stereo board views {len(cal.stereo_obs)}, wrist views with a board {views} of {len(qfiles) - 3}")
    if len(cal.samples) < 4:
        raise SystemExit("fewer than 4 wrist views with a board: not enough")
    res = cal.solve()
    from calib.session import _summary
    print(_summary(res, len(cal.samples)))
    # per-view errors: a few bad views (outliers) or a uniform misfit (systematic)?
    for rnd in range(3):
        rms = cal.per_view_rms(res)
        med = float(np.median(rms))
        print(f"\nper-view RMS error (px): median {med:.1f}, min {rms.min():.1f}, max {rms.max():.1f}")
        for k in np.argsort(-rms):
            s_ = cal.samples[k]
            print(f"  {s_['name']:8s} board {boards[s_['b']][0]}  {len(s_['obj']):3d} points  {rms[k]:6.1f} px"
                  + ("   <- outlier" if rms[k] > max(args.outlier_factor * med, 5.0) else ""))
        bad = [k for k in range(len(rms)) if rms[k] > max(args.outlier_factor * med, 5.0)]
        if not bad or args.outlier_factor <= 0 or len(cal.samples) - len(bad) < 8:
            break
        cal.samples = [s_ for k, s_ in enumerate(cal.samples) if k not in bad]
        print(f"dropping {len(bad)} outlier view(s) and solving again")
        res = cal.solve()
        print(_summary(res, len(cal.samples)))
    if res.get("sag"):
        from calib.urdf_fk import gravity_moments
        arms = np.array([gravity_moments(fk, np.radians(s_["q"] + res["d"]))[1:4] for s_ in cal.samples])
        print("gravity sag (deflection = c x moment arm of the gripper end): " + ", ".join(
            f"{j} {v['deg_per_m']:+.1f} +- {v['std']:.1f} deg/m (-> {abs(v['deg_per_m']) * np.abs(arms[:, k]).max():.2f} deg "
            f"at the largest moment arm {100 * np.abs(arms[:, k]).max():.0f} cm)" for k, (j, v) in enumerate(res["sag"].items())))
    if res.get("wrist_intrinsics"):
        w = res["wrist_intrinsics"]
        print(f"wrist intrinsics estimated: f {w['f_px']:.1f} px (+-{w['std']['f_px']:.1f}; separate calibration "
              f"{0.5 * (Kw[0, 0] + Kw[1, 1]):.1f}), principal point ({w['cx']:.1f}, {w['cy']:.1f}), k1 {w['k1']:.3f}, k2 {w['k2']:.3f}")
    if args.no_write:
        print("(--no-write: setup folder not changed)")
        return
    write_results(args, setup, cal, res, boards, args.from_session)


def choose_intrinsics(args, setup, wrist_file, det_views, size):
    """--wrist-intrinsics auto: computed from the session's views (written to the setup unless --no-write);
    file: the setup's wrist_calib.json; default: auto, falling back to the file if the views are insufficient."""
    if args.wrist_intrinsics == "file":
        if wrist_file is None:
            raise SystemExit("--wrist-intrinsics file, but the setup has no wrist_calib.json")
        return wrist_file
    auto = intrinsics_from_views(det_views, size) if size else None
    if auto is None:
        if wrist_file is None:
            raise SystemExit("too few wrist views for automatic intrinsics and no wrist_calib.json in the setup")
        print(f"too few wrist views ({len(det_views)}) for automatic intrinsics: using the setup's wrist_calib.json")
        return wrist_file
    K = np.array(auto["K"])
    print(f"wrist intrinsics from this session: {auto['views']} views, error {auto['rms_px']:.2f} px, f {K[0, 0]:.1f} px, "
          f"principal point ({K[0, 2]:.1f}, {K[1, 2]:.1f}), k1 {auto['dist'][0]:.3f}, k2 {auto['dist'][1]:.3f}"
          + (f" (setup's file: f {wrist_file['K'][0][0]:.1f} px)" if wrist_file else ""))
    empty = sum(v == 0 for row in auto["coverage_4x4"] for v in row)
    if empty > 4:
        print(f"  note: {empty} of 16 image cells have no board points (the jaws hide some; more spread helps)")
    if not args.no_write:
        setup.save("wrist_calib.json", auto, note="wrist intrinsics from the automated calibration")
    return auto


def write_results(args, setup, cal, res, boards, out):
    """report.json + observations.json into the session folder; mounting, stereo pose and offsets into the setup."""
    report = {"poses": len(cal.samples), "wrist_rms_px": res["wrist_rms_px"], "stereo_rms_px": res["stereo_rms_px"],
              "std": res["std"], "offsets_deg": dict(zip(JOINTS[:5], res["d"].tolist())),
              "X_cam2gripper": res["X"].tolist(),
              "Z_board2base": {boards[b][0]: res["Z"][b].tolist() for b in res["Z"]},
              "views_per_board": {boards[b][0]: n for b, n in res["views_per_board"].items()},
              "S_stereo_left2base": res["S"].tolist() if cal.stereo_obs else None}
    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=2, default=str)
    setup.save("wrist_mount.json", {"R_cam2gripper": res["X"][:3, :3].tolist(), "t_cam2gripper_m": res["X"][:3, 3].tolist(),
                                    "std": {"pos_mm": res["std"]["X_pos_mm"], "rot_deg": res["std"]["X_rot_deg"]},
                                    "session": os.path.basename(os.path.normpath(out)), "frame": "gripper_link (URDF)"},
               note="automated calibration")
    if cal.stereo_obs:
        st = setup.load("stereo_calib.json")
        st["pose_base"] = {"T_left2base": res["S"].tolist(), "std_mm": res["std"]["S_pos_mm"],
                           "std_deg": res["std"]["S_rot_deg"], "session": os.path.basename(os.path.normpath(out))}
        setup.save("stereo_calib.json", st, note="stereo pose from the automated calibration")
    arm = setup.arm
    for j, v in zip(JOINTS[1:4], res["d"][1:4]):
        arm["offsets"][j] = round(float(v), 2)
    arm.setdefault("notes", {})["fitted"] = f"lift/elbow/wrist_flex fitted in session {os.path.basename(os.path.normpath(out))}"
    setup.save("arm.json", arm, note="joint offsets from the automated calibration")
    print(f"results written into {setup.path} (wrist_mount.json, stereo_calib.json pose_base, arm.json); "
          f"report in {out}/report.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", choices=["sim", "real"], required=True)
    ap.add_argument("--setup", required=True)
    ap.add_argument("--urdf", default=os.path.join(ROOT, "assets_so101", "so101_sim.urdf"))
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--out", default="", help="session folder (default: <setup>/calibration/<time>)")
    ap.add_argument("--max-poses", type=int, default=40)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="real rig: plan and check poses, do not move")
    ap.add_argument("--outlier-factor", type=float, default=3.0,
                    help="--from-session: drop views whose error exceeds this x the median (0 = keep all)")
    ap.add_argument("--no-write", action="store_true", help="--from-session: analyse only, do not update the setup")
    ap.add_argument("--wrist-intrinsics", choices=["auto", "file"], default="auto",
                    help="auto (default): compute the wrist intrinsics from the session's own views first (falls back "
                         "to the setup's wrist_calib.json if the views are too few); file: use wrist_calib.json")
    ap.add_argument("--sag", action="store_true",
                    help="--from-session: also estimate gravity sag of shoulder_lift, elbow_flex, wrist_flex")
    ap.add_argument("--free-intrinsics", action="store_true",
                    help="--from-session: also estimate the wrist camera's focal length, principal point, k1, k2")
    ap.add_argument("--from-session", default="", help="recompute from a session folder's saved images and joint "
                    "readings (no arm, no cameras); results are written into the setup folder as after a real run")
    ap.add_argument("--boards", default="2xA4", help="board set (camera/charuco_board.py BOARD_SETS)")
    ap.add_argument("--board-cm", type=float, default=18.0,
                    help="distance of the boards' centre (middle between the sheets) in front of the pan axis (cm)")
    ap.add_argument("--board-side-cm", type=float, default=0.0, help="board centre to the robot's left (+) / right (-)")
    ap.add_argument("--clearance-mm", type=float, default=10.0,
                    help="the jaws stay at least this much higher than the lowest jaw point you taught "
                         "(applies to every pose and to every point of every move)")
    ap.add_argument("--no-workspace", action="store_true", help="ignore workspace.json (geometric safety only)")
    ap.add_argument("--mode", choices=["replay", "planned"], default="replay",
                    help="replay: travel along the taught recording and stop on it (default, needs workspace.json); "
                         "planned: the older planner-based session")
    ap.add_argument("--stops", type=int, default=60, help="replay: number of stops along the recording")
    ap.add_argument("--revisit-every", type=int, default=8,
                    help="replay: after every N-th stop, revisit the stop 5 earlier (repeatability; 0 = off)")
    ap.add_argument("--settle-s", type=float, default=1.0, help="minimum pause after every motion (s)")
    ap.add_argument("--still-px", type=float, default=1.5,
                    help="then wait (max 4 s) until consecutive wrist frames differ by less than this (mean, grey levels)")
    ap.add_argument("--stop-radius-cm", type=float, default=15.0,
                    help="replay: stops only where the gripper is within this distance of the boards' centre")
    args = ap.parse_args()

    setup = Setup(args.setup)
    stereo, wrist = setup.load("stereo_calib.json"), setup.load("wrist_calib.json")
    if stereo is None:
        raise SystemExit("the setup needs stereo_calib.json (stereo intrinsics) first")
    if wrist is None and not args.from_session and args.rig == "real":
        print("no wrist_calib.json yet: run the session, then compute everything with --from-session "
              "(the wrist intrinsics are then derived from the session's own views)")
    out = args.out or setup.file(os.path.join("calibration", time.strftime("%Y%m%d_%H%M%S")))
    os.makedirs(out, exist_ok=True)
    fk = UrdfChain(args.urdf, "gripper_link")
    boards, grids = board_set(args.boards)
    offsets = np.asarray(setup.offsets()[:5])
    if args.from_session:
        recompute(args, setup, fk, boards, wrist, stereo, offsets)
        return
    if args.rig == "sim":
        truth = sim_truth(args.seed, args.urdf)
        rig = SimRig(args.urdf, boards, grids, float(wrist["K"][0][0]), float(stereo["left"]["K"][0][0]), truth, out)
        Kw, Dw, Ks, Ds = rig.Kw, np.zeros(5), rig.Ks, np.zeros(5)
        d_prior = np.r_[truth["d"][0], 0, 0, 0, 0]                       # pan known (physical check)
    else:
        rig = RealRig(setup, args.port, out, boards, settle_s=args.settle_s, still_px=args.still_px)
        Kw, Dw = np.array(wrist["K"]), np.array(wrist["dist"])
        Ks, Ds = np.array(stereo["left"]["K"]), np.array(stereo["left"]["dist"])
        d_prior = np.r_[offsets[0], 0, 0, 0, 0]
    ws_data = None if args.no_workspace else setup.load("workspace.json")
    if ws_data is not None:
        from calib.workspace import Workspace
        from calib.gripper_axes import jaw_points
        ws = Workspace(ws_data["points"])
        planner = Planner(fk, Kw, Dw, board_points=grids, seed=args.seed, workspace=ws, d=d_prior)
        pts, src = jaw_points(args.urdf)
        lowest = planner.set_jaw_rule(pts, [q + d_prior for q in ws.P], args.clearance_mm / 1000)
        n0 = len(ws.P)
        planner.ws_raw = ws                                             # leaving / returning to rest
        print(f"taught poses form {ws.components()} connected part(s) (1 is expected: one continuous recording)")
        ws = ws.filtered(lambda p: planner.jaw_height(p + d_prior) >= planner.min_jaw_z)
        planner.ws = ws
        print(f"taught workspace: {len(ws.P)} of {n0} poses kept; lowest taught jaw point {100 * lowest:.1f} cm, "
              f"jaws kept above {100 * planner.min_jaw_z:.1f} cm (clearance {args.clearance_mm:.0f} mm; gripper geometry: {src})")
        if "toy" in src:
            raise SystemExit("the gripper geometry could not be read from the simulator (needed for the height rule)")
        rig.safe_pose_sim = max((q + d_prior for q in ws.P), key=lambda q: planner.jaw_height(q))
    else:
        if args.rig == "real":
            print("WARNING: no taught workspace (workspace.json); using geometric safety rules only")
        planner = Planner(fk, Kw, Dw, board_points=grids, seed=args.seed)
        rig.safe_pose_sim = max(planner.random_safe(400), key=lambda q: fk.pose(np.radians(q))[2, 3])
    cal = CellCalibration(fk, (Kw, Dw), (Ks, Ds), d_prior=d_prior)
    if args.rig == "sim":                                                # rough guess, as a user would give it
        centre = np.mean([truth["Z"][k][:3, :3] @ grids[k].mean(axis=0) + truth["Z"][k][:3, 3] for k in grids], axis=0) \
            + np.array([0.02, -0.02, 0.0])
    else:
        f, pan = planner.forward()
        left = np.array([-f[1], f[0], 0.0])
        centre = pan + f * args.board_cm / 100 + left * args.board_side_cm / 100
    if args.dry_run and (args.mode != "replay" or ws_data is None):
        poses = planner.bootstrap_poses(centre)
        print(f"dry run: {len(poses)} safe bootstrap poses above the board (simulator convention), first 10:")
        for q in poses[:10]:
            jaw = f", lowest jaw point {100 * planner.jaw_height(q):.1f} cm" if planner.jaw_points is not None else ""
            print("  ", np.round(q, 1), f"gripper link {100 * fk.pose(np.radians(q))[2, 3]:.1f} cm{jaw}")
        rig.close()
        return
    try:
        if args.rig == "real" and not args.dry_run:
            input("REAL ARM: the arm will move by itself through ~40-80 poses above the board (slowly, torque on). "
                  "Board lying still, arm folded at rest, workspace clear, hand near the power switch. Enter...")
        if args.mode == "replay" and ws_data is not None:
            from calib.session import choose_stops, run_replay
            P = np.asarray(ws_data["points"], float)
            keep = lambda i: (planner.jaw_height(P[i, :5] + d_prior) >= planner.min_jaw_z and np.linalg.norm(
                fk.pose(np.radians(P[i, :5] + d_prior))[:2, 3] - centre[:2]) <= args.stop_radius_cm / 100)
            from calib.session import with_revisits
            stops = with_revisits(choose_stops(P, args.stops, keep), every=args.revisit_every)
            print(f"replay: {len(stops)} stops along the {len(P)}-pose recording "
                  f"(gripper within {args.stop_radius_cm:.0f} cm of the boards' centre, jaws above the clearance limit)")
            if args.dry_run:
                rig.close()
                return
            res = run_replay(rig, cal, P, stops, jaw_ok=lambda q: planner.jaw_height(np.asarray(q)[:5] + d_prior) >= planner.min_jaw_z)
        else:
            res = run_session(rig, cal, planner, batch=args.batch, max_poses=args.max_poses, board_centre=centre)
    finally:
        rig.close()

    report = {"poses": res["poses"], "visited": res["visited"], "minutes": res["minutes"],
              "wrist_rms_px": res["wrist_rms_px"], "stereo_rms_px": res["stereo_rms_px"], "std": res["std"],
              "offsets_deg": dict(zip(JOINTS[:5], res["d"].tolist())), "X_cam2gripper": res["X"].tolist(),
              "Z_board2base": {boards[b][0]: res["Z"][b].tolist() for b in res["Z"]},
              "views_per_board": {boards[b][0]: n for b, n in res["views_per_board"].items()},
              "S_stereo_left2base": res["S"].tolist() if cal.stereo_obs else None}
    if args.rig == "sim":
        t = rig.truth
        report["truth_errors"] = {
            "mount_mm": float(1000 * np.linalg.norm(res["X"][:3, 3] - t["X"][:3, 3])),
            "mount_deg": rot_angle_deg(t["X"][:3, :3].T @ res["X"][:3, :3]),
            "stereo_mm": float(1000 * np.linalg.norm(res["S"][:3, 3] - t["S"][:3, 3])) if cal.stereo_obs else None,
            "stereo_deg": rot_angle_deg(t["S"][:3, :3].T @ res["S"][:3, :3]),
            "offsets_deg": (res["d"] - t["d"]).round(3).tolist(),
            "boards_mm": {boards[b][0]: float(1000 * np.linalg.norm(res["Z"][b][:3, 3] - t["Z"][b][:3, 3])) for b in res["Z"]}}
        print("\nSIMULATION CHECK against the known truth:", json.dumps(report["truth_errors"], indent=1))
    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=2)
    json.dump([{"q": s["q"].tolist(), "board": s["b"], "obj": s["obj"].tolist(), "img": s["img"].tolist()} for s in cal.samples]
              + [{"stereo": True, "board": o["b"], "obj": o["obj"].tolist(), "img": o["img"].tolist()} for o in cal.stereo_obs],
              open(os.path.join(out, "observations.json"), "w"))
    if args.rig == "real":
        setup.save("wrist_mount.json", {"R_cam2gripper": res["X"][:3, :3].tolist(), "t_cam2gripper_m": res["X"][:3, 3].tolist(),
                                        "std": {"pos_mm": res["std"]["X_pos_mm"], "rot_deg": res["std"]["X_rot_deg"]},
                                        "session": os.path.basename(out), "frame": "gripper_link (URDF)"},
                   note="automated calibration")
        if cal.stereo_obs:
            st = setup.load("stereo_calib.json")
            st["pose_base"] = {"T_left2base": res["S"].tolist(), "std_mm": res["std"]["S_pos_mm"],
                               "std_deg": res["std"]["S_rot_deg"], "session": os.path.basename(out)}
            setup.save("stereo_calib.json", st, note="stereo pose from the automated calibration")
        else:
            print("WARNING: the stereo camera never detected the board; its pose was NOT updated")
        arm = setup.arm
        for j, v in zip(JOINTS[1:4], res["d"][1:4]):
            arm["offsets"][j] = round(float(v), 2)
        arm.setdefault("notes", {})["fitted"] = f"lift/elbow/wrist_flex fitted in session {os.path.basename(out)}"
        setup.save("arm.json", arm, note="joint offsets from the automated calibration")
    print(f"\n{res['poses']} views in {res['minutes']:.1f} min; report in {out}/report.json")


if __name__ == "__main__":
    main()
