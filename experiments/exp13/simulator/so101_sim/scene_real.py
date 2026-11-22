"""
The real cell as the simulator's scene: the stereo camera's own picture of the empty cell (the setup's
stereo_background.png) is the background of every simulated image; only the arm, the objects and their shadows
are rendered and drawn on top.

Why: the model reads more from an image than the objects. With a synthetic board, backdrop and lighting it can
tie its answers to things that do not exist in the real picture (the simulated board's edges and corners) and
is then off by a constant amount on the real arm, although the arm, the camera and the objects are modelled
correctly. With the real background, everything that does not move is exactly what the real camera will show.

Geometry: the background is rectified and zoomed exactly like the images the model gets on the real arm
(record_real / predict), which makes it the picture of a pinhole camera at the calibrated pose. For a simulated
camera at another pose (camera randomisation) it is warped with the homography of the BOARD PLANE between the
two views: exact for everything on the board, approximate for the walls behind it.

The area around the arm's base (the real base, arm remnants of the median) comes in three variants, one per
episode: untouched, remnants painted over, base and remnants painted over. On top of the real picture every
episode gets a few random incidental disturbances on the board (soft patches, thin strokes): things that are
always in the same place in training and never in reality - or the reverse - are what a model latches on to,
so what is not the arm or an object is made to vary.

Photometry: the background keeps the real camera's look (a little gain / white-balance / blur / noise jitter).
The rendered arm and objects are scaled so that the brightest faces of the (white, printed) arm reach a
near-white level like the real arm; edges are blended softly.
"""
import json
import math
import os

import cv2
import numpy as np
import pybullet

from . import scene_v7


class RealBackground:
    def __init__(self, stereo_calib, background_png, base_offset, inpaint_base=True):
        c = stereo_calib
        W, H = c.get("image_size", (1280, 720))
        frame = cv2.imread(background_png)
        if frame is None or frame.shape[1] != 2 * W or frame.shape[0] != H:
            raise ValueError(f"{background_png}: expected a raw {2 * W}x{H} side-by-side frame")
        K1, D1 = np.array(c["left"]["K"]), np.array(c["left"]["dist"])
        K2, D2 = np.array(c["right"]["K"]), np.array(c["right"]["dist"])
        R, T = np.array(c["stereo"]["R"]), np.array(c["stereo"]["T_m"]).reshape(3, 1)
        R1, R2, P1, _, _, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (W, H), R, T, alpha=0)
        self.W, self.H = W, H
        self.f = float(P1[0, 0] * c.get("rectify_zoom", 1.0))
        dy = c.get("rectify_right_dy_px", 0.0) * H / 360.0
        self.img = {}
        for k, (name, K, D, Rr, cy) in enumerate((("left", K1, D1, R1, H / 2), ("right", K2, D2, R2, H / 2 + dy))):
            P = np.array([[self.f, 0, W / 2], [0, self.f, cy], [0, 0, 1.0]])
            mx, my = cv2.initUndistortRectifyMap(K, D, Rr, P, (W, H), cv2.CV_32FC1)
            self.img[name] = cv2.cvtColor(cv2.remap(frame[:, k * W:(k + 1) * W], mx, my, cv2.INTER_LINEAR),
                                          cv2.COLOR_BGR2RGB)
        Tlb = np.array(c["pose_base"]["T_left2base"], float)
        self.base_offset = np.asarray(base_offset, float)
        self.R_wc = Tlb[:3, :3] @ R1.T                       # columns: rectified camera axes (right, down, forward)
        left = Tlb[:3, 3] + self.base_offset
        self.eye = {"left": left, "right": left + self.R_wc[:, 0] * float(np.linalg.norm(T))}
        # Variants of the area around the base, one chosen per episode: untouched (the real base stays under the
        # simulated one; arm remnants of the median may show), remnants above the base painted over, base and
        # remnants painted over. No variant is "right": the model must not rely on that area's details.
        self.variants = [dict(self.img)]
        if inpaint_base:
            self.variants.append(self._paint_over_base(half_w=0.06, back=0.05, front=0.05, z0=0.05, height=0.19))
            self.variants.append(self._paint_over_base())

    # ---------------------------------------------------------------- geometry
    def project(self, name, X):
        """World points -> pixels of the rectified background image of that eye."""
        xc = (np.atleast_2d(X) - self.eye[name]) @ self.R_wc
        return np.c_[self.f * xc[:, 0] / xc[:, 2] + self.W / 2, self.f * xc[:, 1] / xc[:, 2] + self.H / 2]

    def _paint_over_base(self, half_w=0.085, back=0.07, front=0.075, z0=0.0, height=0.19):
        """Inpaint a box around the arm's base (arm remnants stay in a median of few frames). Returns the images."""
        o = self.base_offset
        box = np.array([[x, y, z] for x in (-back, front) for y in (-half_w, half_w) for z in (z0, height)]) + \
            [o[0], o[1], 0.0]
        s = 4                                                # inpaint at quarter resolution: fast, and smooth
        out = {}
        for name in ("left", "right"):
            px = self.project(name, box)
            hull = cv2.convexHull((px / s).astype(np.float32)).astype(np.int32)
            small = cv2.resize(self.img[name], (self.W // s, self.H // s), interpolation=cv2.INTER_AREA)
            mask = np.zeros(small.shape[:2], np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            filled = cv2.inpaint(small, mask, 6, cv2.INPAINT_TELEA)
            big = cv2.resize(filled, (self.W, self.H), interpolation=cv2.INTER_CUBIC)
            m = cv2.GaussianBlur(cv2.resize(mask, (self.W, self.H)).astype(np.float32) / 255, (0, 0), 6)[..., None]
            out[name] = np.clip(self.img[name] * (1 - m) + big * m, 0, 255).astype(np.uint8)
        return out

    def view(self, name, eye, fwd, up, vfov_deg, w, h, variant=-1):
        """The background as a camera at (eye, fwd, up) with that field of view would see it (RGB, h x w)."""
        fwd, up = np.asarray(fwd, float), np.asarray(up, float)
        right = np.cross(fwd, up)
        Rv = np.c_[right, -up, fwd]
        fv = (h / 2) / math.tan(math.radians(vfov_deg) / 2)
        c = self.eye[name] + self.R_wc[:, 2] * 0.5           # four board points in front of the camera
        c = np.array([c[0], c[1], 0.0])
        pts = c + np.array([[0.2, 0.2, 0], [0.2, -0.2, 0], [-0.2, -0.2, 0], [-0.2, 0.2, 0]])
        src = self.project(name, pts).astype(np.float32)
        xc = (pts - np.asarray(eye, float)) @ Rv
        ss = 2 if w < self.W else 1                          # small views: warp at twice the size, average down
        dst = (np.c_[fv * xc[:, 0] / xc[:, 2] + w / 2, fv * xc[:, 1] / xc[:, 2] + h / 2] * ss).astype(np.float32)
        Hm = cv2.getPerspectiveTransform(src, dst)
        big = cv2.warpPerspective(self.variants[variant % len(self.variants)][name], Hm, (w * ss, h * ss), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA) if ss > 1 else big


# ------------------------------------------------------------------------------------------------ look
def sample_disturbances(rng, origin, fwd, left):
    """Incidental things on and around the board that differ from day to day: soft darker / lighter patches
    (stains, shadows of things outside the picture, light spots) and thin strokes (cables, pencil lines, edges of
    paper). Defined in the world on the board plane, so both eyes see them consistently. They are low in
    contrast or thin: nothing here looks like an object to pick."""
    at = lambda: origin + fwd * rng.uniform(-0.08, 0.45) + left * rng.uniform(-0.32, 0.32)
    blobs = [{"p": at().tolist(), "r": float(rng.uniform(0.01, 0.06)), "stretch": float(rng.uniform(1.0, 3.0)),
              "angle": float(rng.uniform(0, math.pi)), "k": float(rng.choice([-1, -1, 1]) * rng.uniform(0.04, 0.22))}
             for _ in range(int(rng.integers(0, 5)))]
    lines = []
    for _ in range(int(rng.integers(0, 3))):
        a = at()
        pts = [a]
        for _ in range(int(rng.integers(1, 4))):
            pts.append(pts[-1] + fwd * rng.uniform(-0.15, 0.15) + left * rng.uniform(-0.15, 0.15))
        lines.append({"pts": [q.tolist() for q in pts], "width": float(rng.uniform(0.0015, 0.006)),
                      "k": float(rng.choice([-1, -1, 1]) * rng.uniform(0.1, 0.7))})
    return {"blobs": blobs, "lines": lines}


def disturb(bg, d, eye, fwd, up, vfov_deg, frame=None, origin=(0, 0)):
    """Draw the episode's disturbances into a background view (multiplicative, soft). frame: (w, h) of the
    whole view if bg is a window of it starting at pixel `origin`."""
    h, w = bg.shape[:2]
    fw, fh = frame or (w, h)
    fwd, up = np.asarray(fwd, float), np.asarray(up, float)
    Rv = np.c_[np.cross(fwd, up), -up, fwd]
    fv = (fh / 2) / math.tan(math.radians(vfov_deg) / 2)

    def px(P):
        xc = (np.atleast_2d(P) - np.asarray(eye, float)) @ Rv
        z = np.maximum(xc[:, 2], 1e-3)
        return np.c_[fv * xc[:, 0] / z + fw / 2 - origin[0], fv * xc[:, 1] / z + fh / 2 - origin[1]], xc[:, 2] > 0.05

    shade = np.zeros((h, w), np.float32)
    for b in d["blobs"]:
        t = np.linspace(0, 2 * math.pi, 24, endpoint=False)
        ca, sa = math.cos(b["angle"]), math.sin(b["angle"])
        ex, ey = b["r"] * b["stretch"] * np.cos(t), b["r"] * np.sin(t)
        ring = np.asarray(b["p"]) + np.c_[ex * ca - ey * sa, ex * sa + ey * ca, np.zeros(24)]
        uv, ok = px(ring)
        if not ok.all():
            continue
        one = np.zeros((h, w), np.float32)
        cv2.fillConvexPoly(one, np.round(uv * 4).astype(np.int32), 1.0, cv2.LINE_AA, 2)
        sigma = max(1.5, 0.35 * float(np.ptp(uv, axis=0).min()))
        shade += b["k"] * cv2.GaussianBlur(one, (0, 0), sigma)
    for ln in d["lines"]:
        uv, ok = px(np.asarray(ln["pts"]))
        if not ok.all():
            continue
        depth = float(np.linalg.norm(np.asarray(ln["pts"][0]) - np.asarray(eye, float)))
        thick = max(1, int(round(fv * ln["width"] / depth)))
        one = np.zeros((h, w), np.float32)
        cv2.polylines(one, [np.round(uv * 4).astype(np.int32)], False, 1.0, thick, cv2.LINE_AA, 2)
        shade += ln["k"] * cv2.GaussianBlur(one, (0, 0), 0.6)
    return np.clip(bg.astype(np.float32) * (1 + np.clip(shade, -0.8, 0.5))[..., None], 0, 255).astype(np.uint8)


def sample_look(rng):
    """Photometric parameters of one episode."""
    return {"white": float(rng.uniform(215, 245)), "black": float(rng.uniform(14, 40)), "variant": int(rng.integers(0, 3)),
            "picture": int(rng.integers(0, 1000)),                       # level of the arm's brightest faces
            "gain": float(rng.uniform(1.1, 1.3) if rng.random() < 0.1 else rng.uniform(0.85, 1.1)),
            "wb": (1 + rng.normal(0, 0.02, 3)).tolist(),
            "blur": float(rng.uniform(0.0, 0.6)), "noise": float(rng.uniform(1.0, 4.0)),
            "arm_shadow": float(rng.uniform(0.25, 0.6))}


def fg_gain(rgb, robot_mask, white):
    """Factor that brings the typical lit face of the rendered arm to the level `white`: the real arm is white
    PLA under diffuse room light and close to the camera's white level nearly everywhere; only its shaded
    sides are grey. (Scaling by the brightest faces left most of the rendered arm mid-grey.)"""
    if robot_mask.sum() < 50:
        return 1.6
    lum = rgb[robot_mask].astype(np.float32).mean(axis=1)
    return float(np.clip(white / max(np.percentile(lum, 65), 1.0), 0.7, 4.0))


def compose(rgb, fg, bg, gain, look, rng, dark=None):
    """Rendered foreground (mask fg) over the background, soft edges, then the episode's camera look.
    dark: mask of foreground pixels painted black (the fixed jaw's tip), drawn at the episode's black level
    with the rendered shading kept."""
    m = fg.astype(np.float32)
    a = cv2.GaussianBlur(m, (0, 0), 0.7)
    col = cv2.GaussianBlur(rgb.astype(np.float32) * m[..., None], (0, 0), 0.7) / np.maximum(a, 1e-3)[..., None]
    col = np.clip(col * gain, 0, 255)
    if dark is not None and dark.any():
        lum = col.mean(axis=2, keepdims=True) / 255.0
        black = look["black"] * (0.55 + 0.9 * lum)                       # matte black paint, some shading
        k = cv2.GaussianBlur(dark.astype(np.float32), (0, 0), 0.6)[..., None]
        col = col * (1 - k) + black * k
    a = np.where(m > 0, np.maximum(a, 0.5), a)[..., None]                # the object's own pixels stay opaque-ish
    a = np.clip(a * 1.5, 0, 1)
    x = bg.astype(np.float32) * (1 - a) + col * a
    x = x * look["gain"] * np.asarray(look["wb"], np.float32)
    if look["blur"] > 0.05:
        x = cv2.GaussianBlur(x, (0, 0), look["blur"])
    x = x + rng.normal(0, look["noise"], x.shape).astype(np.float32)
    return np.clip(x, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------------ rendering
def object_shapes(env):
    """World-space surface points of every object (printed shapes: their vertices; balls: sphere samples)."""
    shapes = []
    k = np.arange(48) + 0.5
    phi, th = np.arccos(1 - 2 * k / 48), math.pi * (1 + 5 ** 0.5) * k
    sphere = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    for o in env.objects:
        pos, orn = env.pb.getBasePositionAndOrientation(o["id"])
        R = np.array(env.pb.getMatrixFromQuaternion(orn)).reshape(3, 3)
        v = o.get("spec", {}).get("vertices")
        local = np.asarray(v, float) if v is not None else sphere * o["size"] / 2
        shapes.append(np.array(pos) + local @ R.T)
    return shapes


def arm_shapes(env):
    """Bounding boxes of the arm's moving links (for a soft, approximate shadow on the board)."""
    shapes = []
    for link in range(env.pb.getNumJoints(env.robot)):
        lo, hi = (np.array(v) for v in env.pb.getAABB(env.robot, link))
        if np.all(hi - lo < 1e-4) or lo[2] > 0.30:
            continue
        shapes.append(np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]))
    return shapes


def _raw(env, eye, fwd, up, vfov_deg):
    """Render the arm and the objects: the whole frame, or only env.crop of it (an off-axis frustum, so that
    the window is rendered directly at its own size)."""
    c, pb = env.cfg, env.pb
    fw, fh = c.front_size
    view = pb.computeViewMatrix(eye.tolist(), (eye + fwd).tolist(), up.tolist())
    if getattr(env, "crop", None):
        x0, y0, x1, y1 = env.crop
        w, h = x1 - x0, y1 - y0
        n = c.stereo_near_m
        f = (fh / 2) / math.tan(math.radians(vfov_deg) / 2)
        proj = pb.computeProjectionMatrix((x0 - fw / 2) / f * n, (x1 - fw / 2) / f * n,
                                          (fh / 2 - y1) / f * n, (fh / 2 - y0) / f * n, n, c.stereo_far_m)
    else:
        w, h = fw, fh
        proj = pb.computeProjectionMatrixFOV(vfov_deg, w / h, c.stereo_near_m, c.stereo_far_m)
    renderer = pybullet.ER_TINY_RENDERER if c.renderer == "tiny" and not env.gui else pybullet.ER_BULLET_HARDWARE_OPENGL
    img = pb.getCameraImage(w, h, view, proj, renderer=renderer, shadow=0,
                            flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, **env.light)
    rgb = np.reshape(np.asarray(img[2], dtype=np.uint8), (h, w, 4))[:, :, :3].copy()
    seg = np.reshape(np.asarray(img[4]), (h, w)).astype(np.int64)
    # With ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX the mask value is the body id + ((link index + 1) << 24);
    # without the flag it is the body id alone. Split it into both (link -1 = base link or no link information).
    body = np.where(seg >= 0, seg & ((1 << 24) - 1), -1)
    link = np.where(seg >= 0, (seg >> 24) - 1, -2)                     # -1 = base link / unknown, -2 = nothing
    depth = np.reshape(np.asarray(img[3], dtype=np.float32), (h, w))
    return rgb, body, view, proj, link, depth


class JawTips:
    """Where the two jaws end, measured once from the arm's collision geometry (the kinematics' own copy of the
    arm): a grid of points around the fingers, gripper closed, is tested for which link occupies it.

    For each jaw (fixed = gripper_link, moving = the jaw link), in the gripper_link frame of the CLOSED gripper:
      d_tip      how far the jaw reaches along the fingers (approach axis)
      face(d)    position of its gripping face across the fingers (closing axis), as a function of d
    The moving jaw's pixels are brought into that frame through its own link frame, so the result holds for
    any opening."""
    _cache = {}

    def __init__(self, kin, res=0.0015):
        key = (id(kin), res)
        if key in JawTips._cache:
            self.__dict__.update(JawTips._cache[key])
            return
        pb = kin.pb
        q = np.zeros(6)
        q[5] = kin.gripper_closed
        kin.set_q(q)
        g_pos, g_rot = kin.frame(kin.g_link)
        j_pos, j_rot = kin.frame(kin.jaw_link)
        # jaw_link frame -> gripper_link frame of the closed gripper (rigid; holds for the jaw at any opening)
        self.jaw_R = g_rot.T @ j_rot
        self.jaw_t = g_rot.T @ (j_pos - g_pos)
        a, c, h = kin.approach_local, kin.closing_local, kin.hinge_local
        self.a, self.c, self.h, self.o = a, c, h, kin.tcp_local
        self.sign = float(getattr(kin, "grasp_sign", 1.0))                # fixed -> moving along c
        probe = pb.createMultiBody(0, pb.createCollisionShape(pybullet.GEOM_SPHERE, radius=res / 2), -1, [0, 0, 5])
        ds = np.arange(-0.03, 0.10, res)
        us = np.arange(-0.045, 0.045 + 1e-9, res)
        vs = np.arange(-0.025, 0.025 + 1e-9, 2 * res)
        occ = {kin.g_link: [], kin.jaw_link: []}
        for d in ds:
            for u in us:
                for v in vs:
                    loc = self.o + a * d + c * u + h * v
                    pb.resetBasePositionAndOrientation(probe, (g_pos + g_rot @ loc).tolist(), [0, 0, 0, 1])
                    cps = pb.getClosestPoints(probe, kin.body, 0.0)
                    if cps and cps[0][4] in occ:
                        occ[cps[0][4]].append((d, u * self.sign))
        pb.removeBody(probe)
        kin.set_q(np.zeros(6))
        self.tips = {}
        for name, link, inner in (("fixed", kin.g_link, np.max), ("moving", kin.jaw_link, np.min)):
            pts = np.array(occ[link])
            if len(pts) == 0:
                raise RuntimeError(f"the {name} jaw was not found by the probe")
            d_tip = float(pts[:, 0].max())
            # gripping face: per depth bin, the jaw's extreme towards the other jaw (fixed: max, moving: min)
            bins = np.unique(np.round(pts[:, 0] / res).astype(int))
            fd = np.array([b * res for b in bins])
            fu = np.array([inner(pts[np.round(pts[:, 0] / res).astype(int) == b, 1]) for b in bins])
            self.tips[name] = {"d_tip": d_tip, "face_d": fd, "face_u": fu}
        JawTips._cache[key] = dict(self.__dict__)


def painted_tip_mask(env, robot_px, link, depth, view, proj):
    """Pixels of the painted jaw tips: on BOTH jaws the last env.tip_paint_m before the tip, every surface but
    the gripping face (as painted on the real arm). Arm pixels are back-projected with the depth buffer, brought
    into the gripper frame of the closed gripper (moving-jaw pixels through the jaw's own frame) and compared
    with the jaws' measured tips and gripping faces (JawTips)."""
    L = float(getattr(env, "tip_paint_m", 0.0) or 0.0)
    if L <= 0 or not robot_px.any():
        return None
    if getattr(env, "_jaw_tips", None) is None:
        env._jaw_tips = JawTips(env.kin)
    J = env._jaw_tips
    h, w = depth.shape
    mask = np.zeros_like(robot_px)
    V, P = np.array(view).reshape(4, 4).T, np.array(proj).reshape(4, 4).T
    inv = np.linalg.inv(P @ V)
    for name, lk in (("fixed", env.g_link), ("moving", env.jaw_link)):
        px = robot_px & (link == lk)
        if not px.any():
            continue
        v, u = np.nonzero(px)
        ndc = np.stack([(u + 0.5) / w * 2 - 1, 1 - (v + 0.5) / h * 2, depth[v, u] * 2 - 1, np.ones(len(u))])
        X = inv @ ndc
        X = (X[:3] / X[3]).T                                             # world points
        s = env.pb.getLinkState(env.robot, lk, computeForwardKinematics=True)
        R = np.array(env.pb.getMatrixFromQuaternion(s[5])).reshape(3, 3)
        loc = (X - np.array(s[4])) @ R                                   # in the link's own frame
        if name == "moving":
            loc = loc @ J.jaw_R.T + J.jaw_t                              # -> gripper frame, gripper closed
        rel = loc - J.o
        d = rel @ J.a
        uu = (rel @ J.c) * J.sign                                        # + = towards the moving jaw
        t = J.tips[name]
        face = np.interp(d, t["face_d"], t["face_u"])
        gripping = (uu > face - 0.0012) if name == "fixed" else (uu < face + 0.0012)
        paint = (d >= t["d_tip"] - L) & ~gripping
        mask[v[paint], u[paint]] = True
    return mask


def render_stereo(env):
    """Left and right image: the real background seen from the simulated camera, arm and objects on top."""
    cam = env.stereo_cam
    if getattr(env, "_look_for", None) is not env.cam_pipe:              # one look per episode
        env._look, env._look_for = sample_look(env.rng), env.cam_pipe
        env._look["disturb"] = sample_disturbances(env.rng, np.asarray(env.pan_axis, float),
                                                   np.asarray(env.fwd, float), np.asarray(env.left, float))
    look = env._look
    bgm = env.real_bg[look["picture"] % len(env.real_bg)]               # one of the setup's background pictures
    center, fwd, right, up = (np.array(cam[k]) for k in ("center", "fwd", "right", "up"))
    w, h = env.cfg.front_size
    crop = getattr(env, "crop", None)
    obj_ids = [o["id"] for o in env.objects]
    out, gain = {}, None
    for name, s in (("left", -cam["baseline"] / 2), ("right", cam["baseline"] / 2)):
        eye = center + right * s
        rgb, seg, view, proj, link, depth = _raw(env, eye, fwd, up, cam["vfov_deg"])
        robot = seg == env.robot
        fg = robot | np.isin(seg, obj_ids)
        dark = painted_tip_mask(env, robot, link, depth, view, proj)
        env.tip_px = int(dark.sum()) if dark is not None else 0         # for the generators' example files
        if gain is None:
            gain = fg_gain(rgb, robot, look["white"])
        bg = bgm.view(name, eye, fwd, up, cam["vfov_deg"], w, h, look["variant"])
        if crop:
            bg = np.ascontiguousarray(bg[crop[1]:crop[3], crop[0]:crop[2]])
        bg = disturb(bg, look["disturb"], eye, fwd, up, cam["vfov_deg"], (w, h), crop[:2] if crop else (0, 0))
        bg = scene_v7.add_object_shadows(bg, ~fg, object_shapes(env), env.light, view, proj, env.rng)
        arm = scene_v7.add_object_shadows(bg, ~fg, arm_shapes(env), env.light, view, proj, env.rng)
        k = look["arm_shadow"]
        bg = np.clip(bg.astype(np.float32) * (1 - k) + arm.astype(np.float32) * k, 0, 255).astype(np.uint8)
        out[name] = compose(rgb, fg, bg, gain, look, env.rng, dark)
    return out


def load(setup_dir, base_offset):
    """The setup's background pictures as RealBackground objects (stereo_background.png and any further
    stereo_background_*.png taken with the SAME camera position, e.g. on another day or under other light; one
    is drawn per episode), or None if the setup has none."""
    import glob
    cal = os.path.join(setup_dir, "stereo_calib.json")
    files = sorted(glob.glob(os.path.join(setup_dir, "stereo_background.png")) +
                   glob.glob(os.path.join(setup_dir, "stereo_background_*.png")))
    if not files or not os.path.exists(cal):
        return None
    calib = json.load(open(cal))
    return [RealBackground(calib, f, base_offset) for f in files]
