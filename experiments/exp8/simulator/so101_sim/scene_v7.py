"""
v7 scene: the real setup's geometry and look, seen by a stereo camera.

* Plywood board 50 x 49 cm (pan axis 7 cm from its back edge) on a larger desk.
* Textures from folders (texture_dir/board, /desk, /backdrop: your photos plus any CC0
  textures you add), split into train / eval sets by file so evaluation can use unseen
  textures. Procedural textures are the fallback when a folder is empty.
* A backdrop panel behind the robot and random dark clutter on the desk beyond the board.
* Mostly near-white arm (like the real one), other colours for robustness.
* Stereo camera modelled as the rectified wide-mode pair of the real camera (about
  44-46 x 26 deg per eye, 6.2 cm baseline, parallel eyes). Position is randomised in front
  of the robot and the camera is aimed so the whole object area plus the approach height is
  visible in BOTH eyes; if it does not fit, the camera is moved back.
* Camera pipeline noise fitted to the real shots: white balance, exposure, gamma, limited
  contrast, vignetting, blur, signal-dependent sensor noise, MJPEG compression.
"""
import glob
import math
import os
import tempfile

import cv2
import numpy as np
import pybullet

from .kinematics import unit

TEXTURE_KINDS = ("board", "desk", "backdrop")
NEAR_WHITE = [(0.93, 0.93, 0.91), (0.96, 0.96, 0.95), (0.86, 0.86, 0.85), (0.80, 0.81, 0.82)]
CLUTTER_COLORS = [(0.05, 0.05, 0.05), (0.12, 0.12, 0.13), (0.25, 0.25, 0.27), (0.85, 0.85, 0.83),
                  (0.6, 0.55, 0.45), (0.15, 0.2, 0.35)]


# ----------------------------------------------------------------------------- textures
def split_files(files, split, every=5):
    """Every `every`-th file (sorted) is held out for evaluation; tiny folders are shared."""
    files = sorted(files)
    if len(files) < every:
        return files
    held = [f for i, f in enumerate(files) if i % every == every - 1]
    return held if split == "eval" else [f for f in files if f not in held] if split == "train" else files


def load_texture_pool(pb, cfg, fallback_fn):
    pools, info = {}, {}
    rng = np.random.default_rng(7)
    for kind in TEXTURE_KINDS:
        files = []
        for ext in ("jpg", "jpeg", "png"):
            files += glob.glob(os.path.join(cfg.texture_dir, kind, f"*.{ext}"))
        files = split_files(files, cfg.texture_split)
        if len(files) > cfg.max_textures_per_kind:
            files = sorted(rng.choice(files, cfg.max_textures_per_kind, replace=False))
        ids = []
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            s = 1024 / max(img.shape[:2])  # cap texture size (memory, render speed)
            if s < 1:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            cv2.imwrite(tmp.name, img)
            ids.append(pb.loadTexture(tmp.name))
            os.unlink(tmp.name)
        if not ids:  # fallback: procedural textures
            for _ in range(8):
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.close()
                cv2.imwrite(tmp.name, fallback_fn(rng))
                ids.append(pb.loadTexture(tmp.name))
                os.unlink(tmp.name)
        pools[kind] = ids
        info[kind] = len(files) if files else "procedural"
    return pools, info


# ----------------------------------------------------------------------------- static scene
def build_static(env):
    c, pb = env.cfg, env.pb
    yaw_q = pb.getQuaternionFromEuler([0, 0, env.kin.forward_yaw])
    bl, bw, bt = c.board_size[0], c.board_size[1], c.board_thickness
    center = env.pan_axis + env.fwd * (bl / 2 - c.board_back_margin) + np.array([0, 0, -bt / 2])
    half = [bl / 2, bw / 2, bt / 2]
    env.table = pb.createMultiBody(0, pb.createCollisionShape(pybullet.GEOM_BOX, halfExtents=half),
                                   pb.createVisualShape(pybullet.GEOM_BOX, halfExtents=half), center.tolist(), yaw_q)
    pb.changeDynamics(env.table, -1, lateralFriction=0.8)
    dh = [0.75, 0.85, 0.015]
    env.desk = pb.createMultiBody(0, pb.createCollisionShape(pybullet.GEOM_BOX, halfExtents=dh),
                                  pb.createVisualShape(pybullet.GEOM_BOX, halfExtents=dh),
                                  (env.pan_axis + env.fwd * 0.15 + np.array([0, 0, -bt - dh[2]])).tolist(), yaw_q)
    env.wall = pb.createMultiBody(0, -1, pb.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.01, 0.9, 0.55]),
                                  [0, 0, -5], yaw_q)  # placed per episode
    env.clutter = []


# ----------------------------------------------------------------------------- per episode
def randomize(env):
    c, pb, r = env.cfg, env.pb, env.rng
    yaw = env.kin.forward_yaw
    # surfaces
    for body, kind, tint_lo, jitter in ((env.table, "board", 0.88, 0.02), (env.desk, "desk", 0.8, 0.03),
                                        (env.wall, "backdrop", 0.6, 0.05)):
        tint = np.clip(r.uniform(tint_lo, 1.0) * (1 + r.normal(0, jitter, 3)), 0, 1).tolist() + [1.0]
        pb.changeVisualShape(body, -1, textureUniqueId=int(r.choice(env.tex_pools[kind])), rgbaColor=tint)
    # desk slides sideways a little (the real desk extends further to one side)
    bt = c.board_thickness
    desk_pos = env.pan_axis + env.fwd * r.uniform(0.05, 0.25) + env.left * r.uniform(-0.25, 0.25) \
        + np.array([0, 0, -bt - 0.015])
    pb.resetBasePositionAndOrientation(env.desk, desk_pos.tolist(), pb.getQuaternionFromEuler([0, 0, yaw]))
    # backdrop panel behind the robot
    wall_pos = env.pan_axis - env.fwd * r.uniform(0.35, 0.75) + env.left * r.uniform(-0.2, 0.2) \
        + np.array([0, 0, -bt + 0.55 - r.uniform(0.0, 0.15)])
    pb.resetBasePositionAndOrientation(env.wall, wall_pos.tolist(),
                                       pb.getQuaternionFromEuler([0, 0, yaw + math.radians(r.uniform(-8, 8))]))
    # clutter beyond the board (visual only)
    for b in env.clutter:
        pb.removeBody(b)
    env.clutter = []
    for _ in range(int(r.integers(c.clutter_range[0], c.clutter_range[1] + 1))):
        behind = r.random() < 0.6
        if behind:
            f, l = r.uniform(-0.45, -0.12), r.uniform(-0.6, 0.6)
        else:
            f, l = r.uniform(-0.1, 0.35), r.choice([-1, 1]) * r.uniform(c.board_size[1] / 2 + 0.05, 0.7)
        col = list(CLUTTER_COLORS[r.integers(len(CLUTTER_COLORS))]) + [1.0]
        if r.random() < 0.7:
            he = [r.uniform(0.02, 0.09), r.uniform(0.02, 0.09), r.uniform(0.015, 0.08)]
            vis = pb.createVisualShape(pybullet.GEOM_BOX, halfExtents=he, rgbaColor=col)
            z = he[2]
        else:
            rad, hgt = r.uniform(0.015, 0.045), r.uniform(0.04, 0.15)
            vis = pb.createVisualShape(pybullet.GEOM_CYLINDER, radius=rad, length=hgt, rgbaColor=col)
            z = hgt / 2
        pos = env.pan_axis + env.fwd * f + env.left * l + np.array([0, 0, -bt + z])
        env.clutter.append(pb.createMultiBody(0, -1, vis, pos.tolist(),
                                              pb.getQuaternionFromEuler([0, 0, r.uniform(-math.pi, math.pi)])))
    # arm colour: mostly near-white like the real arm
    if r.random() < c.p_white_arm:
        base = np.array(NEAR_WHITE[r.integers(len(NEAR_WHITE))])
    else:
        from .env import PRINT_COLORS
        base = np.array(PRINT_COLORS[r.integers(len(PRINT_COLORS))])
    color = np.clip(base + r.normal(0, 0.02, 3), 0, 1).tolist() + [1.0]
    for link, k in env._printed_shapes:
        try:
            pb.changeVisualShape(env.robot, link, shapeIndex=k, rgbaColor=color)
        except TypeError:
            break
    # camera pipeline parameters for this episode
    env.cam_pipe = {
        "wb": r.normal(1.0, 0.04, 3).tolist(), "gain": float(r.uniform(0.85, 1.12)),
        "gamma": float(r.uniform(0.9, 1.15)), "lo": float(r.uniform(0.0, 0.05)), "hi": float(r.uniform(0.7, 0.95)),
        "vignette": float(r.uniform(0.0, 0.25)), "blur": float(r.uniform(0.0, 0.9)),
        "shot": float(r.uniform(0.05, 0.3)), "read": float(r.uniform(1.0, 3.0)), "jpeg": int(r.integers(60, 91)),
    }


# ----------------------------------------------------------------------------- stereo camera
def workspace_points(env, z_max):
    c = env.cfg
    pts = []
    for rad in np.linspace(c.r_range[0], c.r_range[1], 5):
        for b in np.radians(np.linspace(-c.half_angle_deg, c.half_angle_deg, 9)):
            for z in (0.0, z_max):
                pts.append(env.pan_axis + rad * (math.cos(b) * env.fwd + math.sin(b) * env.left) + np.array([0, 0, z]))
    return np.array(pts)


def _fits(pts, center, fwd, right, up, baseline, hfov, vfov, margin):
    for s in (-baseline / 2, baseline / 2):
        rel = pts - (center + right * s)
        depth = rel @ fwd
        if np.any(depth <= 0.05):
            return False
        if (np.max(np.abs(rel @ right) / depth) > margin * math.tan(hfov / 2)
                or np.max(np.abs(rel @ up) / depth) > margin * math.tan(vfov / 2)):
            return False
    return True


def sample_stereo_camera(env):
    c, r = env.cfg, env.rng
    w, h = c.front_size
    hfov = math.radians(r.uniform(*c.stereo_hfov_deg))
    vfov = 2 * math.atan(math.tan(hfov / 2) * h / w)
    baseline = r.uniform(*c.stereo_baseline_m)
    dist, height = r.uniform(*c.stereo_distance_m), r.uniform(*c.stereo_height_m)
    lateral, roll = r.uniform(-0.03, 0.03), math.radians(r.uniform(-1.5, 1.5))
    pts = workspace_points(env, c.stereo_view_height)
    target = pts.mean(axis=0) * [1, 1, 0] + np.array([0, 0, c.stereo_view_height / 3])
    for _ in range(40):
        center = env.pan_axis + env.fwd * dist + env.left * lateral + np.array([0, 0, height])
        fwd = unit(target - center)
        right = unit(np.cross(fwd, [0, 0, 1]))
        up = np.cross(right, fwd)
        right, up = right * math.cos(roll) + up * math.sin(roll), -right * math.sin(roll) + up * math.cos(roll)
        if _fits(pts, center, fwd, right, up, baseline, hfov, vfov, c.stereo_margin):
            break
        dist += 0.02
        height += 0.01
    env.stereo_cam = {"center": center.tolist(), "fwd": fwd.tolist(), "right": right.tolist(), "up": up.tolist(),
                      "baseline": float(baseline), "hfov_deg": math.degrees(hfov), "vfov_deg": math.degrees(vfov),
                      "distance_m": float(dist), "height_m": float(height)}


def render_stereo(env):
    cam = env.stereo_cam
    center, fwd, right, up = (np.array(cam[k]) for k in ("center", "fwd", "right", "up"))
    out = {}
    for name, s in (("left", -cam["baseline"] / 2), ("right", cam["baseline"] / 2)):
        eye = center + right * s
        out[name] = render_eye(env, eye, fwd, up, cam["vfov_deg"])
    return out


def render_eye(env, eye, fwd, up, vfov_deg):
    c, pb = env.cfg, env.pb
    w, h = c.front_size
    view = pb.computeViewMatrix(eye.tolist(), (eye + fwd).tolist(), up.tolist())
    proj = pb.computeProjectionMatrixFOV(vfov_deg, w / h, c.stereo_near_m, c.stereo_far_m)
    renderer = pybullet.ER_TINY_RENDERER if c.renderer == "tiny" and not env.gui else pybullet.ER_BULLET_HARDWARE_OPENGL
    img = pb.getCameraImage(w, h, view, proj, renderer=renderer, shadow=1, **env.light)
    rgb = np.reshape(np.asarray(img[2], dtype=np.uint8), (h, w, 4))[:, :, :3].copy()
    if c.v7_contact_shadows:
        seg = np.reshape(np.asarray(img[4]), (h, w))
        receivers = np.isin(seg, [env.table, env.desk])
        rgb = add_object_shadows(rgb, receivers, object_shapes(env), env.light, view, proj, env.rng)
    return rgb


# ----------------------------------------------------------------------------- object shadows
def object_shapes(env):
    """World-space surface points of every object (cube corners, sphere samples)."""
    shapes = []
    k = np.arange(48) + 0.5
    phi, th = np.arccos(1 - 2 * k / 48), math.pi * (1 + 5 ** 0.5) * k
    sphere = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    for o in env.objects:
        pos, orn = env.pb.getBasePositionAndOrientation(o["id"])
        R = np.array(env.pb.getMatrixFromQuaternion(orn)).reshape(3, 3)
        local = cube * o["size"] / 2 if o["shape"] == "cube" else sphere * o["size"] / 2
        shapes.append(np.array(pos) + local @ R.T)
    return shapes


def _to_pixels(pts, view, proj, w, h):
    V, P = np.array(view).reshape(4, 4).T, np.array(proj).reshape(4, 4).T
    c = (P @ V @ np.c_[pts, np.ones(len(pts))].T).T
    ok = c[:, 3] > 1e-6
    ndc = c[:, :3] / c[:, 3:4]
    return np.stack([(ndc[:, 0] + 1) / 2 * w, (1 - ndc[:, 1]) / 2 * h], 1), ok


def add_object_shadows(rgb, receivers, shapes, light, view, proj, rng, board_z=0.0):
    """Soft shadows of the objects on the board/desk, computed geometrically.
    The CPU renderer's own object shadows are sparse dot patterns that the camera pipeline
    (blur, noise, contrast) washes out; the arm's large shadows are left to the renderer."""
    L = np.asarray(light["lightDirection"], float)
    L /= np.linalg.norm(L)
    if L[2] < 0.1:
        return rgb
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.float32)
    for pts in shapes:
        ground = pts - L[None, :] * ((pts[:, 2] - board_z) / L[2])[:, None]   # cast along the light onto the board
        allp = np.vstack([ground, pts * [1, 1, 0] + [0, 0, board_z]])
        hull = cv2.convexHull(allp[:, :2].astype(np.float32)).reshape(-1, 2)
        poly = np.c_[hull, np.full(len(hull), board_z + 5e-4)]
        px, ok = _to_pixels(poly, view, proj, w, h)
        if not ok.all():
            continue
        one = np.zeros((h, w), np.float32)
        cv2.fillConvexPoly(one, np.round(px * 4).astype(np.int32), 1.0, cv2.LINE_AA, 2)
        size_px = np.ptp(_to_pixels(pts, view, proj, w, h)[0], axis=0).max()
        sigma = max(1.0, 0.12 * size_px)                                # soft penumbra like the real shots
        mask = np.maximum(mask, cv2.GaussianBlur(one, (0, 0), sigma))
    sin_el = L[2]
    direct = light["lightDiffuseCoeff"] * sin_el
    strength = direct / (light["lightAmbientCoeff"] + direct) * rng.uniform(0.7, 0.9)   # some bounce light
    shade = 1.0 - strength * mask * receivers
    return np.clip(rgb.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- camera pipeline
def camera_pipeline(rgb, p, rng):
    """Turn a clean render into something like the real USB camera's output."""
    x = rgb.astype(np.float32) / 255.0
    x = x * np.array(p["wb"], np.float32) * p["gain"]
    if p["vignette"] > 0:
        h, w = x.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        rr = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        x *= (1 - p["vignette"] * rr / 2)[..., None]
    x = np.clip(x, 0, 1) ** p["gamma"]
    x = p["lo"] + (p["hi"] - p["lo"]) * x                        # limited contrast like the real shots
    if p["blur"] > 0.05:
        x = cv2.GaussianBlur(x, (0, 0), p["blur"])
    v = x * 255.0
    v += rng.normal(0, 1, v.shape).astype(np.float32) * np.sqrt(p["shot"] * v + p["read"] ** 2)
    v = np.clip(v, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", v[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, p["jpeg"]])  # MJPEG transport
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1].copy() if ok else v
