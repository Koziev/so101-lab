"""
The exp11 cell in the v9 simulator: the printed objects (objects_v11) placed by a task (tasks.py), so that the
scene's instruction refers to exactly one object; the stereo camera centred on the calibrated pose of the new
cell (2.8 mm lenses, 60.9 deg rectified view). Appearance (board, backdrop textures, light) is still the v9 one.
"""
import math

import numpy as np
import pybullet

from . import objects_v11
from .env import SO101GraspEnv


class CellEnvV11(SO101GraspEnv):
    task_kwargs = {}                    # passed to tasks.sample_task (weights, n_objects, white_weight, ...)
    crop = None                         # (x0, y0, x1, y1) window of the rendered frame given to the model
    tip_paint_m = 0.0                   # the fixed jaw's last N m painted black (the real arm's marker)
    real_bg = None                      # scene_real.RealBackground: the real cell's picture as the scene
    grasp_z = 0.020                     # height of the grasp point above the board, the same for every object:
                                        # the fingertips (12 mm beyond the grasp point) stay ~8 mm above the board

    def _spawn_objects(self):
        import tasks
        task = tasks.sample_task(self.rng, **self.task_kwargs)
        placed, objs = [], []
        for spec in task["specs"]:
            xy = self._sample_xy(placed)
            if xy is None:
                break
            placed.append(xy)
            yaw = float(self.rng.uniform(-math.pi, math.pi))
            body = objects_v11.create_body(pybullet, spec, xy, yaw, self.pb._client)
            objs.append({"id": body, "descr": spec["label"], "shape": spec["shape"], "color": spec["colour"],
                         "size": float(2 * spec["half_height"]), "yaw": yaw, "spec": spec})
        if not objs or not tasks.valid(task, [o["spec"] for o in objs]):
            for o in objs:                         # target not placeable or the prompt became ambiguous:
                self.pb.removeBody(o["id"])        # leave no objects -> reset() tries a new camera and scene
            self.objects, self.task = [], None
            return
        self.objects, self.task = objs, task

    def observe(self, render=True):
        if not (render and self.real_bg is not None):
            return super().observe(render)
        from . import scene_real
        obs = super().observe(False)
        obs["images"] = scene_real.render_stereo(self)
        return obs

    def in_image(self, point, margin_px=None):
        """Is a world point inside the image the model gets (both eyes; the crop window if there is one)?"""
        cam = self.stereo_cam
        w, h = self.cfg.front_size
        x0, y0, x1, y1 = self.crop or (0, 0, w, h)
        if margin_px is None:
            margin_px = 0.025 * w                    # about the width of an object: none is cut by the edge
        center, fwd, right, up = (np.array(cam[k]) for k in ("center", "fwd", "right", "up"))
        f = (h / 2) / math.tan(math.radians(cam["vfov_deg"]) / 2)
        for s in (-cam["baseline"] / 2, cam["baseline"] / 2):
            d = np.asarray(point, float) - (center + right * s)
            z = float(d @ fwd)
            if z <= 0.05:
                return False
            u, v = f * float(d @ right) / z + w / 2, -f * float(d @ up) / z + h / 2
            if not (x0 + margin_px <= u <= x1 - margin_px and y0 + margin_px <= v <= y1 - margin_px):
                return False
        return True

    def target_point(self, obj):
        """Where the grasp point should go: above the object's footprint centre, at grasp_z above the board
        (defined by the clearance of the fingertips, not by the object's height: a label at mid-height of a
        small object would put the fingertips on or below the board)."""
        pos = np.asarray(self.object_pose(obj["id"])[0], float)
        return np.array([pos[0], pos[1], self.grasp_z])


def v9_center_from_pose_base(env, T_left2base, baseline):
    """v9 camera parameters (distance, height, pitch, azimuth, yaw, roll relative to the pan axis and the
    robot's forward direction) of the stereo centre, from the calibrated left-eye pose in the URDF base frame
    (stereo_calib.json pose_base). Inverse of scene_v9.sample_camera_v9's construction."""
    T = np.asarray(T_left2base, float)
    R = T[:3, :3]
    right, fwd, up = R[:, 0], R[:, 2], -R[:, 1]                    # OpenCV camera axes
    left_eye = T[:3, 3] + np.asarray(env.kin.base_offset, float)
    cen = left_eye + right * baseline / 2 - env.pan_axis
    d_f, d_l = float(cen @ env.fwd), float(cen @ env.left)
    dist, az = math.hypot(d_f, d_l), math.atan2(d_l, d_f)
    h0 = -(math.cos(az) * env.fwd + math.sin(az) * env.left)
    fh = fwd[:2] / np.linalg.norm(fwd[:2])
    yaw = math.atan2(h0[0] * fh[1] - h0[1] * fh[0], float(h0[:2] @ fh))
    pitch = math.asin(-fwd[2])
    r0 = np.cross(fwd, [0, 0, 1.0])
    r0 /= np.linalg.norm(r0)
    u0 = np.cross(r0, fwd)
    roll = math.atan2(-float(up @ r0), float(up @ u0))
    return {"distance_m": dist, "height_m": float(cen[2]), "pitch_deg": math.degrees(pitch),
            "azimuth_deg": math.degrees(az), "yaw_deg": math.degrees(yaw), "roll_deg": math.degrees(roll)}


def rectified_left_pose(stereo_calib):
    """Pose of the RECTIFIED left eye in the URDF base frame (4x4, as nested lists). pose_base in the setup is
    the pose of the raw left camera; the images the model gets are rectified, which turns the view by the
    rectification's rotation (a fraction of a degree, a few pixels). The simulated camera uses this pose, so
    that a simulated image and a real one coincide."""
    import cv2
    c = stereo_calib
    size = tuple(c.get("image_size", (1280, 720)))
    R1 = cv2.stereoRectify(np.array(c["left"]["K"]), np.array(c["left"]["dist"]), np.array(c["right"]["K"]),
                           np.array(c["right"]["dist"]), size, np.array(c["stereo"]["R"]),
                           np.array(c["stereo"]["T_m"]).reshape(3, 1), alpha=0)[0]
    T = np.array(c["pose_base"]["T_left2base"], float)
    T[:3, :3] = T[:3, :3] @ R1.T
    return T.tolist()


def rectified_hfov_deg(stereo_calib, width=1280, height=720):
    """Horizontal field of view of the rectified, zoomed view the models see (as real_check / record_real)."""
    import cv2
    c = stereo_calib
    K1, D1 = np.array(c["left"]["K"]), np.array(c["left"]["dist"])
    K2, D2 = np.array(c["right"]["K"]), np.array(c["right"]["dist"])
    R, T = np.array(c["stereo"]["R"]), np.array(c["stereo"]["T_m"]).reshape(3, 1)
    _, _, P1, *_ = cv2.stereoRectify(K1, D1, K2, D2, tuple(c.get("image_size", (width, height))), R, T, alpha=0)
    f = P1[0, 0] * c.get("rectify_zoom", 1.0)
    return 2 * math.degrees(math.atan(c.get("image_size", (width, height))[0] / 2 / f))
