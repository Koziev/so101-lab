"""
PyBullet tabletop environment for the SO-101.

The environment knows nothing about who chooses actions: a scripted expert (data
generation) and a fine-tuned VLA (evaluation) drive it through the same interface:

    obs, info = env.reset(seed)
    obs = env.step(action)    # action: 6 values in LeRobot units

LeRobot units: 5 arm joints in degrees (URDF angles, 0 = mid-range) + gripper 0..100
(0 = closed, 100 = open).
"""
import importlib.util
import itertools
import math
import os
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np
import pybullet
from pybullet_utils import bullet_client

import collections

from . import scene_v7, scene_v8, scene_v9
from .kinematics import ALL_JOINTS, GRIPPER_LINK, JAW_LINK, TCP_LINK, SO101Kinematics, name_maps, unit

OBJECT_COLORS = [
    ("red", (0.9, 0.05, 0.05)), ("green", (0.05, 0.6, 0.1)), ("blue", (0.05, 0.15, 0.9)),
    ("yellow", (0.95, 0.9, 0.05)), ("orange", (1.0, 0.5, 0.0)), ("purple", (0.5, 0.1, 0.6)),
    ("pink", (1.0, 0.55, 0.7)), ("cyan", (0.0, 0.85, 0.9)), ("white", (0.95, 0.95, 0.95)),
    ("black", (0.05, 0.05, 0.05)), ("brown", (0.45, 0.25, 0.1)), ("gray", (0.5, 0.5, 0.5)),
]
SHAPES = ["cube", "ball"]
PRINT_COLORS = [(1.0, 0.82, 0.12), (0.95, 0.95, 0.95), (0.1, 0.1, 0.1), (0.85, 0.1, 0.1),
                (0.1, 0.3, 0.85), (1.0, 0.45, 0.0), (0.2, 0.6, 0.25), (0.55, 0.55, 0.6)]
INSTRUCTIONS = ["pick up the {}", "grasp the {}", "grab the {}", "lift the {}", "take the {}"]


# photo textures shipped with the simulator (simulator/textures)
DEFAULT_TEXTURE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "textures"))


@dataclass
class EnvConfig:
    control_hz: int = 30
    physics_hz: int = 240
    renderer: str = "tiny"            # "tiny" (CPU) or "egl" (GPU, headless); GUI uses OpenGL
    front_size: tuple = (640, 480)
    wrist_size: tuple = (640, 480)
    arm_torque: float = 2.5           # N*m cap; STS3215 stall torque is roughly 2-3 N*m
    gripper_torque: float = 1.0
    r_range: tuple = (0.12, 0.24)     # object distance from the pan axis, m
    half_angle_deg: float = 50.0      # object bearing relative to the arm's forward direction
    n_distractors: tuple = (0, 0)     # inclusive range
    min_separation: float = 0.07
    sphere_radius: tuple = (0.012, 0.02)
    cube_size: tuple = (0.022, 0.035)
    object_friction: float = 1.0
    jaw_friction: float = 1.5
    camera_noise_std: float = 3.0
    wrist_cam: tuple = (-0.075, 0.0, 0.035)  # eye offset from TCP in (approach, closing, hinge) axes
    wrist_fov: float = 70.0
    home_tcp: tuple = (0.15, 0.0, 0.15)      # forward, left, height of TCP at home
    home_gripper_pct: float = 20.0
    n_textures: int = 16
    forward_yaw_deg: float = None     # override the auto-detected forward direction if needed
    camera_preset: str = "v1"         # "v1": original framing; "v4": front aimed at the object area with a
                                      # field of view fitted to it, wrist camera further back/aside
    top_camera: bool = False          # overhead camera looking straight down at the object area
    # ---- v7 scene (scene_preset="v7"): real board/desk geometry, photo textures, stereo camera
    scene_preset: str = "v1"          # "v1": original table scene, "v7": see so101_sim/scene_v7.py
    texture_dir: str = DEFAULT_TEXTURE_DIR  # subfolders board/, desk/, backdrop/ with .jpg/.png files
    texture_split: str = "train"      # "train" | "eval" (held-out textures) | "all"
    max_textures_per_kind: int = 48
    board_size: tuple = (0.50, 0.49)  # along the forward axis, across (m)
    board_thickness: float = 0.012
    board_back_margin: float = 0.07   # board behind the pan axis (m)
    clutter_range: tuple = (2, 6)
    p_white_arm: float = 0.6
    stereo_hfov_deg: tuple = (43.5, 46.5)    # rectified wide mode of the real camera
    stereo_baseline_m: tuple = (0.061, 0.063)
    stereo_distance_m: tuple = (0.55, 0.65)  # from the pan axis (moved back automatically if needed)
    stereo_height_m: tuple = (0.25, 0.32)    # above the board
    stereo_view_height: float = 0.10         # keep this much approach height above the objects in view
    stereo_margin: float = 0.95
    v7_half_angle_deg: float = 30.0          # object sector the stereo camera can see
    stereo_near_m: float = 0.20              # near clipping plane: small values destroy depth precision,
    stereo_far_m: float = 2.0                # and with it the shadows of objects resting on the board
    v7_light_elev_deg: tuple = (30, 65)      # never nearly overhead: shadows stay visible
    v7_light_ambient: tuple = (0.30, 0.45)
    v7_light_diffuse: tuple = (0.55, 0.75)
    v7_light_behind_p: float = 0.85          # share of episodes lit from behind/beside the robot
    v7_light_behind_deg: float = 75.0        # spread of that light direction around "behind"
    v7_contact_shadows: bool = True          # draw object shadows geometrically (renderer's are too faint)
    v7_ready_pose: bool = True               # start each episode from a random low "ready pose" whose
    v7_ready_fwd: tuple = (0.08, 0.12)       # jaws are visible to the stereo camera (instead of the high
    v7_ready_lat: tuple = (-0.05, 0.05)      # home pose); closer to the base than any object, so the
    v7_ready_height: tuple = (0.07, 0.10)    # gripper never hides an object from the camera
    v7_ready_gripper_pct: tuple = (10.0, 40.0)
    # ---- v8 (scene_preset="v8"): v7 scene + wide camera randomisation + actuator model
    v8_azimuth_deg: tuple = (-45.0, 45.0)    # around the robot's front, + = robot's left
    v8_distance_m: tuple = (0.45, 0.95)      # horizontal, from the pan axis
    v8_height_m: tuple = (0.15, 0.60)
    v8_pitch_deg: tuple = (15.0, 55.0)       # looking down
    v8_roll_deg: float = 4.0
    v8_hfov_deg: tuple = (42.0, 68.0)
    v8_unseen_az_deg: float = 30.0           # held-out viewpoints: azimuth beyond this ...
    v8_unseen_dist_m: float = 0.85           # ... or farther than this
    v8_camera_split: str = "train"           # "train" | "unseen" | "all"
    v8_actuator: bool = True                 # dead band + command delay measured on the real arm
    v8_no_actuator_p: float = 0.15           # share of episodes with an ideal actuator (e.g. 12 V servos)
    v8_deadband_k: tuple = (8.2, 9.7)        # lag (deg) per N*m of gravity torque: shoulder_lift, elbow_flex
    v8_deadband_scale: tuple = (0.5, 1.5)    # per-episode random scale of those values
    v8_deadband_pos_deg: tuple = (0.1, 0.6)  # small error when moving in the other direction
    v8_delay_steps: tuple = (0, 2)           # command delay (control steps, inclusive)
    # ---- v9 (scene_preset="v9"): v8 scene + servo model, camera normally distributed around the real pose
    v9_scale: float = 1.0                    # width factor s: 0 = fixed camera at the real pose
    v9_center: dict = field(default_factory=lambda: {"distance_m": 0.525, "height_m": 0.312, "pitch_deg": 38.7,
                                                     "azimuth_deg": 0.0, "yaw_deg": 0.0, "roll_deg": 2.5})
    v9_sigma: dict = field(default_factory=lambda: {"distance_m": 0.03, "height_m": 0.025, "pitch_deg": 3.0,
                                                    "azimuth_deg": 4.0, "yaw_deg": 2.0, "roll_deg": 1.5})
    v9_hfov_deg: tuple = (44.1, 0.5)         # rectified 3.6 mm lens: mean, std (not scaled by s)
    v9_p_white_arm: float = 1.0              # both real arms are white PLA: only light-grey..white shades
    record_wrist: bool = True                # v7: also render the wrist camera (for comparisons)
    top_size: tuple = None            # defaults to front_size


def _rotate(v, axis, ang):
    axis = unit(axis)
    return v * np.cos(ang) + np.cross(axis, v) * np.sin(ang) + axis * (axis @ v) * (1 - np.cos(ang))


def _make_texture(rng, size=256):
    kind = rng.integers(3)
    base = int(rng.integers(150, 235))
    img = np.full((size, size, 3), base, np.uint8)
    if kind == 0:    # speckles
        for _ in range(300):
            c = rng.integers(100, 255, 3).tolist()
            cv2.circle(img, tuple(int(x) for x in rng.integers(0, size, 2)), int(rng.integers(1, 4)), c, -1)
    elif kind == 1:  # scratches
        for _ in range(60):
            g = int(rng.integers(base - 50, base + 1))
            p1, p2 = rng.integers(0, size, 2), rng.integers(0, size, 2)
            cv2.line(img, tuple(int(x) for x in p1), tuple(int(x) for x in p2), (g, g, g), 1)
    else:            # wood-like stripes
        y = np.arange(size)[:, None]
        stripes = (np.sin(y / rng.uniform(3, 9) + 3 * np.sin(np.arange(size)[None] / 40.0)) * 18).astype(np.int16)
        tint = rng.uniform(0.75, 1.0, 3)
        img = np.clip((base + stripes)[..., None] * tint, 0, 255).astype(np.uint8)
    return img


class SO101GraspEnv:
    def __init__(self, urdf_path, cfg=None, gui=False):
        self.cfg = cfg or EnvConfig()
        self.gui = gui
        self.pb = bullet_client.BulletClient(connection_mode=pybullet.GUI if gui else pybullet.DIRECT)
        if gui:
            self.pb.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        elif self.cfg.renderer == "egl":
            self._load_egl()
        self.substeps = self.cfg.physics_hz // self.cfg.control_hz
        self.pb.setGravity(0, 0, -9.81)
        self.pb.setPhysicsEngineParameter(fixedTimeStep=1.0 / self.cfg.physics_hz, numSolverIterations=150)

        # Kinematics twin (private client), then the physical robot with its base on the table.
        self.kin = SO101Kinematics(urdf_path)
        base_pos = np.array([0.0, 0.0, -self.kin.base_bottom_z])
        self.kin.base_offset = base_pos
        self.robot = self.pb.loadURDF(urdf_path, base_pos.tolist(), useFixedBase=True,
                                      flags=pybullet.URDF_USE_INERTIA_FROM_FILE)
        self.joint_idx, self.link_idx = name_maps(self.pb, self.robot)
        self.q_idx = [self.joint_idx[n] for n in ALL_JOINTS]
        self.g_link, self.jaw_link = self.link_idx[GRIPPER_LINK], self.link_idx[JAW_LINK]
        self.tcp_link = self.link_idx[TCP_LINK]
        for link in (self.g_link, self.jaw_link):
            self.pb.changeDynamics(self.robot, link, lateralFriction=self.cfg.jaw_friction)

        self.pan_axis = np.r_[self.kin.pan_axis_local + base_pos[:2], 0.0]
        if self.cfg.forward_yaw_deg is not None:
            self.kin.forward_yaw = math.radians(self.cfg.forward_yaw_deg)
        y = self.kin.forward_yaw
        self.fwd = np.array([math.cos(y), math.sin(y), 0.0])
        self.left = np.array([-math.sin(y), math.cos(y), 0.0])

        self.rng = np.random.default_rng()
        self.v7 = self.cfg.scene_preset in ("v7", "v8", "v9")   # v8 and v9 build on the v7 scene
        self.v8 = self.cfg.scene_preset in ("v8", "v9")    # v9 = v8 with a different camera distribution
        self.v9 = self.cfg.scene_preset == "v9"
        if self.v9:
            self.cfg.p_white_arm = self.cfg.v9_p_white_arm
        self.act_params = {"active": False}
        self._cmd_hist = collections.deque(maxlen=8)
        if self.v7:
            self.cfg.half_angle_deg = self.cfg.v7_half_angle_deg
        self._printed_shapes = self._find_printed_shapes()
        if self.v7:
            scene_v7.build_static(self)
            self.tex_pools, self.tex_info = scene_v7.load_texture_pool(self.pb, self.cfg, _make_texture)
            self.cam_pipe, self.stereo_cam = {}, {}
        else:
            self._build_static_scene()
        self.q_home = self._solve_home()
        self.objects = []
        self.light = {}
        self.front_cam = {}
        self.top_cam = {}
        self.top_rng = np.random.default_rng(0)

    # ================================================================= setup
    def _load_egl(self):
        try:
            spec = importlib.util.find_spec("eglRenderer")
            if spec is not None:
                self.pb.loadPlugin(spec.origin, "_eglRendererPlugin")
            else:
                self.pb.loadPlugin("eglRendererPlugin")
        except Exception as e:  # fall back to the CPU renderer
            print("EGL plugin not available, using TinyRenderer:", e)
            self.cfg.renderer = "tiny"

    def _find_printed_shapes(self):
        """(link, shape_index) of 3D-printed parts (non-dark visuals), for recolouring."""
        out, counter = [], {}
        for d in self.pb.getVisualShapeData(self.robot):
            link = d[1]
            k = counter.get(link, 0)
            counter[link] = k + 1
            if sum(d[7][:3]) > 0.6:
                out.append((link, k))
        return out

    def _build_static_scene(self):
        yaw_q = self.pb.getQuaternionFromEuler([0, 0, self.kin.forward_yaw])
        half = [0.45, 0.45, 0.01]
        center = self.pan_axis + self.fwd * 0.2 + np.array([0, 0, -0.01])
        self.table = self.pb.createMultiBody(
            0, self.pb.createCollisionShape(pybullet.GEOM_BOX, halfExtents=half),
            self.pb.createVisualShape(pybullet.GEOM_BOX, halfExtents=half), center.tolist(), yaw_q)
        wall_half = [0.01, 1.5, 1.0]
        wall_pos = self.pan_axis - self.fwd * 0.4 + np.array([0, 0, 0.98])
        self.wall = self.pb.createMultiBody(
            0, -1, self.pb.createVisualShape(pybullet.GEOM_BOX, halfExtents=wall_half), wall_pos.tolist(), yaw_q)
        self.pb.changeDynamics(self.table, -1, lateralFriction=0.8)

        # Texture pool created once and reused (avoids leaking textures every episode).
        rng = np.random.default_rng(12345)
        self.textures = []
        for _ in range(self.cfg.n_textures):
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            cv2.imwrite(tmp.name, _make_texture(rng))
            self.textures.append(self.pb.loadTexture(tmp.name))
            os.unlink(tmp.name)

    def _sample_ready_pose(self):
        """Random low ready pose (v7): jaws visible to the stereo camera, behind all objects."""
        c, r = self.cfg, self.rng
        target = (self.pan_axis + self.fwd * r.uniform(*c.v7_ready_fwd) + self.left * r.uniform(*c.v7_ready_lat)
                  + np.array([0, 0, r.uniform(*c.v7_ready_height)]))
        for tilt in (0, 20, 40):
            t = math.radians(tilt)
            q5, ok, _, _ = self.kin.solve(target, self.fwd * math.sin(t) - np.array([0, 0, math.cos(t)]), 0.0,
                                          seeds=[self.q_home[:5], np.zeros(5)])
            if ok:
                return np.r_[q5, self.kin.gripper_pct_to_rad(r.uniform(*c.v7_ready_gripper_pct))]
        return self.q_home

    def _solve_home(self):
        h = self.cfg.home_tcp
        target = self.pan_axis + self.fwd * h[0] + self.left * h[1] + np.array([0, 0, h[2]])
        # A ready pose above the table; the gripper may tilt toward the workspace if a
        # vertical gripper is not reachable at this height.
        for tilt in (0, 20, 40, 60):
            t = math.radians(tilt)
            q5, ok, pe, ae = self.kin.solve(target, self.fwd * math.sin(t) - np.array([0, 0, math.cos(t)]), 0.0,
                                            seeds=[np.zeros(5), [0, -0.5, 0.5, 1.0, 0]])
            if ok:
                break
        else:
            print(f"warning: home IK inexact (pos err {pe * 1000:.1f} mm, angle err {ae:.1f} deg)")
        return np.r_[q5, self.kin.gripper_pct_to_rad(self.cfg.home_gripper_pct)]

    # ================================================================= units
    def to_lerobot(self, q6):
        return np.r_[np.degrees(q6[:5]), self.kin.gripper_rad_to_pct(q6[5])]

    def from_lerobot(self, a6):
        return np.r_[np.radians(a6[:5]), self.kin.gripper_pct_to_rad(a6[5])]

    # ================================================================= episode API
    def reset(self, seed=None, render=True):
        self.rng = np.random.default_rng(seed)
        for o in self.objects:
            self.pb.removeBody(o["id"])
        self.objects = []
        if self.v7:
            scene_v7.randomize(self)
            if self.v8:
                ok = scene_v9.sample_camera_v9(self) if self.v9 else scene_v8.sample_camera_v8(self)
                if not ok:
                    scene_v7.sample_stereo_camera(self)
            else:
                scene_v7.sample_stereo_camera(self)
        else:
            self._randomize_appearance()
            self._sample_front_camera()
        self._sample_light()
        self._spawn_objects()
        for _ in range(20):        # v8: a camera may see too little of the object area -> new camera
            if self.objects or not self.v8:
                break
            scene_v9.sample_camera_v9(self) if self.v9 else scene_v8.sample_camera_v8(self)
            self._spawn_objects()
        if not self.objects:
            raise RuntimeError("could not place the target object where the cameras see it")
        self._sample_actuator()
        q_start = self._sample_ready_pose() if self.v7 and self.cfg.v7_ready_pose else self.q_home
        self.set_joint_positions(q_start)
        for _ in range(60):  # let objects settle
            self.pb.stepSimulation()
        for o in self.objects:
            o["z0"] = self.object_pose(o["id"])[0][2]
            o["pos"] = self.object_pose(o["id"])[0].tolist()
        # The overhead camera uses its own random stream, so enabling it leaves the scene,
        # instruction and front/wrist images identical to a run without it.
        self.top_rng = np.random.default_rng([0 if seed is None else seed, 2])
        if self.cfg.top_camera:
            self._sample_top_camera()
        tgt = self.objects[0]
        info = {
            "instruction": self.rng.choice(INSTRUCTIONS).format(tgt["descr"]),
            "target": {k: tgt[k] for k in ("id", "descr", "shape", "color", "size", "yaw", "pos")},
            "distractors": [{k: o[k] for k in ("descr", "shape", "size", "yaw", "pos")} for o in self.objects[1:]],
            "front_camera": self.front_cam, "light": self.light,
            "top_camera": self.top_cam if self.cfg.top_camera else None,
        }
        if self.v8:
            info.update({"camera_desc": scene_v8.camera_desc(self).tolist(), "actuator": dict(self.act_params)})
        if self.v7:
            info.update({"stereo_camera": self.stereo_cam, "camera_pipeline": self.cam_pipe,
                         "scene_preset": "v7", "texture_split": self.cfg.texture_split})
        return self.observe(render), info

    def step(self, action, render=True):
        q = np.clip(self.from_lerobot(np.asarray(action, float)), self.kin.lower, self.kin.upper)
        if self.v8 and self.act_params.get("active"):
            q = self._actuator(q)
        self._apply_targets(q)
        for _ in range(self.substeps):
            self.pb.stepSimulation()
        return self.observe(render)

    def observe(self, render=True):
        q = np.array([s[0] for s in self.pb.getJointStates(self.robot, self.q_idx)])
        obs = {"q": q, "state": self.to_lerobot(q), "tcp": self.tcp_pos()}
        if render and self.v7:
            imgs = scene_v7.render_stereo(self)
            if self.cfg.record_wrist:
                imgs["wrist"] = self.render_wrist(raw=True)
            obs["images"] = {k: scene_v7.camera_pipeline(v, self.cam_pipe, self.rng) for k, v in imgs.items()}
        elif render:
            obs["images"] = {"front": self.render_front(), "wrist": self.render_wrist()}
            if self.cfg.top_camera:
                obs["images"]["top"] = self.render_top()
        return obs

    def _sample_actuator(self):
        c, r = self.cfg, self.rng
        self._cmd_hist.clear()
        if not self.v8 or r.random() < c.v8_no_actuator_p:
            self.act_params = {"active": False}
            return
        # the same random draws whether or not the actuator model is switched on, so that
        # --no-actuator evaluations see exactly the same scenes as normal ones
        s = r.uniform(*c.v8_deadband_scale, size=2)
        pos = float(r.uniform(*c.v8_deadband_pos_deg))
        delay = int(r.integers(c.v8_delay_steps[0], c.v8_delay_steps[1] + 1))
        self.act_params = {"active": bool(c.v8_actuator), "k_deg_per_nm": [c.v8_deadband_k[0] * s[0],
                           c.v8_deadband_k[1] * s[1]], "pos_deg": pos, "delay": delay}

    def _actuator(self, q_cmd):
        """Real-servo behaviour measured on the 7.4 V arm: commands arrive `delay` steps late, and
        shoulder_lift / elbow_flex stop short when moving in the negative direction (against gravity) by
        k * |gravity torque|, but arrive within ~0.1-0.6 deg in the positive direction. Inside that
        asymmetric band the joint simply stays where it is."""
        p = self.act_params
        self._cmd_hist.append(np.array(q_cmd, float))
        q = self._cmd_hist[max(0, len(self._cmd_hist) - 1 - p["delay"])].copy()
        q_meas = np.array([st[0] for st in self.pb.getJointStates(self.robot, self.q_idx)])
        zeros = [0.0] * len(q_meas)
        tau = np.array(self.pb.calculateInverseDynamics(self.robot, [float(v) for v in q_meas], zeros, zeros))
        lp = math.radians(p["pos_deg"])
        for k, j in enumerate((1, 2)):                       # shoulder_lift, elbow_flex
            lag = math.radians(p["k_deg_per_nm"][k] * abs(tau[j]))
            lo, hi = q[j] - lp, q[j] + lag
            if lo <= q_meas[j] <= hi:
                q[j] = q_meas[j]                             # inside the dead band: no motion
            elif q_meas[j] > hi:
                q[j] = hi                                    # moving down: stops `lag` short
            else:
                q[j] = lo
        return q

    def set_joint_positions(self, q6):
        for idx, v in zip(self.q_idx, q6):
            self.pb.resetJointState(self.robot, idx, float(v))
        self._apply_targets(q6)

    def _apply_targets(self, q6):
        c = self.cfg
        self.pb.setJointMotorControlArray(self.robot, self.q_idx, pybullet.POSITION_CONTROL,
                                          targetPositions=[float(v) for v in q6],
                                          forces=[c.arm_torque] * 5 + [c.gripper_torque])

    # ================================================================= state queries
    def tcp_pos(self):
        return np.array(self.pb.getLinkState(self.robot, self.tcp_link, computeForwardKinematics=True)[4])

    def object_pose(self, body):
        p, o = self.pb.getBasePositionAndOrientation(body)
        return np.array(p), np.array(o)

    def grasp_state(self, body, z0):
        jaw = len(self.pb.getContactPoints(self.robot, body, linkIndexA=self.jaw_link)) > 0
        fixed = len(self.pb.getContactPoints(self.robot, body, linkIndexA=self.g_link)) > 0
        lifted = self.object_pose(body)[0][2] > z0 + 0.03
        table_hit = any(c[3] in (self.g_link, self.jaw_link) for c in self.pb.getContactPoints(self.robot, self.table))
        return {"jaw_contact": jaw, "fixed_contact": fixed, "lifted": bool(lifted),
                "grasped": bool(jaw and fixed and lifted), "gripper_table_contact": table_hit}

    # ================================================================= randomisation
    def _randomize_appearance(self):
        r = self.rng
        for body in (self.table, self.wall):
            tint = r.uniform(0.65, 1.0, 3).tolist() + [1.0]
            self.pb.changeVisualShape(body, -1, textureUniqueId=int(r.choice(self.textures)), rgbaColor=tint)
        base = np.array(PRINT_COLORS[r.integers(len(PRINT_COLORS))])
        color = np.clip(base + r.normal(0, 0.03, 3), 0, 1).tolist() + [1.0]
        for link, k in self._printed_shapes:
            try:
                self.pb.changeVisualShape(self.robot, link, shapeIndex=k, rgbaColor=color)
            except TypeError:  # very old pybullet without shapeIndex
                break

    def _sample_front_camera(self):
        # Both presets consume exactly the same random draws in the same order, so the rest of
        # the scene (light, objects, instruction) is identical whichever preset is used.
        r = self.rng
        u_d, u_lat, u_h = r.uniform(0.45, 0.60), r.uniform(-0.06, 0.06), r.uniform(0.30, 0.42)
        u_tf, u_tl, u_tz = r.uniform(0.10, 0.17), r.uniform(-0.02, 0.02), r.uniform(0.02, 0.06)
        u_roll, u_fov = r.uniform(-2, 2), r.uniform(48, 58)
        if self.cfg.camera_preset == "v4" or self.v7:  # same spread, shifted back/up with the target moved forward
            eye = self.pan_axis + self.fwd * (u_d + 0.07) + self.left * u_lat + np.array([0, 0, u_h + 0.03])
        else:
            eye = self.pan_axis + self.fwd * u_d + self.left * u_lat + np.array([0, 0, u_h])
        if self.cfg.camera_preset == "v4" or self.v7:
            # aim at the middle of the object area (forward 0.15-0.19 m), lower target
            tgt = (self.pan_axis + self.fwd * (0.15 + (u_tf - 0.10) / 0.07 * 0.04) + self.left * u_tl
                   + np.array([0, 0, 0.01 + (u_tz - 0.02) / 0.04 * 0.03]))
        else:
            tgt = self.pan_axis + self.fwd * u_tf + self.left * u_tl + np.array([0, 0, u_tz])
        up = _rotate(np.array([0.0, 0.0, 1.0]), tgt - eye, math.radians(u_roll))
        if self.cfg.camera_preset == "v4" or self.v7:
            # narrowest field of view that keeps the whole object area (and a raised gripper above
            # it) in view, plus an 8-18 % margin
            fov = self._required_fov(eye, tgt, up, self.cfg.front_size) * (1.08 + (u_fov - 48) / 10 * 0.10)
        else:
            fov = u_fov
        self.front_cam = {"eye": eye.tolist(), "target": tgt.tolist(), "up": up.tolist(), "fov": float(fov)}

    def _required_fov(self, eye, tgt, up, size, z_max=0.12):
        """Vertical FOV (deg) needed to see every point of the object area up to z_max."""
        f = unit(tgt - eye)
        rgt = unit(np.cross(f, up))
        u = np.cross(rgt, f)
        c = self.cfg
        aspect = size[0] / size[1]
        need = 0.0
        for rad in np.linspace(c.r_range[0], c.r_range[1], 7):
            for b in np.radians(np.linspace(-c.half_angle_deg, c.half_angle_deg, 11)):
                for z in (0.0, z_max):
                    p = (self.pan_axis + rad * (math.cos(b) * self.fwd + math.sin(b) * self.left)
                         + np.array([0, 0, z])) - eye
                    depth = p @ f
                    tan_x, tan_y = abs(p @ rgt) / depth, abs(p @ u) / depth
                    need = max(need, tan_y, tan_x / aspect)
        return math.degrees(2 * math.atan(need))

    def _sample_top_camera(self):
        r = self.top_rng
        center = self.pan_axis + self.fwd * 0.17
        eye = center + self.fwd * r.uniform(-0.02, 0.02) + self.left * r.uniform(-0.02, 0.02) \
            + np.array([0, 0, r.uniform(0.55, 0.65)])
        up = _rotate(self.fwd.copy(), np.array([0.0, 0.0, 1.0]), math.radians(r.uniform(-3, 3)))
        self.top_cam = {"eye": eye.tolist(), "target": center.tolist(), "up": up.tolist(),
                        "fov": float(r.uniform(38, 44))}

    def _sample_light(self):
        r = self.rng
        el_range = self.cfg.v7_light_elev_deg if self.v7 else (35, 80)
        az, el = r.uniform(0, 2 * math.pi), math.radians(r.uniform(*el_range))
        if self.v7 and r.random() < self.cfg.v7_light_behind_p:
            # light from behind / beside the robot (window, lamp), as in the real room: shadows fall
            # towards the camera or sideways instead of hiding behind the objects
            az = self.kin.forward_yaw + math.pi + math.radians(r.uniform(-self.cfg.v7_light_behind_deg,
                                                                         self.cfg.v7_light_behind_deg))
        self.light = {
            "lightDirection": [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
            "lightColor": np.clip(1.0 + r.normal(0, 0.06, 3), 0.8, 1.0).tolist(),
            "lightAmbientCoeff": float(r.uniform(*(self.cfg.v7_light_ambient if self.v7 else (0.35, 0.6)))),
            "lightDiffuseCoeff": float(r.uniform(*(self.cfg.v7_light_diffuse if self.v7 else (0.4, 0.7)))),
            "lightSpecularCoeff": float(r.uniform(0.05, 0.3)),
        }

    def _spawn_objects(self):
        c, r = self.cfg, self.rng
        n = 1 + int(r.integers(c.n_distractors[0], c.n_distractors[1] + 1))
        combos = list(itertools.product(range(len(OBJECT_COLORS)), SHAPES))
        picks = [combos[i] for i in r.choice(len(combos), size=n, replace=False)]
        placed = []
        for ci, shape in picks:
            xy = self._sample_xy(placed)
            if xy is None:
                break
            placed.append(xy)
            cname, rgb = OBJECT_COLORS[ci]
            yaw = float(r.uniform(-math.pi, math.pi))
            size = float(r.uniform(*c.cube_size) if shape == "cube" else 2 * r.uniform(*c.sphere_radius))
            body = self._create_object(shape, size, rgb, [xy[0], xy[1], size / 2 + 0.001], yaw)
            self.objects.append({"id": body, "descr": f"{cname} {shape}", "shape": shape,
                                 "color": cname, "size": size, "yaw": yaw})

    def _sample_xy(self, placed):
        c, r = self.cfg, self.rng
        for _ in range(200):
            rad, th = r.uniform(*c.r_range), math.radians(r.uniform(-c.half_angle_deg, c.half_angle_deg))
            xy = self.pan_axis[:2] + rad * (math.cos(th) * self.fwd[:2] + math.sin(th) * self.left[:2])
            if all(np.linalg.norm(xy - p) > c.min_separation for p in placed):
                if self.v8 and not scene_v8.object_visible(self, xy):
                    continue
                return xy
        return None

    def _create_object(self, shape, size, rgb, pos, yaw):
        if shape == "cube":
            h = [size / 2] * 3
            col = self.pb.createCollisionShape(pybullet.GEOM_BOX, halfExtents=h)
            vis = self.pb.createVisualShape(pybullet.GEOM_BOX, halfExtents=h, rgbaColor=list(rgb) + [1])
            mass = 0.03
        else:
            col = self.pb.createCollisionShape(pybullet.GEOM_SPHERE, radius=size / 2)
            vis = self.pb.createVisualShape(pybullet.GEOM_SPHERE, radius=size / 2, rgbaColor=list(rgb) + [1])
            mass = 0.02
        body = self.pb.createMultiBody(mass, col, vis, pos, self.pb.getQuaternionFromEuler([0, 0, yaw]))
        self.pb.changeDynamics(body, -1, lateralFriction=self.cfg.object_friction,
                               rollingFriction=0.001, spinningFriction=0.001)
        return body

    # ================================================================= cameras
    def camera_names(self):
        if self.v7:
            return ["left", "right"] + (["wrist"] if self.cfg.record_wrist else [])
        return ["front", "wrist"] + (["top"] if self.cfg.top_camera else [])

    def _render(self, eye, target, up, fov, size, near=0.01, rng=None, raw=False, far=3.0):
        w, h = size
        view = self.pb.computeViewMatrix(list(eye), list(target), list(up))
        proj = self.pb.computeProjectionMatrixFOV(fov, w / h, near, far)
        renderer = pybullet.ER_TINY_RENDERER if self.cfg.renderer == "tiny" and not self.gui \
            else pybullet.ER_BULLET_HARDWARE_OPENGL
        img = self.pb.getCameraImage(w, h, view, proj, renderer=renderer, shadow=1, **self.light)
        rgb = np.reshape(np.asarray(img[2], dtype=np.uint8), (h, w, 4))[:, :, :3]
        if self.cfg.camera_noise_std > 0 and not raw:
            noise = (rng or self.rng).normal(0, self.cfg.camera_noise_std, rgb.shape)
            rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return rgb

    def render_front(self):
        c = self.front_cam
        return self._render(c["eye"], c["target"], c["up"], c["fov"], self.cfg.front_size)

    def render_top(self):
        c = self.top_cam
        return self._render(c["eye"], c["target"], c["up"], c["fov"], self.cfg.top_size or self.cfg.front_size,
                            rng=self.top_rng)

    def render_wrist(self, raw=False):
        s = self.pb.getLinkState(self.robot, self.g_link, computeForwardKinematics=True)
        g_pos, g_rot = np.array(s[4]), np.array(self.pb.getMatrixFromQuaternion(s[5])).reshape(3, 3)
        k = self.kin
        a, cl, hg = g_rot @ k.approach_local, g_rot @ k.closing_local, g_rot @ k.hinge_local
        tcp = g_pos + g_rot @ k.tcp_local
        if self.cfg.camera_preset == "v4" or self.v7:
            # further back and further to the hinge side, looking at the grasp point: the object
            # stays visible between the jaws until they close
            g = g_pos + g_rot @ k.grasp_local(0.03)
            eye = tcp + a * -0.095 + hg * 0.05
            return self._render(eye, g + a * 0.01, hg, 70.0, self.cfg.wrist_size, near=0.005, raw=raw)
        o = self.cfg.wrist_cam
        eye = tcp + a * o[0] + cl * o[1] + hg * o[2]
        return self._render(eye, tcp + a * 0.03, hg, self.cfg.wrist_fov, self.cfg.wrist_size, near=0.005, raw=raw)

    def add_marker(self, pos, rgba, radius=0.003):
        """Visual-only sphere (no collision), e.g. to show grasp points in images."""
        vis = self.pb.createVisualShape(pybullet.GEOM_SPHERE, radius=radius, rgbaColor=list(rgba))
        return self.pb.createMultiBody(0, -1, vis, list(pos))

    def gripper_frame(self):
        s = self.pb.getLinkState(self.robot, self.g_link, computeForwardKinematics=True)
        return np.array(s[4]), np.array(self.pb.getMatrixFromQuaternion(s[5])).reshape(3, 3)

    def close(self):
        self.pb.disconnect()
        self.kin.pb.disconnect()
