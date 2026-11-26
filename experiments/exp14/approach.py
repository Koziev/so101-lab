"""
The "Approach the object" command (experiment 12): definitions shared by the trajectory generator and the
evaluation.

Goal pose: gripper CLOSED, its tip 1.5 cm (CLEARANCE) from the target object's surface, pointing at the
object's centre. The pointing direction is the steepest one the arm reaches: vertical, else tilted 15/30/45 deg
(fingers pointing away from the base, as the grasp labels of experiments 10 and 11). "Tip" = the point of the
closed gripper farthest along the fingers (the longer jaw's end), at the fixed jaw's gripping face, i.e. between
the two closed jaws.

Expert trajectory (one episode, 30 Hz, absolute joint commands as in LeRobot):
  start   the arm's rest pose (folded, from the setup's taught workspace), gripper closed
  transit min-jerk motion in joint space to the PRE-approach pose: the goal pose moved PRE_BACK further back
          along the pointing direction (above the object); via the ready pose if the direct path would come
          too close to the board or to another object
  descend min-jerk motion to the goal pose
  hold    the goal pose is held (the policy has to learn to stop)
During descend and hold the command is corrected from the joint readings (integral term), so that the arm
reaches the goal although the servos give way under load (sag); the labels are these corrected commands.

Success of an approach (evaluate at the end of an episode):
  tip within TOL_XY of the goal tip position across the pointing direction, its distance to the surface
  along the pointing direction within TOL_GAP, the gripper's axis within TOL_ANGLE of the goal direction,
  the jaws closed, and no contact of the arm with any object or the board on the way.
"""
import math
import os

import numpy as np

import common

CLEARANCE = 0.015          # m, tip to object surface at the goal (the command says 1-2 cm)
PRE_BACK = 0.05            # m, the pre-approach pose is this much further back along the pointing direction
TILTS_DEG = (0, 15, 30, 45)
GRIPPER_CLOSED_PCT = 1.0
HOLD_S = (0.8, 1.2)        # s, how long the goal pose is held at the end of an episode
HOLD_LONG_S = (2.0, 2.4)   # s, the hold of a "long hold" episode: longer than one SmolVLA chunk (50 steps)
DIRECT_S = (0.5, 0.9)      # s, the short correction from a near-goal start straight to the goal
SPEED_DEG_S = (55.0, 75.0)  # joint-space speed of the transit (largest joint), per episode
DESCEND_S = (0.6, 0.9)
COMP_GAIN = 0.3            # integral correction of the command from the readings, per step
COMP_LIMIT = 8.0           # deg
COMP_LAG_STEPS = 4         # exp14: see comp_reference
COMP_STILL_STEPS = 3       # exp14: the correction integrates only once the nominal pose has not changed for this many
COMP_STILL_DEG = 0.05      # steps (by more than this): during a motion the readings lag the command (delay, tracking),
                           # and integrating that lag wound the correction up into an overshoot of several mm at the
                           # end of every descent
MIN_TIP_Z = 0.012          # m, nominal path: tip at least this high above the board ...
OBJ_MARGIN = 0.012         # m, ... and this far above any object it passes over (except the target at the end)

TOL_XY = 0.010             # m, success: across the pointing direction
TOL_GAP = (0.005, 0.025)   # m, success: tip-to-surface distance along the pointing direction
TOL_ANGLE = 15.0           # deg, success: angle between gripper axis and goal direction

VERBS = ["approach {np}", "move to {np}", "go to {np}", "move the gripper above {np}",
         "point the gripper at {np}", "bring the gripper to {np}"]
SUPERLATIVE_VERBS = ["find the {sup} of the {plural} and approach it",
                     "compare the {plural} on the table, choose the {sup} and move the gripper above it",
                     "look at the {plural}, approach the {sup} one"]
TASK_KWARGS = {"verbs": VERBS, "superlative_verbs": SUPERLATIVE_VERBS}

def task_kwargs(objects="2,4", white_weight=0.5, weights="", shapes=""):
    """tasks.sample_task arguments for approach scenes. objects "min,max"; weights "full=0.5,colour=0.25,..."
    (default: tasks.DEFAULT_WEIGHTS); shapes "cube,ball,..." (default: all of objects_v11.SHAPES).
    One object per scene needs weights without "superlative"."""
    from so101_sim import objects_v11 as ob
    n = tuple(int(v) for v in str(objects).split(","))
    tk = dict(TASK_KWARGS, n_objects=n, white_weight=white_weight)
    if shapes:
        tk["shapes"] = [v.strip() for v in shapes.split(",") if v.strip()]
        bad = [v for v in tk["shapes"] if v not in ob.SHAPES]
        if bad:
            raise SystemExit(f"unknown shapes {bad}; known: {', '.join(ob.SHAPES)}")
    if weights:
        tk["weights"] = {k: float(v) for k, v in (p.split("=") for p in weights.split(","))}
    if n[1] < 2 and tk.get("weights", {"superlative": 1}).get("superlative", 0) > 0:
        raise SystemExit('one object per scene: pass --weights without superlative, e.g. full=0.5,colour=0.25,shape=0.25')
    return tk


REST_REAL = np.array([3.5, -104.6, 96.0, 71.9, -4.4])   # first pose of the 12 V arm's taught workspace


# ------------------------------------------------------------------------------------------------ geometry
def tip_local(env):
    """The fixed jaw's end at its gripping face, in the gripper_link frame."""
    from so101_sim.scene_real import JawTips
    if getattr(env, "_jaw_tips", None) is None:
        env._jaw_tips = JawTips(env.kin)
    J = env._jaw_tips
    t = J.tips["fixed"]
    d = max(J.tips["fixed"]["d_tip"], J.tips["moving"]["d_tip"])
    u = float(np.interp(min(d, t["d_tip"]) - 0.002, t["face_d"], t["face_u"])) * J.sign
    return J.o + J.a * d + J.c * u


def gripper_axis_world(env):
    """Pointing direction of the gripper (approach axis) in the world, current pose of the simulated arm."""
    s = env.pb.getLinkState(env.robot, env.g_link, computeForwardKinematics=True)
    R = np.array(env.pb.getMatrixFromQuaternion(s[5])).reshape(3, 3)
    return R @ env.kin.approach_local, np.array(s[4]), R


def tip_world(env):
    a, pos, R = gripper_axis_world(env)
    return pos + R @ tip_local(env)


def tip_of(env, deg5):
    """Tip position for joint angles (deg, simulator convention), from the kinematic model."""
    q = env.from_lerobot(np.r_[np.asarray(deg5, float)[:5], GRIPPER_CLOSED_PCT])
    return env.kin.points(q[:5], tip_local(env))[0]


def object_vertices(env, obj):
    pos, orn = env.pb.getBasePositionAndOrientation(obj["id"])
    R = np.array(env.pb.getMatrixFromQuaternion(orn)).reshape(3, 3)
    v = obj["spec"].get("vertices")
    if v is None:                                    # ball: points on the sphere
        k = np.arange(64) + 0.5
        phi, th = np.arccos(1 - 2 * k / 64), math.pi * (1 + 5 ** 0.5) * k
        v = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1) * obj["size"] / 2
    return np.asarray(pos, float), np.asarray(pos, float) + np.asarray(v, float) @ R.T


# ------------------------------------------------------------------------------------------------ clearance
# exp14: the tip is not the lowest point of the gripper, and the jaws and the gripper body are wider than it, so
# the expert checks the real collision shapes of the arm (gripper body, moving jaw, wrist camera; all links for
# objects) at the nominal poses of a path, not only the tip.
GRIPPER_BOARD_CLEAR = 0.004    # m, gripper links to the board along an approach path (nominal pose)
ARM_TARGET_CLEAR = 0.003       # m, any link to the target object along the approach (descent and goal)
ARM_OTHER_CLEAR = 0.008        # m, any link to the other objects along the whole path
PROBE_RANGE = 0.05             # m, getClosestPoints search distance (larger distances read as this)
START_GRACE = 12               # path steps: a rest start that leans on the board may stay this low this long


def gripper_links(env):
    return (env.g_link, env.jaw_link) + tuple(getattr(env, "cam_links", ()))


class pose_probe:
    """Context manager: the simulated arm is put into poses with resetJointState (no physics step) to measure
    distances with getClosestPoints; the joint states are restored on exit, motor targets are not touched."""

    def __init__(self, env):
        self.env = env

    def __enter__(self):
        pb, r = self.env.pb, self.env.robot
        self.saved = [pb.getJointState(r, j)[:2] for j in range(pb.getNumJoints(r))]
        return self

    def __exit__(self, *exc):
        pb, r = self.env.pb, self.env.robot
        for j, (pos, vel) in enumerate(self.saved):
            pb.resetJointState(r, j, pos, vel)
        return False

    def set(self, q_deg5, grip_pct=GRIPPER_CLOSED_PCT):
        env = self.env
        q6 = env.from_lerobot(np.r_[np.asarray(q_deg5, float)[:5], grip_pct])
        for idx, v in zip(env.q_idx, q6):
            env.pb.resetJointState(env.robot, idx, float(v))

    def board(self):
        """Smallest distance (m, negative = penetration) of the gripper links to the board."""
        env = self.env
        d = PROBE_RANGE
        for link in gripper_links(env):
            for c in env.pb.getClosestPoints(env.robot, env.table, PROBE_RANGE, linkIndexA=link):
                d = min(d, float(c[8]))
        return d

    def body(self, obj_id):
        """Smallest distance (m) of any link of the arm to a body."""
        d = PROBE_RANGE
        for c in self.env.pb.getClosestPoints(self.env.robot, obj_id, PROBE_RANGE):
            d = min(d, float(c[8]))
        return d


def gripper_board_distance(env, rng=0.3):
    """Physical distance (m) of the gripper links to the board in the current state (up to rng)."""
    d = [float(c[8]) for l in gripper_links(env)
         for c in env.pb.getClosestPoints(env.robot, env.table, rng, linkIndexA=l)]
    return min(d + [rng])


def gripper_board_clearance(env, q_deg5, grip_pct=GRIPPER_CLOSED_PCT):
    """Board clearance (m) of the gripper links at a nominal pose (deg, LeRobot units)."""
    with pose_probe(env) as p:
        p.set(q_deg5, grip_pct)
        return p.board()


ROLL_SYMMETRY_DEG = {"cube": (90.0, 0.0), "hex_prism": (60.0, 30.0)}   # (symmetry, face normal vs the body's yaw)


def face_roll(env, obj, q5):
    """wrist_roll (rad) that turns the jaws' closing axis onto a face normal of the object (cube, hex prism), so
    that a later grasp closes on two parallel faces; None for round bodies (any roll will do)."""
    sym = ROLL_SYMMETRY_DEG.get(obj["shape"])
    if sym is None:
        return None
    _, orn = env.pb.getBasePositionAndOrientation(obj["id"])
    yaw = env.pb.getEulerFromQuaternion(orn)[2] + math.radians(sym[1])
    return env.kin.best_roll(q5, yaw, sym[0])


def goal(env, obj, tilts=TILTS_DEG, align_roll=False, check_clear=True):
    """Goal and pre-approach poses for an object: dict with q (5 angles, deg), q_pre, tip, centre, direction,
    tilt; None if the arm reaches neither. align_roll (experiment 14): the wrist roll puts the jaws' closing axis
    on a face normal of a cube or hex prism (face_roll), so that the grasp that follows needs no turn.
    check_clear (exp14): a tilt is skipped if at its goal pose any link of the arm is closer than
    ARM_TARGET_CLEAR to the object or the gripper closer than GRIPPER_BOARD_CLEAR to the board."""
    centre, verts = object_vertices(env, obj)
    radial = centre[:2] - env.pan_axis[:2]
    radial = np.r_[radial / (np.linalg.norm(radial) + 1e-9), 0.0]
    local = tip_local(env)
    seeds = [env.q_home[:5], np.zeros(5)]
    for tilt in tilts:
        t = math.radians(tilt)
        a = radial * math.sin(t) - np.array([0, 0, math.cos(t)])
        a /= np.linalg.norm(a)                                        # from the gripper towards the object
        support = float(np.max((verts - centre) @ (-a)))              # surface along -a, from the centre
        tip = centre - a * (support + CLEARANCE)
        q5, ok, _, _ = env.kin.solve(tip, a, 0.0, seeds, local)
        if not ok:
            continue
        roll = face_roll(env, obj, q5) if align_roll else None
        if roll is not None:
            q5, ok, _, _ = env.kin.solve(tip, a, roll, [q5] + seeds, local)
            if not ok:
                continue
        roll = 0.0 if roll is None else roll
        pre_tip = tip - a * PRE_BACK
        q5p, okp, _, _ = env.kin.solve(pre_tip, a, roll, [q5] + seeds, local)
        if not okp:
            continue
        deg = lambda q: env.to_lerobot(np.r_[q, env.kin.gripper_closed])[:5]
        if check_clear:                       # exp14: the whole arm, not only the tip, clear at the goal pose
            with pose_probe(env) as p:
                p.set(deg(q5))
                if p.body(obj["id"]) < ARM_TARGET_CLEAR or p.board() < GRIPPER_BOARD_CLEAR:
                    continue
        return {"q": deg(q5), "q_pre": deg(q5p), "tip": tip, "pre_tip": pre_tip, "centre": centre,
                "direction": a, "tilt": tilt, "surface": float(support)}
    return None


def evaluate_pose(env, g):
    """Errors of the arm's current pose against a goal (dict from goal())."""
    a_now, _, _ = gripper_axis_world(env)
    tip = tip_world(env)
    a = g["direction"]
    d = tip - g["tip"]
    along = float(d @ a)                                              # + = closer to the object than the goal
    across = float(np.linalg.norm(d - a * along))
    gap = float((g["centre"] - tip) @ a) - g["surface"]               # tip to surface along the direction
    angle = math.degrees(math.acos(float(np.clip(a_now @ a, -1, 1))))
    return {"across_cm": 100 * across, "gap_cm": 100 * gap, "angle_deg": angle,
            "tip": tip.tolist(), "dist_cm": float(100 * np.linalg.norm(d))}


def success(errors, contact, gripper_pct):
    return (errors["across_cm"] <= 100 * TOL_XY and 100 * TOL_GAP[0] <= errors["gap_cm"] <= 100 * TOL_GAP[1]
            and errors["angle_deg"] <= TOL_ANGLE and not contact and gripper_pct <= 15.0)


TOUCH_DIST = 0.0           # m, a contact point counts as a touch below this contactDistance. PyBullet also reports
                           # points of separated surfaces (positive distance, inside the contact breaking threshold)


def touching(points, links=None, tol=TOUCH_DIST):
    """Any of getContactPoints' points a real touch (contactDistance < tol), optionally only on these links?"""
    return any(float(c[8]) < tol and (links is None or c[3] in links) for c in points)


def arm_contact(env):
    """(object contact, board contact): any link of the arm touching an object; the gripper, the moving jaw or the
    wrist camera (sim v14) touching the board (the base and the folded rest pose may lean on it). exp14: only
    points with a negative contactDistance count (touching)."""
    obj = any(touching(env.pb.getContactPoints(env.robot, o["id"])) for o in env.objects)
    table = getattr(env, "table", None)
    links = (env.g_link, env.jaw_link) + tuple(getattr(env, "cam_links", ()))     # exp14: the wrist camera too
    board = table is not None and touching(env.pb.getContactPoints(env.robot, table), links)
    return bool(obj), bool(board)


# ------------------------------------------------------------------------------------------------ rest pose
def rest_pose(setup_dir=None, real=None):
    """The arm's rest pose in the simulator's convention (deg, 5 joints): the real rest reading (REST_REAL or
    `real`) plus the setup's joint offsets (simulator = real + offset)."""
    q = np.asarray(REST_REAL if real is None else real, float)[:5].copy()
    if setup_dir and os.path.exists(os.path.join(setup_dir, "setup.json")):
        from setup_io import Setup
        q = q + np.asarray(Setup(setup_dir).offsets(), float)[:5]
    return q


REST_NOISE_DEG = 2.0       # the real arm never rests at exactly the same angles


def start_from_rest(env, rest_deg5, rng, settle_steps=10):
    """Put the simulated arm in (a jittered copy of) the rest pose, gripper closed, and let it settle under the
    actuator model. Returns the observation (not rendered)."""
    q = np.asarray(rest_deg5, float) + rng.normal(0, REST_NOISE_DEG, 5)
    cmd = np.r_[q, GRIPPER_CLOSED_PCT]
    env.set_joint_positions(env.from_lerobot(cmd))
    if hasattr(env, "_cmd_hist"):
        env._cmd_hist.clear()
    if any(env.pb.getClosestPoints(env.robot, o["id"], REST_OBJ_CLEAR) for o in env.objects):
        return None                           # exp14: an object placed against the folded arm
    obs = None
    for _ in range(settle_steps):
        obs = env.step(cmd, render=False)
    return obs


REST_OBJ_CLEAR = 0.005     # m, rest start: every link of the arm at least this far from every object


# ------------------------------------------------------------------------------------------------ random starts
START_MIN_DIST = 0.05      # m, gripper (both jaws, closed) at least this far from the target object
START_ARM_CLEAR = 0.01     # m, every link of the arm at least this far from every object
START_TRIES = 30


def load_workspace(setup_dir):
    """The setup's taught safe workspace (calib/workspace.py, real readings) and the joint offsets that turn
    real readings into simulator angles. (None, None) if the setup has no workspace.json."""
    from calib.workspace import Workspace
    from setup_io import Setup
    s = Setup(setup_dir)
    d = s.load("workspace.json")
    if not d or not d.get("points"):
        return None, None
    return Workspace(d["points"]), np.asarray(s.offsets(), float)[:5]


def start_clear(env, target):
    """Is the arm, in its current (just set) pose, clear of the objects: gripper >= START_MIN_DIST from the
    target, every link >= START_ARM_CLEAR from every object, jaws above the board?"""
    pb = env.pb
    for link in (env.g_link, env.jaw_link):
        if pb.getClosestPoints(env.robot, target["id"], START_MIN_DIST, linkIndexA=link):
            return False
    for o in env.objects:
        if pb.getClosestPoints(env.robot, o["id"], START_ARM_CLEAR):
            return False
    return float(tip_world(env)[2]) >= MIN_TIP_Z


def start_from_workspace(env, ws, offsets, rng, target, settle_steps=10):
    """Put the arm in a random pose of the taught safe workspace (Workspace.sample: between two nearby taught
    poses), gripper closed, clear of the objects (start_clear), and let it settle. Returns the observation, or
    None if no such pose was found in START_TRIES draws."""
    for _ in range(START_TRIES):
        q = np.asarray(ws.sample(1, rng)[0], float)[:5] + offsets
        cmd = np.r_[q, GRIPPER_CLOSED_PCT]
        env.set_joint_positions(env.from_lerobot(cmd))
        if not start_clear(env, target):
            continue
        if hasattr(env, "_cmd_hist"):
            env._cmd_hist.clear()
        obs = None
        for _ in range(settle_steps):
            obs = env.step(cmd, render=False)
        if not any(arm_contact(env)):
            return obs
    return None


def start_kind(seed, random_frac, near_frac=0.0):
    """'workspace' for a share random_frac of the seeds, 'near' for a share near_frac, else 'rest'; from its own
    random stream, so the other draws of an episode do not depend on it (recorder and evaluation agree for the
    same seed; with near_frac=0 the assignment is that of v12c)."""
    u = np.random.default_rng([int(seed), 12, 2]).random()
    return "workspace" if u < random_frac else ("near" if u < random_frac + near_frac else "rest")


NEAR_ACROSS = (0.005, 0.03)   # m, near-goal start: tip moved this far across the pointing direction ...
NEAR_ALONG = (-0.007, 0.03)   # m, ... and this far back along it (negative = closer to the object)
NEAR_TILT_DEG = 8.0           # gripper axis up to this far off the goal direction
NEAR_CLEAR = 0.003            # m, every link at least this far from every object


def start_near_goal(env, rng, target, settle_steps=10):
    """Put the arm in a pose close to the goal: the goal tip moved 0.5-3 cm across the pointing direction and
    -0.7..+3 cm along it (tip-to-surface 0.8-4.5 cm), the axis tilted up to 8 deg; gripper closed, clear of
    the objects and the board. Returns the settled observation, or None."""
    g = goal(env, target)
    if g is None:
        return None
    a = g["direction"]
    local = tip_local(env)
    seed_q = env.from_lerobot(np.r_[g["q"], GRIPPER_CLOSED_PCT])[:5]

    def perp():
        v = rng.normal(size=3)
        v -= a * float(v @ a)
        return v / (np.linalg.norm(v) + 1e-12)

    for _ in range(START_TRIES):
        tip = g["tip"] + perp() * rng.uniform(*NEAR_ACROSS) - a * rng.uniform(*NEAR_ALONG)
        t = math.radians(rng.uniform(0, NEAR_TILT_DEG))
        d = a * math.cos(t) + perp() * math.sin(t)
        q5, ok, _, _ = env.kin.solve(tip, d, 0.0, [seed_q, env.q_home[:5]], local)
        if not ok:
            continue
        q = env.to_lerobot(np.r_[q5, env.kin.gripper_closed])[:5]
        cmd = np.r_[q, GRIPPER_CLOSED_PCT]
        env.set_joint_positions(env.from_lerobot(cmd))
        if any(env.pb.getClosestPoints(env.robot, o["id"], NEAR_CLEAR) for o in env.objects):
            continue
        if float(tip_world(env)[2]) < MIN_TIP_Z:
            continue
        if hasattr(env, "_cmd_hist"):
            env._cmd_hist.clear()
        obs = None
        for _ in range(settle_steps):
            obs = env.step(cmd, render=False)
        if not any(arm_contact(env)):
            return obs
    return None


def start_episode(env, kind, rest_deg5, ws, offsets, rng):
    """kind 'rest', 'workspace' or 'near'; returns the settled observation or None."""
    if kind == "rest":
        return start_from_rest(env, rest_deg5, rng)
    if kind == "near":
        return start_near_goal(env, rng, env.objects[0])
    if ws is None:
        raise SystemExit("random starts need the setup's workspace.json (calib/workspace.py record)")
    return start_from_workspace(env, ws, offsets, rng, env.objects[0])


def target_moved(env, target, start_pos, tol=0.01):
    """Did the target move more than tol (m) since start_pos (e.g. a lying cylinder rolling away)?"""
    return float(np.linalg.norm(np.asarray(env.object_pose(target["id"])[0]) - np.asarray(start_pos))) > tol


# ------------------------------------------------------------------------------------------------ expert
def nominal_still(qs, t, n=None, tol=None):
    """Has the nominal joint path qs (list of 5-angle poses) been still at step t for the last n steps?"""
    n = COMP_STILL_STEPS if n is None else n
    tol = COMP_STILL_DEG if tol is None else tol
    if t < n:
        return False
    q = np.asarray(qs[t], float)
    return all(float(np.max(np.abs(np.asarray(qs[t - k], float) - q))) <= tol for k in range(1, n + 1))


def comp_reference(qs, t):
    """The pose the readings are compared with by the sag correction. Still: the nominal pose. Moving (exp14):
    the nominal pose COMP_LAG_STEPS steps ago, so that the command delay and the servo tracking lag are not
    integrated (that wound the correction up into an overshoot), while the servo dead band, which leaves a
    joint several degrees short in one direction of motion, still is (frozen correction: the gripper dropped
    up to 6 mm below its path during the grasp shift)."""
    if nominal_still(qs, t):
        return np.asarray(qs[t], float)
    return np.asarray(qs[max(0, t - COMP_LAG_STEPS)], float)


def min_jerk(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def segment(q0, q1, duration, hz):
    n = max(2, int(round(duration * hz)))
    s = min_jerk(np.linspace(0, 1, n + 1)[1:])
    return [q0 + (q1 - q0) * v for v in s]


TRANSIT_LIFT = 0.003       # m, exp14: see ApproachExpert._clear
TRANSIT_GRACE = 8          # steps
RISE_DZ = 0.04             # m, exp14: via-point of a low start, the tip raised this much, axis kept
TRANSIT_BOARD_CLEAR = 0.008   # m, exp14: gripper links to the board along the transit (the transit is not
                              # sag-corrected, so a wider margin than the descent's GRIPPER_BOARD_CLEAR)


def rise_pose(env, q_deg5, dz):
    """The pose (deg, 5) with the tip dz higher, gripper axis and wrist roll kept; None if IK fails."""
    q = env.from_lerobot(np.r_[np.asarray(q_deg5, float)[:5], GRIPPER_CLOSED_PCT])[:5]
    local = tip_local(env)
    tip, _, rot = env.kin.points(q, local)
    a = rot @ env.kin.approach_local
    q5, ok, _, _ = env.kin.solve(np.asarray(tip) + np.array([0, 0, dz]), a, float(q[4]), [q], local)
    return env.to_lerobot(np.r_[q5, env.kin.gripper_closed])[:5] if ok else None


class ApproachExpert:
    """Plans the approach of one scene and produces its commands step by step (see the module docstring)."""

    PHASES = ["transit", "descend", "hold"]

    def __init__(self, env, rng, ready_deg5=None, long_hold_p=0.0, align_roll=False):
        self.env, self.rng, self.ready = env, rng, ready_deg5
        self.align_roll = align_roll             # exp14: jaws turned onto the object's faces (goal())
        self.long_hold_p = long_hold_p           # share of episodes with the long hold (HOLD_LONG_S)
        self.plan = None
        self.long_hold = False

    def reset(self, q_start_deg5, target, direct=False):
        """Plan from the current pose; False if the scene cannot be approached cleanly. direct=True (a
        near-goal start): one short correction straight to the goal (phase 'descend'), then the hold."""
        env, r = self.env, self.rng
        self.goal = goal(env, target, align_roll=self.align_roll)
        if self.goal is None:
            return False
        hz = env.cfg.control_hz
        q0 = np.asarray(q_start_deg5, float)
        others = [o for o in env.objects if o["id"] != target["id"]]
        self.long_hold = bool(r.random() < self.long_hold_p)
        hold_s = HOLD_LONG_S if self.long_hold else HOLD_S
        if direct:
            descend = segment(q0, self.goal["q"], r.uniform(*DIRECT_S), hz)
            if not self._clear(descend, others, True, None, target):
                return False
            hold = [self.goal["q"]] * int(round(r.uniform(*hold_s) * hz))
            self.plan = [(q, 1) for q in descend] + [(q, 2) for q in hold]
            self.t, self.corr = 0, np.zeros(5)
            return True
        speed = r.uniform(*SPEED_DEG_S)
        vias = [[]] + ([[self.ready]] if self.ready is not None else [])
        up = rise_pose(env, q0, RISE_DZ)                   # exp14: rise first (a low workspace start)
        if up is not None:
            vias += [[up]] + ([[up, self.ready]] if self.ready is not None else [])
        z0 = float(tip_of(env, q0)[2])
        for via in vias:
            pts = [q0] + via + [self.goal["q_pre"]]
            transit = []
            for a, b in zip(pts[:-1], pts[1:]):
                transit += segment(a, b, max(0.6, float(np.max(np.abs(b - a))) / speed), hz)
            descend = segment(self.goal["q_pre"], self.goal["q"], r.uniform(*DESCEND_S), hz)
            if self._clear(transit, others, False, z0, target) and self._clear(descend, others, True, None, target):
                break
        else:
            return False
        self.info_via = len(via)
        hold = [self.goal["q"]] * int(round(r.uniform(*hold_s) * hz))
        self.plan = [(q, 0) for q in transit] + [(q, 1) for q in descend] + [(q, 2) for q in hold]
        self.t, self.corr = 0, np.zeros(5)
        return True

    def _clear(self, path, others, final, z_start=None, target=None):
        """Nominal path: tip above the board (or not lower than where it started, for a rest pose that holds
        the jaws low), and above every other object it passes over. exp14, the real collision shapes at the
        nominal poses: the gripper links at least TRANSIT_BOARD_CLEAR (transit; or not closer than at the
        start) / GRIPPER_BOARD_CLEAR (descent) from the board, every link ARM_OTHER_CLEAR from the other
        objects and ARM_TARGET_CLEAR from the target (or not closer than at the start)."""
        z_min = MIN_TIP_Z if z_start is None else min(MIN_TIP_Z, z_start - 0.003)
        for q in path[::2]:
            tip = tip_of(self.env, q)
            if tip[2] < z_min:
                return False
            for o in others:
                c, verts = object_vertices(self.env, o)
                if np.linalg.norm(tip[:2] - c[:2]) < o["spec"]["radius"] + 0.02 and tip[2] < verts[:, 2].max() + OBJ_MARGIN:
                    return False
        idx = list(range(0, len(path), 2)) + ([len(path) - 1] if (len(path) - 1) % 2 else [])
        bodies = [o["id"] for o in others] + ([target["id"]] if target is not None else [])
        need = {o["id"]: ARM_OTHER_CLEAR for o in others}
        if target is not None:
            need[target["id"]] = ARM_TARGET_CLEAR
        with pose_probe(self.env) as p:
            p.set(path[0])
            d0 = p.board()
            board_min = GRIPPER_BOARD_CLEAR if final else min(TRANSIT_BOARD_CLEAR, d0 - 0.001)
            # exp14: a start clear of the board (not the folded rest pose) must gain TRANSIT_LIFT after the first
            # TRANSIT_GRACE steps: the transit is not sag-corrected, and a path that stays as low as it started
            # dragged the gripper along the board
            board_later = board_min if (final or d0 < 0.0005) else min(TRANSIT_BOARD_CLEAR, d0 + TRANSIT_LIFT)
            if not final:                                    # not closer than at the start
                need = {b: min(n, p.body(b) - 0.001) for b, n in need.items()}
            for k in idx:
                p.set(path[k])
                if p.board() < (board_min if k < TRANSIT_GRACE else board_later):
                    return False
                if any(p.body(b) < need[b] for b in bodies):
                    return False
        return True

    def done(self):
        return self.plan is None or self.t >= len(self.plan)

    def act(self, obs):
        """Command (6 values, LeRobot units) and phase index for this step."""
        q_nom, phase = self.plan[self.t]
        if phase >= 1:                                              # integral correction from the readings
            err = comp_reference([q for q, _ in self.plan], self.t) - np.asarray(obs["state"], float)[:5]
            err[0] = err[4] = 0.0                                   # pan and roll do not sag
            self.corr = np.clip(self.corr + COMP_GAIN * err, -COMP_LIMIT, COMP_LIMIT)
        self.t += 1
        return np.r_[q_nom + self.corr, GRIPPER_CLOSED_PCT], phase


def ready_pose(env):
    """The simulator's ready pose (deg, 5): jaws 8.5 cm high, 10 cm in front of the pan axis."""
    target = env.pan_axis + env.fwd * 0.10 + np.array([0, 0, 0.085])
    q5, ok, _, _ = env.kin.solve(target, -np.array([0, 0, 1.0]), 0.0, seeds=[env.q_home[:5], np.zeros(5)])
    return env.to_lerobot(np.r_[q5, env.kin.gripper_closed])[:5] if ok else None
