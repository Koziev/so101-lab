"""
The wrist camera and its printed mount on the simulated arm (experiment 14).

The real gripper carries a 30 x 30 mm camera board on a printed bracket (STL files in simulator/assets/wrist_camera:
camera_head = frame + arm, camera_base = L-bracket screwed to the gripper, camera_screw = thumb nut). From the
stereo camera it hides most of the fixed jaw. Here it becomes two extra links of the PHYSICAL robot, fixed to
gripper_link, written into a copy of the URDF:

    wrist_cam_mount   printed parts (white PLA, recoloured with the arm)    head + base meshes
    wrist_cam_board   camera board, lens, thumb nut (black)                 boxes / cylinder / nut mesh

Both are rendered, take part in contact detection (they are links of env.robot) and add their mass to the
gripper. The kinematics twin (env.kin: IK, jaw tips, goal geometry) keeps the original URDF, so the definition of
the tip and of the goal does not change.

Where the parts are:
  * the camera's optical centre and axes in the gripper_link frame come from the hand-eye calibration
    (setup's wrist_mount.json, OpenCV convention: x right, y down, z along the view);
  * board (30 x 30 x 5 mm) behind the lens; lens barrel (5 mm) centred on the board, its front LENS_FRONT_M
    ahead of the optical centre; the frame (camera_head plate, 4 mm) on the lens side of the board, the barrel
    through its opening;
  * the board's edges are parallel to the image axes, so the head's rotation about the view axis is a multiple
    of 90 deg; with the two ways the frame can face, 8 orientations are possible. The one whose bolt (end of the
    head's arm) lies closest to the gripper body is used, unless head_roll / head_flip are given;
  * camera_base: its slotted tab against the head's arm, on the side of the gripper, bolt at the far end of
    the slot (away from the plate, as on the real arm), its plate pointing towards the gripper body (the real pivot is adjustable, so this part is approximate);
  * the thumb nut on the other side of the arm.
Collision shapes are boxes / cylinders (a mesh would collide as its convex hull, which fills the frame's corner).

Geometry of the STL files (mm, their own coordinates), measured from the files:
"""
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

ASSET_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "wrist_camera"))
MM = 0.001

HEAD_LENS_CENTRE = np.array([15.65, 16.6]) * MM       # centre of the frame's opening = board centre
HEAD_PLATE = (np.array([0.0, 0.0, 0.0]), np.array([31.3, 32.96, 4.0]))       # frame plate box (min, max)
HEAD_ARM = (np.array([27.3, 35.5, 0.0]), np.array([31.3, 58.05, 7.3]))       # arm box
HEAD_BOLT = np.array([29.3, 54.4, 3.65]) * MM          # bolt hole centre; the bolt runs along the head's x
BASE_TAB = (np.array([15.8, 0.0, 0.0]), np.array([19.3, 7.3, 27.0]))         # slotted tab
BASE_PLATE = (np.array([0.0, 0.0, 0.0]), np.array([15.8, 7.3, 1.8]))         # plate screwed to the gripper
BASE_SLOT_Z = (4.35, 25.22)                           # slot ends (mm, base z); slot 3.3 mm wide
BOLT_RADIUS = 1.65
# where the bolt sits in the slot: at the slot's far end, away from the plate (photo of 22 Nov 2026)
BASE_BOLT = np.array([17.55, 3.65, BASE_SLOT_Z[1] - BOLT_RADIUS]) * MM   # the bolt runs along the base's x
NUT_CENTRE = np.array([5.685, 5.685, 2.5]) * MM        # thumb nut, axis along its z, 5 mm thick
NUT_RADIUS, NUT_THICK = 5.685 * MM, 5.0 * MM

BOARD_SIZE = (30 * MM, 30 * MM, 5 * MM)                # board with its components
LENS_RADIUS, LENS_LENGTH = 6.0 * MM, 5.0 * MM
LENS_FRONT_M = 1.0 * MM                                # barrel front ahead of the optical centre
FRAME_T = 4.0 * MM

WHITE = (0.95, 0.95, 0.95, 1.0)
BLACK = (0.06, 0.06, 0.06, 1.0)
MOUNT_LINK, BOARD_LINK = "wrist_cam_mount", "wrist_cam_board"


# ------------------------------------------------------------------------------------------------ geometry
def load_mount(setup_dir):
    """4x4 camera -> gripper_link transform from the setup's wrist_mount.json (None if missing)."""
    f = os.path.join(setup_dir, "wrist_mount.json")
    if not os.path.exists(f):
        return None
    m = json.load(open(f))
    T = np.eye(4)
    T[:3, :3] = np.asarray(m["R_cam2gripper"], float)
    T[:3, 3] = np.asarray(m["t_cam2gripper_m"], float)
    return T


def _T(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def _rz(k):
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def head_in_camera(roll, flip):
    """Head STL frame (metres) -> camera frame. flip 0: the head's z=0 face towards the scene (z_cam = -z_head);
    flip 1: its z=4 mm face towards the scene. The plate always occupies z_cam in [-FRAME_T, 0]."""
    if flip:
        M, z_ref = np.eye(3), FRAME_T
    else:
        M, z_ref = np.diag([1.0, -1.0, -1.0]), 0.0
    R = _rz(roll) @ M
    return _T(R, -R @ np.r_[HEAD_LENS_CENTRE, z_ref])


def assembly(T_cam, roll, flip, body_centre):
    """Parts of the camera assembly as (link, kind, size or mesh, T in gripper_link frame, colour).
    body_centre: a point inside the gripper body (gripper_link frame), the side the base and its plate face."""
    parts = []
    T_head = T_cam @ head_in_camera(roll, flip)
    parts.append((MOUNT_LINK, "mesh", "camera_head.stl", T_head, WHITE))
    for lo, hi in (HEAD_PLATE, HEAD_ARM):
        parts.append((MOUNT_LINK, "box_col", (hi - lo) * MM, T_head @ _T(t=(lo + hi) / 2 * MM), None))
    # board and lens, in the camera frame
    z_board = -FRAME_T - BOARD_SIZE[2] / 2
    parts.append((BOARD_LINK, "box", BOARD_SIZE, T_cam @ _T(t=[0, 0, z_board]), BLACK))
    parts.append((BOARD_LINK, "cylinder", (LENS_RADIUS, LENS_LENGTH),
                  T_cam @ _T(t=[0, 0, LENS_FRONT_M - LENS_LENGTH / 2]), BLACK))
    # bolt, base and nut
    B = (T_head @ np.r_[HEAD_BOLT, 1])[:3]
    a = T_head[:3, 0]                                                   # bolt axis (head's x)
    s = 1.0 if np.dot(body_centre - B, a) >= 0 else -1.0                # the gripper's side of the arm
    arm_half = (HEAD_ARM[1][0] - HEAD_ARM[0][0]) / 2 * MM
    tab_half = (BASE_TAB[1][0] - BASE_TAB[0][0]) / 2 * MM
    x_b = -s * a                                                        # the base's plate extends along -x_b
    away = B - body_centre
    z_b = away - np.dot(away, a) * a
    z_b = z_b / (np.linalg.norm(z_b) + 1e-12)                           # plate (z=0 end) towards the body
    R_b = np.stack([x_b, np.cross(z_b, x_b), z_b], axis=1)
    tab_centre = B + s * a * (arm_half + tab_half)
    T_base = _T(R_b, tab_centre - R_b @ BASE_BOLT)
    parts.append((MOUNT_LINK, "mesh", "camera_base.stl", T_base, WHITE))
    for lo, hi in (BASE_TAB, BASE_PLATE):
        parts.append((MOUNT_LINK, "box_col", (hi - lo) * MM, T_base @ _T(t=(lo + hi) / 2 * MM), None))
    z_n = -s * a
    x_n = np.cross(z_n, [0.0, 0.0, 1.0] if abs(z_n[2]) < 0.9 else [1.0, 0.0, 0.0])
    x_n /= np.linalg.norm(x_n)
    R_n = np.stack([x_n, np.cross(z_n, x_n), z_n], axis=1)
    nut_c = B - s * a * (arm_half + NUT_THICK / 2)
    T_nut = _T(R_n, nut_c - R_n @ NUT_CENTRE)
    parts.append((BOARD_LINK, "mesh", "camera_screw.stl", T_nut, BLACK))
    parts.append((BOARD_LINK, "cylinder_col", (NUT_RADIUS, NUT_THICK), _T(R_n, nut_c), None))
    return parts, {"bolt": B, "side": s}


def choose_orientation(T_cam, body_centre, roll=None, flip=None):
    """(roll, flip) with the bolt closest to the gripper body; given values are kept. Also returns the table
    of all candidates (roll, flip, distance in mm) for the report."""
    table = []
    for f in (0, 1):
        for k in range(4):
            B = (T_cam @ head_in_camera(k, f) @ np.r_[HEAD_BOLT, 1])[:3]
            table.append((k, f, float(np.linalg.norm(B - body_centre)) * 1000))
    ok = [r for r in table if (roll is None or r[0] == roll) and (flip is None or r[1] == flip)]
    best = min(ok, key=lambda r: r[2])
    return best[0], best[1], table


# ------------------------------------------------------------------------------------------------ URDF
def _rpy(R):
    """URDF rpy (R = Rz(y) Ry(p) Rx(r))."""
    p = math.asin(-max(-1.0, min(1.0, R[2, 0])))
    if abs(math.cos(p)) > 1e-6:
        r = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(R[1, 0], R[0, 0])
    else:
        r, y = math.atan2(-R[1, 2], R[1, 1]), 0.0
    return r, p, y


def _origin(T):
    r, p, y = _rpy(T[:3, :3])
    return {"xyz": " ".join(f"{v:.6f}" for v in T[:3, 3]), "rpy": f"{r:.6f} {p:.6f} {y:.6f}"}


def _geometry(parent, kind, size):
    g = ET.SubElement(parent, "geometry")
    if kind == "mesh":
        ET.SubElement(g, "mesh", filename=os.path.join(ASSET_DIR, size), scale="0.001 0.001 0.001")
    elif kind.startswith("box"):
        ET.SubElement(g, "box", size=" ".join(f"{v:.6f}" for v in size))
    else:
        ET.SubElement(g, "cylinder", radius=f"{size[0]:.6f}", length=f"{size[1]:.6f}")


def write_urdf(src_urdf, parts, parent_link="gripper_link", mass_kg=(0.008, 0.006), mass_at=(0.0, 0.0, 0.0)):
    """Copy of src_urdf with the two camera links; written next to it (its relative mesh paths stay valid),
    named by a hash of the content. The parts' mass sits at mass_at (gripper_link frame). Returns the path."""
    tree = ET.parse(src_urdf)
    robot = tree.getroot()
    for (name, m), colour in zip(((MOUNT_LINK, mass_kg[0]), (BOARD_LINK, mass_kg[1])), (WHITE, BLACK)):
        link = ET.SubElement(robot, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=" ".join(f"{v:.6f}" for v in mass_at), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=f"{m}")
        ET.SubElement(inertial, "inertia", ixx="1e-6", ixy="0", ixz="0", iyy="1e-6", iyz="0", izz="1e-6")
        for lk, kind, size, T, col in parts:
            if lk != name:
                continue
            if not kind.endswith("_col"):
                v = ET.SubElement(link, "visual")
                ET.SubElement(v, "origin", **_origin(T))
                _geometry(v, kind, size)
                mat = ET.SubElement(v, "material", name=f"{name}_{'white' if col == WHITE else 'black'}")
                ET.SubElement(mat, "color", rgba=" ".join(f"{c:g}" for c in col))
            if kind != "mesh":
                c = ET.SubElement(link, "collision")
                ET.SubElement(c, "origin", **_origin(T))
                _geometry(c, kind.replace("_col", ""), size)
        joint = ET.SubElement(robot, "joint", name=f"{name}_joint", type="fixed")
        ET.SubElement(joint, "parent", link=parent_link)
        ET.SubElement(joint, "child", link=name)
        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
    text = ET.tostring(robot, encoding="unicode")
    tag = hashlib.sha1(text.encode()).hexdigest()[:10]
    out = os.path.join(os.path.dirname(os.path.abspath(src_urdf)),
                       os.path.splitext(os.path.basename(src_urdf))[0] + f"_wristcam_{tag}.urdf")
    if not os.path.exists(out):
        tree.write(out)
    return out


def gripper_body_centre(kin):
    """Centre of the gripper_link's bounding box, in its own frame (the side the camera base faces)."""
    q = np.zeros(6)
    q[5] = kin.gripper_closed
    kin.set_q(q)
    lo, hi = (np.array(v) for v in kin.pb.getAABB(kin.body, kin.g_link))
    pos, rot = kin.frame(kin.g_link)
    return rot.T @ ((lo + hi) / 2 - pos)


def build(src_urdf, setup_dir, kin, roll=None, flip=None, verbose=True):
    """Augmented URDF for the physical robot, or None if the setup has no wrist_mount.json."""
    T_cam = load_mount(setup_dir)
    if T_cam is None:
        return None
    G = gripper_body_centre(kin)
    roll, flip, table = choose_orientation(T_cam, G, roll, flip)
    parts, info = assembly(T_cam, roll, flip, G)
    path = write_urdf(src_urdf, parts, mass_at=T_cam[:3, 3])
    if verbose:
        d = next(r[2] for r in table if r[0] == roll and r[1] == flip)
        print(f"wrist camera: head roll {roll * 90} deg, flip {flip} (bolt {d:.0f} mm "
              f"from the gripper body centre; candidates: "
              + ", ".join(f"{k * 90}/{f}: {d:.0f}" for k, f, d in table) + f") -> {os.path.basename(path)}")
    return path
