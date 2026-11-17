"""
Forward kinematics of a serial arm read directly from its URDF, NumPy only (no PyBullet), for calibration.

    fk = UrdfChain("assets_so101/so101_sim.urdf", tip_link="gripper_link")
    T = fk.pose(q_rad)            # 4x4 pose of tip_link in the base frame, q in URDF radians (arm joints in order)

Joint angles in LeRobot degrees (degrees mode, 0 = middle of the range) are the URDF angles in degrees for the
arm joints of the SO-101 model used here; `deg_to_rad` converts. self_test() compares against the simulator's
PyBullet kinematics on the user's machine.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(axis, a):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def homog(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


class UrdfChain:
    def __init__(self, urdf_path, tip_link="gripper_link", base_link=None):
        root = ET.parse(urdf_path).getroot()
        joints = {}
        for j in root.findall("joint"):          # top-level joints only (<transmission> blocks contain <joint> too)
            if j.find("parent") is None or j.find("child") is None:
                continue
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0").split() if o is not None else "0 0 0".split())]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0").split() if o is not None else "0 0 0".split())]
            a = j.find("axis")
            lim = j.find("limit")
            joints[j.find("child").get("link")] = {
                "name": j.get("name"), "type": j.get("type"), "parent": j.find("parent").get("link"),
                "T": homog(rpy_matrix(*rpy), xyz),
                "axis": [float(v) for v in (a.get("xyz").split() if a is not None else "1 0 0".split())],
                "limit": (float(lim.get("lower", "-3.14")), float(lim.get("upper", "3.14"))) if lim is not None else None}
        chain, link = [], tip_link
        while link in joints and link != base_link:
            chain.append(joints[link])
            link = joints[link]["parent"]
        self.base_link = link
        self.chain = chain[::-1]                                       # base -> tip
        self.movable = [j for j in self.chain if j["type"] in ("revolute", "continuous", "prismatic")]
        self.names = [j["name"] for j in self.movable]
        self.lower = np.array([j["limit"][0] if j["limit"] else -math.pi for j in self.movable])
        self.upper = np.array([j["limit"][1] if j["limit"] else math.pi for j in self.movable])

    def pose(self, q, offsets=None):
        """4x4 pose of the tip link in the base frame for joint angles q (rad, one per movable joint)."""
        q = np.asarray(q, float)
        if offsets is not None:
            q = q + np.asarray(offsets, float)
        T, k = np.eye(4), 0
        for j in self.chain:
            T = T @ j["T"]
            if j["type"] in ("revolute", "continuous"):
                T = T @ homog(axis_angle(j["axis"], q[k]), np.zeros(3))
                k += 1
            elif j["type"] == "prismatic":
                T = T @ homog(np.eye(3), np.asarray(j["axis"]) * q[k])
                k += 1
        return T


def gravity_moments(chain, q):
    """Moment arm (m) of a load at the tip link's origin about every revolute joint's axis, signed:
    axis . ((p_tip - p_joint) x (0, 0, -1)). Proportional to the gravity moment each joint carries from the
    gripper end (gripper, camera, jaws)."""
    q = np.asarray(q, float)
    T, k, joints = np.eye(4), 0, []
    for j in chain.chain:
        T = T @ j["T"]
        if j["type"] in ("revolute", "continuous"):
            joints.append((T[:3, 3].copy(), T[:3, :3] @ (np.asarray(j["axis"], float) / np.linalg.norm(j["axis"]))))
            T = T @ homog(axis_angle(j["axis"], q[k]), np.zeros(3))
            k += 1
    tip = T[:3, 3]
    g = np.array([0, 0, -1.0])
    return np.array([float(ax @ np.cross(tip - p, g)) for p, ax in joints])


def deg_to_rad(q_deg):
    return np.radians(np.asarray(q_deg, float))


def self_test(urdf_path, n=20, seed=0):
    """Compare with the simulator's PyBullet kinematics (run on the machine where pybullet is installed)."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulator"))
    from so101_sim.env import EnvConfig, SO101GraspEnv
    env = SO101GraspEnv(urdf_path, EnvConfig(scene_preset="v1"))
    env.reset(seed=0, render=False)
    fk = UrdfChain(urdf_path, "gripper_link")
    rng = np.random.default_rng(seed)
    P, F = [], []
    for _ in range(n):
        q_deg = rng.uniform(-60, 60, 5)
        q_pb = env.from_lerobot(np.r_[q_deg, 50.0])[:5]
        env.kin.set_q(q_pb)
        p, R = env.kin.frame(env.kin.g_link)
        P.append(homog(np.asarray(R, float), np.asarray(p, float)))
        F.append(fk.pose(deg_to_rad(q_deg)[: len(fk.names)]))
    # compare RELATIVE poses (independent of how the simulator places the base in its world)
    worst_p, worst_r = 0.0, 0.0
    for Pi, Fi in zip(P[1:], F[1:]):
        a, b = np.linalg.inv(P[0]) @ Pi, np.linalg.inv(F[0]) @ Fi
        worst_p = max(worst_p, float(np.linalg.norm(a[:3, 3] - b[:3, 3])))
        dR = a[:3, :3].T @ b[:3, :3]
        worst_r = max(worst_r, math.degrees(math.acos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    env.close()
    print(f"URDF chain {fk.names}: worst difference to PyBullet over {n} random poses: "
          f"position {1000 * worst_p:.3f} mm, rotation {worst_r:.4f} deg")
    return worst_p < 1e-4 and worst_r < 0.01


if __name__ == "__main__":
    import sys
    ok = self_test(sys.argv[1] if len(sys.argv) > 1 else "assets_so101/so101_sim.urdf")
    print("self-test", "PASSED" if ok else "FAILED: the calibration's kinematics do not match the simulator's")
