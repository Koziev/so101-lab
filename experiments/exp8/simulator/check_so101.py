"""
Sanity checks for the SO-101 PyBullet model. Run before generating data:

    python check_so101.py --out check_out [--gui]

Prints the joint map, detected gripper geometry and a reachability map, saves camera
images (home pose, +30 deg per-joint sweeps) and runs scripted grasp trials.
Compare the sweep images with your real arm commanded to the same angles in LeRobot
(degrees mode) to confirm signs and zero positions match.
"""
import argparse
import math
import os
import time

import cv2
import numpy as np

from so101_sim import scene_v7
from so101_sim.env import EnvConfig, SO101GraspEnv, DEFAULT_TEXTURE_DIR
from so101_sim.expert import ExpertConfig, ScriptedGraspExpert
from so101_sim.kinematics import ALL_JOINTS


def save(path, rgb):
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default="assets_so101/so101_sim.urdf")
    ap.add_argument("--out", default="check_out")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--top-camera", action="store_true")
    ap.add_argument("--camera-preset", default="v1", choices=["v1", "v4"])
    ap.add_argument("--scene-preset", default="v1", choices=["v1", "v7", "v8"])
    ap.add_argument("--texture-dir", default=DEFAULT_TEXTURE_DIR)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    size = (args.width, args.height)
    env = SO101GraspEnv(args.urdf, EnvConfig(top_camera=args.top_camera, camera_preset=args.camera_preset,
                                             front_size=size, wrist_size=size, scene_preset=args.scene_preset,
                                             texture_dir=args.texture_dir), gui=args.gui)
    kin, pb = env.kin, env.pb
    np.set_printoptions(precision=4, suppress=True)

    print("\n== joints (name, pybullet index, limits in deg) ==")
    for i, n in enumerate(ALL_JOINTS):
        print(f"  {n:14s} idx={env.q_idx[i]}  [{math.degrees(kin.lower[i]):7.1f}, {math.degrees(kin.upper[i]):7.1f}]")
    print("\n== detected geometry ==")
    print("  base lifted by (m):          ", -kin.base_bottom_z)
    print("  pan axis (world xy):         ", env.pan_axis[:2])
    print("  forward yaw (deg):           ", math.degrees(kin.forward_yaw))
    print("  TCP in gripper_link frame:   ", kin.tcp_local)
    print("  approach / hinge / closing:  ", kin.approach_local, kin.hinge_local, kin.closing_local)
    print("  gripper closed / open (deg): ", math.degrees(kin.gripper_closed), math.degrees(kin.gripper_open))
    for name, link in (("gripper_link", env.g_link), ("moving jaw", env.jaw_link)):
        print(f"  collision shapes on {name}: {len(pb.getCollisionShapeData(env.robot, link))} "
              "(1 means a single convex hull: run prepare_assets.py with VHACD)")
    print("  home pose (LeRobot units):   ", env.to_lerobot(env.q_home))

    print("\n== jaw gap probe (depth > 0 = beyond URDF tool point toward fingertips) ==")
    print(f"  fingertip depth: {kin.tip_depth * 1000:+.1f} mm, fixed->moving jaw direction: {kin.grasp_sign:+.0f} "
          "along closing axis")
    for d, u in list(zip(kin.probe_depths, kin.probe_fixed_u))[::3]:
        print(f"    depth {d * 1000:+6.1f} mm: fixed jaw inner face at {u * 1000:+6.1f} mm")
    print("  gap width vs opening:", ", ".join(f"{p}%={w * 1000:.0f}mm" for p, w in zip(kin.width_pcts, kin.widths)))
    for size in (0.024, 0.03, 0.035):
        print(f"  object {size * 1000:.0f} mm: grasp point (gripper frame) {kin.grasp_local(size)}, "
              f"min opening {kin.min_open_pct(size)}%")

    print("\n== reachability, top-down grasp at z=1.5 cm (rows: distance, cols: bearing) ==")
    bearings = list(range(-60, 61, 20))
    print("  r (m)  " + " ".join(f"{b:+4d}" for b in bearings))
    for rad in np.arange(0.08, 0.32, 0.02):
        row = []
        for b in bearings:
            th = math.radians(b)
            p = env.pan_axis + rad * (math.cos(th) * env.fwd + math.sin(th) * env.left) + np.array([0, 0, 0.015])
            _, ok, *_ = kin.solve(p, [0, 0, -1], 0.0, seeds=[env.q_home[:5], np.zeros(5)])
            row.append("  ok" if ok else "   .")
        print(f"  {rad:.2f}   " + " ".join(row))

    obs, info = env.reset(seed=0)
    for cam, img in obs["images"].items():
        save(os.path.join(args.out, f"home_{cam}.png"), img)
    if env.v7:  # a few more scenes, to see the randomisation
        for k in range(1, 5):
            o, _ = env.reset(seed=k)
            save(os.path.join(args.out, f"scene{k}_stereo.png"), np.concatenate([o["images"]["left"], o["images"]["right"]], 1))
        print("v7 textures:", env.tex_info, "| stereo camera (seed 4):",
              {k: round(v, 3) for k, v in env.stereo_cam.items() if isinstance(v, float)})
        obs, info = env.reset(seed=0)
    for i, n in enumerate(ALL_JOINTS[:5]):
        q = env.q_home.copy()
        q[i] = np.clip(q[i] + math.radians(30), kin.lower[i], kin.upper[i])
        env.set_joint_positions(q)
        save(os.path.join(args.out, f"sweep_{n}_plus30.png"),
             scene_v7.render_stereo(env)["left"] if env.v7 else env.render_front())
    # Close-up of the gripper from the side: red = URDF tool point, green/blue = grasp
    # points for 35 mm and 24 mm objects (where the object centre will be placed).
    q = env.q_home.copy()
    q[5] = kin.gripper_pct_to_rad(80)
    env.set_joint_positions(q)
    g_pos, g_rot = env.gripper_frame()
    markers = [env.add_marker(g_pos + g_rot @ kin.tcp_local, (1, 0, 0, 1)),
               env.add_marker(g_pos + g_rot @ kin.grasp_local(0.035), (0, 1, 0, 1)),
               env.add_marker(g_pos + g_rot @ kin.grasp_local(0.024), (0, 0, 1, 1))]
    tcp = g_pos + g_rot @ kin.tcp_local
    hinge, approach = g_rot @ kin.hinge_local, g_rot @ kin.approach_local
    for name, side in (("gripper_side_a", 1), ("gripper_side_b", -1)):
        save(os.path.join(args.out, f"{name}.png"),
             env._render(tcp + side * hinge * 0.18, tcp, -approach, 45, (640, 480)))
    for m in markers:
        pb.removeBody(m)
    print(f"\nimages saved to {args.out}/")

    print(f"\n== {args.trials} scripted grasp trials ==")
    expert = ScriptedGraspExpert(env, ExpertConfig(), np.random.default_rng(0))
    ok_n = skipped = table_hits = 0
    for k in range(args.trials):
        obs, info = env.reset(seed=1000 + k, render=False)
        if not expert.reset(obs, info):
            skipped += 1
            continue
        hit = squeezed = False
        while True:
            a, phase = expert.act(obs)
            if a is None:
                break
            obs = env.step(a, render=False)
            st = env.grasp_state(info["target"]["id"], info["target"]["pos"][2])
            hit |= st["gripper_table_contact"]
            if phase in ("close", "lift"):
                squeezed |= st["jaw_contact"] and st["fixed_contact"]
            if args.gui:
                time.sleep(1.0 / env.cfg.control_hz)
        gs = env.grasp_state(info["target"]["id"], info["target"]["pos"][2])
        ok_n += gs["grasped"]
        table_hits += hit
        print(f"  trial {k}: {info['target']['descr']:16s} tilt={expert.plan_info['tilt_deg']:2d}  "
              f"grasped={gs['grasped']}  lifted={gs['lifted']}  contacts(jaw/fixed)={gs['jaw_contact']}/"
              f"{gs['fixed_contact']}  both_jaws_touched_while_closing={squeezed}  table_contact={hit}  "
              f"open={expert.plan_info['open_pct']:.0f}%")
    tried = args.trials - skipped
    print(f"\nsuccess {ok_n}/{tried} planned trials, {skipped} unreachable, {table_hits} with gripper-table contact")
    env.close()


if __name__ == "__main__":
    main()
