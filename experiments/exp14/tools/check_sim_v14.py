"""
Check of the v14 simulator changes: the wrist camera on the arm, and the arm's brightness against a real frame.

    python tools/check_sim_v14.py --setup setups/2026-11_12v_stereo28mm_wrist --out outputs/check_v14
    # with a real stereo frame and the joint readings it was taken at (real convention, 5 joints + gripper %):
    python tools/check_sim_v14.py --setup ... --real-left real_left.png --real-q "1.2,-98.0,95.3,70.1,-3.0,1.0"

Writes into --out:
  gripper_closeup.png      the gripper with the camera assembly from 6 directions (plain PyBullet colours):
                           board behind the lens, frame on the lens side, arm of the mount reaching back to the
                           gripper, base and nut at its end
  stereo_<pose>.png        left | right image of the stereo camera (calibrated pose), v11 above, v14 below,
                           for the rest pose, the ready pose and --real-q
  scene_v14_<n>.png        four v14 scenes with 3-4 objects of the new set (sizes, colours, brightness)
  overlay.png              (with --real-left/--real-q) the real frame with the edges of the v14 render (red),
                           and a 50/50 blend; the camera board's outline should match the real one
and prints:
  * the chosen orientation of the camera head (and the other candidates: if the overlay shows the mount on the
    wrong side, rerun with --roll/--flip, then set EnvConfig.wrist_cam_orientation to the same values);
  * the camera assembly's distance to the board and contacts in every pose (it must not touch the board at rest);
  * with a real frame: brightness percentiles and mean RGB of the arm's pixels, real against v11 and v14 renders
    (20 random looks each), measured inside the simulated arm's outline (eroded by 2 px).
Runs in the simulator environment (PyBullet), no arm or model needed.
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import approach as ap  # noqa: E402
import common  # noqa: E402


def make(setup, sim, roll=None, flip=None):
    kw = {"wrist_cam_orientation": (roll, flip)} if sim == "v14" else {}
    env = common.make_env(0.0, cell=common.cell_config(setup), sim=sim, **kw)
    for s in range(20):
        try:
            env.reset(seed=s, render=False)
            break
        except RuntimeError:
            continue
    for o in env.objects:                             # an empty board
        env.pb.removeBody(o["id"])
    env.objects = []
    return env


def set_pose(env, q5_sim_deg, grip_pct=ap.GRIPPER_CLOSED_PCT, steps=30):
    env.set_joint_positions(env.from_lerobot(np.r_[np.asarray(q5_sim_deg, float)[:5], grip_pct]))
    for _ in range(steps):
        env.pb.stepSimulation()


def camera_report(env, name):
    links = getattr(env, "cam_links", ())
    if not links:
        return
    table = getattr(env, "table", None)
    d = min((c[8] for l in links for c in env.pb.getClosestPoints(env.robot, table, 0.10, linkIndexA=l)),
            default=float("inf")) if table is not None else float("nan")
    touch = [l for l in links if table is not None and env.pb.getContactPoints(env.robot, table, linkIndexA=l)]
    print(f"  {name:10s} camera assembly {d * 1000:6.1f} mm from the board" + ("   TOUCHES THE BOARD" if touch else ""))


def closeup(env, size=320):
    import pybullet
    s = env.pb.getLinkState(env.robot, env.g_link, computeForwardKinematics=True)
    target = np.array(s[4]) + np.array(env.pb.getMatrixFromQuaternion(s[5])).reshape(3, 3) @ [0, 0.035, -0.03]
    tiles = []
    for yaw, pitch in ((0, 20), (60, 20), (120, 20), (180, 20), (240, 20), (300, 20)):
        y, p = math.radians(yaw), math.radians(pitch)
        eye = target + 0.16 * np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
        view = env.pb.computeViewMatrix(eye.tolist(), target.tolist(), [0, 0, 1])
        proj = env.pb.computeProjectionMatrixFOV(40, 1.0, 0.01, 1.0)
        img = env.pb.getCameraImage(size, size, view, proj, renderer=pybullet.ER_TINY_RENDERER)
        rgb = np.reshape(np.asarray(img[2], np.uint8), (size, size, 4))[:, :, :3]
        t = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        cv2.putText(t, f"azimuth {yaw}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        tiles.append(t)
    return np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])


def left_mask(env):
    """Simulated arm's pixels in the left image (white parts; the camera board excluded)."""
    from so101_sim import scene_real
    cam = env.stereo_cam
    center, fwd, right, up = (np.array(cam[k]) for k in ("center", "fwd", "right", "up"))
    rgb, seg, view, proj, link, depth = scene_real._raw(env, center - right * cam["baseline"] / 2, fwd, up,
                                                        cam["vfov_deg"])
    m = (seg == env.robot) & ~np.isin(link, getattr(env, "cam_dark_links", ()))
    return cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def stats(img_rgb, mask):
    px = img_rgb[mask].astype(np.float32)
    lum = px.mean(axis=1)
    return np.percentile(lum, [50, 65, 90, 99]), px.mean(axis=0)


def render_left_looks(env, q5, grip, n=20):
    out = []
    for k in range(n):
        env._look_for = None                              # a new look per render
        env.rng = np.random.default_rng(1000 + k)
        set_pose(env, q5, grip, steps=1)
        out.append(env.observe(render=True)["images"]["left"])
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--setup", required=True)
    ap_.add_argument("--out", default="outputs/check_v14")
    ap_.add_argument("--real-left", default=None, help="real rectified left image, 640x360 (or a 1280x360 pair)")
    ap_.add_argument("--real-q", default=None, help="joint readings of that frame, real convention: 5 deg + gripper %%")
    ap_.add_argument("--roll", type=int, default=None, help="override: head roll 0..3 (x 90 deg)")
    ap_.add_argument("--flip", type=int, default=None, help="override: head flip 0/1")
    args = ap_.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("building the v14 simulator ...")
    v14 = make(args.setup, "v14", args.roll, args.flip)
    v11 = make(args.setup, "v11")
    poses = {"rest": (ap.rest_pose(args.setup), ap.GRIPPER_CLOSED_PCT), "ready": (ap.ready_pose(v14), ap.GRIPPER_CLOSED_PCT)}
    if args.real_q:
        q = [float(v) for v in args.real_q.split(",")]
        poses["real"] = (ap.rest_pose(args.setup, real=q[:5]), q[5] if len(q) > 5 else ap.GRIPPER_CLOSED_PCT)

    print("camera assembly against the board:")
    for name, (q5, g) in poses.items():
        set_pose(v14, q5, g)
        camera_report(v14, name)
    set_pose(v14, poses["ready"][0], 30.0)
    cv2.imwrite(os.path.join(args.out, "gripper_closeup.png"), closeup(v14))
    for name, (q5, g) in poses.items():
        rows = []
        for env, tag in ((v11, "v11"), (v14, "v14")):
            set_pose(env, q5, g)
            im = env.observe(render=True)["images"]
            row = cv2.cvtColor(np.hstack([im["left"], im["right"]]), cv2.COLOR_RGB2BGR)
            cv2.putText(row, f"{tag} {name}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            rows.append(row)
        cv2.imwrite(os.path.join(args.out, f"stereo_{name}.png"), np.vstack(rows))
    ex = common.make_env(0.0, cell=common.cell_config(args.setup), sim="v14",
                         task_kwargs={"n_objects": (3, 4), "white_weight": 1 / 6},
                         wrist_cam_orientation=(args.roll, args.flip))
    for seed in (1, 2, 3, 4):                         # v14 scenes with the new object set
        try:
            obs, _ = ex.reset(seed=seed, render=True)
        except RuntimeError:
            continue
        img = cv2.cvtColor(np.hstack([obs["images"]["left"], obs["images"]["right"]]), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.out, f"scene_v14_{seed}.png"), img)
        print(f"  scene_v14_{seed}.png: " + ", ".join(o["descr"] for o in ex.objects))
    print("wrote", args.out)

    if args.real_left and "real" in poses:
        real = cv2.imread(args.real_left)
        if real is None:
            raise SystemExit(f"cannot read {args.real_left}")
        w, h = v14.cfg.front_size
        if real.shape[1] == 2 * w:
            real = real[:, :w]
        real = cv2.resize(real, (w, h), interpolation=cv2.INTER_AREA) if real.shape[:2] != (h, w) else real
        real_rgb = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)
        q5, g = poses["real"]
        set_pose(v14, q5, g)
        sim = v14.observe(render=True)["images"]["left"]
        edges = cv2.Canny(cv2.cvtColor(sim, cv2.COLOR_RGB2GRAY), 50, 140) > 0
        over = real.copy()
        over[edges] = (0, 0, 255)
        blend = cv2.addWeighted(real, 0.5, cv2.cvtColor(sim, cv2.COLOR_RGB2BGR), 0.5, 0)
        cv2.imwrite(os.path.join(args.out, "overlay.png"), np.hstack([over, blend]))
        mask = left_mask(v14)
        print(f"\narm pixels inside the simulated outline: {int(mask.sum())}")
        print(f"{'':14s} {'p50':>6s} {'p65':>6s} {'p90':>6s} {'p99':>6s}   mean RGB")
        p, c = stats(real_rgb, mask)
        print(f"{'real':14s} " + " ".join(f"{v:6.0f}" for v in p) + "   " + " ".join(f"{v:4.0f}" for v in c))
        for env, tag in ((v11, "sim v11"), (v14, "sim v14")):
            ps, cs = zip(*(stats(im, mask) for im in render_left_looks(env, q5, g)))
            ps, cs = np.array(ps), np.array(cs)
            print(f"{tag + ' (median)':14s} " + " ".join(f"{v:6.0f}" for v in np.median(ps, 0)) + "   "
                  + " ".join(f"{v:4.0f}" for v in np.median(cs, 0)))
            print(f"{tag + ' (range)':14s} " + " ".join(f"{a:3.0f}-{b:<3.0f}"[:6] for a, b in zip(ps.min(0), ps.max(0))))
        print("If real and v14 differ, adjust V14_ARM_WHITE / V14_ARM_TINT in so101_sim/scene_real.py.")


if __name__ == "__main__":
    main()
