"""
Does the simulator show the cell as the real camera sees it? Renders the simulated scene from the calibrated
camera pose (no camera randomisation, no objects, arm in the ready pose in which the real images were taken)
and puts it next to real left images.

    python real/compare_sim.py --setup setups/<name> --data data_real_v11 --samples 0,5,12

For every sample, <data>/compare/<n>.png holds four panels:
    real image                      | simulated image
    real image + simulated edges    | 50/50 blend
What to look at: the ARM and its BASE (outline, position, size) and the board's edges. They should coincide to
a few pixels. A simulated arm that is shifted, rotated or of a different size than the real one is a geometry
error in the simulator (camera pose, field of view, pan zero, base placement), which the model cannot know.
With the setup's stereo_background.png the simulated scene IS the real picture, so the blend should look like
one image; without it the textures, colours and lighting differ.
Also written: sim_example_*.png, complete simulated samples (left | right) as the dataset generator makes them,
two with the calibrated camera (T0) and two with a randomised camera (T1.5).
Run in an environment with PyBullet (the simulator); no arm, no camera, no model needed.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

READY_GRIPPER_PCT = 25.0


def ready_pose_sim(env):
    """As real/record_real.py: jaws 8.5 cm high, 10 cm in front of the pan axis (degrees, gripper in %)."""
    target = env.pan_axis + env.fwd * 0.10 + np.array([0, 0, 0.085])
    q5, ok, _, _ = env.kin.solve(target, -np.array([0, 0, 1.0]), 0.0, seeds=[env.q_home[:5], np.zeros(5)])
    if not ok:
        raise SystemExit("the simulator's ready pose could not be solved")
    return np.r_[env.to_lerobot(np.r_[q5, env.kin.gripper_closed])[:5], READY_GRIPPER_PCT]


def render_sim(setup_dir, urdf, seed=0):
    env = common.make_env(0.0, urdf=urdf, cell=common.cell_config(setup_dir))
    for s in range(seed, seed + 20):
        try:
            env.reset(seed=s, render=False)
            break
        except RuntimeError:
            continue
    else:
        raise SystemExit("the simulator could not build a scene")
    for o in env.objects:                                  # an empty board: only the cell itself is compared
        env.pb.removeBody(o["id"])
    env.objects = []
    env.set_joint_positions(env.from_lerobot(ready_pose_sim(env)))
    for _ in range(30):
        env.pb.stepSimulation()
    left = env.observe(render=True)["images"]["left"]
    cam = dict(env.stereo_cam)
    env.close()
    return cv2.cvtColor(np.asarray(left), cv2.COLOR_RGB2BGR), cam


def examples(setup_dir, urdf, out, scale, seeds):
    """A few complete simulated samples (objects, random ready pose), as the dataset generator makes them."""
    env = common.make_env(scale, urdf=urdf, cell=common.cell_config(setup_dir))
    for s in seeds:
        try:
            obs, _ = env.reset(seed=s, render=True)
        except RuntimeError:
            continue
        f = os.path.join(out, f"sim_example_T{scale:g}_{s}.png")
        cv2.imwrite(f, cv2.cvtColor(np.hstack([obs["images"]["left"], obs["images"]["right"]]), cv2.COLOR_RGB2BGR))
        print("wrote", f, "|", env.task["instruction"])
    env.close()


def label(img, text):
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True)
    ap.add_argument("--data", default="data_real_v11")
    ap.add_argument("--samples", default="0,1,2", help="indices of real samples (<data>/samples/<n>_left.png)")
    ap.add_argument("--image", default=None, help="instead of --samples: one real rectified 640x360 left image")
    ap.add_argument("--urdf", default=common.DEFAULT_URDF)
    args = ap.parse_args()

    sim, cam = render_sim(args.setup, args.urdf)
    print("simulated camera: " + ", ".join(f"{k} {cam[k]:.2f}" for k in ("distance_m", "height_m", "pitch_deg",
                                                                        "azimuth_deg", "yaw_deg", "roll_deg",
                                                                        "hfov_deg") if k in cam))
    edges = cv2.Canny(cv2.cvtColor(sim, cv2.COLOR_BGR2GRAY), 50, 140) > 0
    out = os.path.join(args.data, "compare")
    os.makedirs(out, exist_ok=True)
    files = [(os.path.splitext(os.path.basename(args.image))[0], args.image)] if args.image else \
        [(n.strip(), os.path.join(args.data, "samples", f"{n.strip()}_left.png")) for n in args.samples.split(",")]
    for name, f in files:
        real = cv2.imread(f)
        if real is None:
            print("cannot read", f)
            continue
        if real.shape[:2] != sim.shape[:2]:
            real = cv2.resize(real, (sim.shape[1], sim.shape[0]), interpolation=cv2.INTER_AREA)
        over = real.copy()
        over[edges] = (0, 0, 255)
        grid = np.vstack([np.hstack([label(real.copy(), f"real #{name}"), label(sim.copy(), "simulated")]),
                          np.hstack([label(over, "real + simulated edges (red)"),
                                     label(cv2.addWeighted(real, 0.5, sim, 0.5, 0), "blend")])])
        cv2.imwrite(os.path.join(out, f"{name}.png"), grid)
        print("wrote", os.path.join(out, f"{name}.png"))
    json.dump(cam, open(os.path.join(out, "sim_camera.json"), "w"), indent=1, default=float)
    examples(args.setup, args.urdf, out, 0.0, (1, 2))
    examples(args.setup, args.urdf, out, 1.5, (3, 4))


if __name__ == "__main__":
    main()
