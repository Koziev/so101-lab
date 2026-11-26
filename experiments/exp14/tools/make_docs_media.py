"""
Media for the exp11 lab-log entry (docs/assets/exp11/).

    # one short clip per task type: a scene built around its prompt, approached by the label-making expert
    python tools/make_docs_media.py tasks --setup setups/<name> --out docs/assets/exp11

    # real stereo frames next to the simulator's rendering of the same cell (arm in the ready pose in which the
    # real images were taken, the target object placed at the recorded grasp position)
    python tools/make_docs_media.py compare --setup setups/<name> --real data_real_v11b --samples 3,12,20,41 \
        --out docs/assets/exp11 --tip-paint-mm 0

Clips: task_<type>.mp4 (H.264 if ffmpeg is installed) and task_<type>.png (a still). The expert is the scripted
closed-loop controller that makes the training labels (closed_loop.Expert), with ideal servos; the arm motion is
kinematic, so the jaws close on the object but it is not lifted.
Comparisons: real_vs_sim_<n>.png = real left image | simulated left image | 50/50 blend.
Real sets recorded before 16 Nov show the arm without the painted jaw tips: use --tip-paint-mm 0 for them.
Run in the simulator environment (PyBullet); no model, no arm, no camera.
"""
import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "real"))
import closed_loop as cl  # noqa: E402
import common  # noqa: E402

TASK_TYPES = ("full", "colour", "shape", "superlative")


# ------------------------------------------------------------------------------------------------ helpers
def caption(img, lines, top=True):
    """Text lines on a dark band at the top (RGB image)."""
    img = img.copy()
    h, w = img.shape[:2]
    band = 22 * len(lines) + 8
    y0 = 0 if top else h - band
    img[y0:y0 + band] = (img[y0:y0 + band] * 0.35).astype(np.uint8)
    for k, t in enumerate(lines):
        cv2.putText(img, t, (10, y0 + 20 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def write_video(frames_rgb, path, fps):
    """mp4 playable in browsers (H.264 via ffmpeg if available, else OpenCV's mp4v)."""
    h, w = frames_rgb[0].shape[:2]
    tmp = path + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames_rgb:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], check=True)
        os.remove(tmp)
    else:
        os.replace(tmp, path)
        print("  (no ffmpeg: written with OpenCV's mp4v codec, which some browsers do not play)")


def render_left(env, q4, grip_pct):
    env.set_joint_positions(env.from_lerobot(np.r_[np.asarray(q4, float), 0.0, grip_pct]))
    return env.observe(render=True)["images"]["left"]


# ------------------------------------------------------------------------------------------------ task clips
def task_clip(args, ttype):
    tk = {"weights": {ttype: 1.0}, "white_weight": 0.5, "margin": 1.35, "n_objects": (3, 4)}
    env = common.make_env(0.0, cell=common.cell_config(args.setup), task_kwargs=tk, tip_paint_mm=args.tip_paint_mm)
    for seed in range(args.seed, args.seed + 200):
        try:
            obs, _ = env.reset(seed=seed, render=False)
        except RuntimeError:
            continue
        ex = cl.Expert(env, env.target_point(env.objects[0]))
        if ex.ok:
            break
    else:
        raise SystemExit(f"no usable '{ttype}' scene")
    instruction = env.task["instruction"]
    lines = [f"task: {ttype}", f'prompt: "{instruction}"']
    q = np.asarray(obs["state"], float)[:4]
    grip = 85.0
    path = [q.copy()]
    for _ in range(args.max_steps):
        delta, g, _ = ex.action(q)
        if g == "close":
            break
        q = q + np.array([cl.one_decimal(v) for v in delta])
        path.append(q.copy())
    frames = []
    still = None
    first = caption(render_left(env, path[0], grip), lines)
    frames += [first] * int(args.fps * 1.5)                       # read the prompt
    for a, b in zip(path[:-1], path[1:]):
        for t in np.linspace(0, 1, args.substeps, endpoint=False)[1:].tolist() + [1.0]:
            frames.append(caption(render_left(env, a + (b - a) * t, grip), lines))
    size = 2 * float(env.objects[0]["spec"]["half_height"])
    k = env.kin
    close_pct = float(np.interp(size * 0.95, np.sort(k.widths), np.asarray(k.width_pcts)[np.argsort(k.widths)])) \
        if len(getattr(k, "widths", [])) else 30.0
    for gp in np.linspace(grip, close_pct, 8):
        frames.append(caption(render_left(env, path[-1], gp), lines))
        still = frames[-1]
    frames += [still] * int(args.fps * 1.5)
    env.close()
    out = os.path.join(args.out, f"task_{ttype}.mp4")
    write_video(frames, out, args.fps)
    cv2.imwrite(os.path.join(args.out, f"task_{ttype}.png"), cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_RGB2BGR))
    print(f"wrote {out} ({len(frames)} frames, {len(path) - 1} expert steps) | {instruction}")


# ------------------------------------------------------------------------------------------------ real vs sim
def compare(args):
    import pybullet
    import tasks
    from compare_sim import ready_pose_sim
    from so101_sim import objects_v11 as ob
    env = common.make_env(0.0, cell=common.cell_config(args.setup), tip_paint_mm=args.tip_paint_mm)
    for s in range(20):
        try:
            env.reset(seed=s, render=False)
            break
        except RuntimeError:
            continue
    q_ready = ready_pose_sim(env)
    for n in (v.strip() for v in args.samples.split(",")):
        f = os.path.join(args.real, "samples", f"{n}.json")
        real = cv2.imread(os.path.join(args.real, "samples", f"{n}_left.png"))
        if real is None or not os.path.exists(f):
            print("missing sample", n)
            continue
        m = json.load(open(f))
        for o in env.objects:
            env.pb.removeBody(o["id"])
        env.objects = []
        rng = np.random.default_rng(int(n))
        spec = ob.sample_object(rng, shapes=[m["target_shape"]], white_weight=0.0, lying_prob=0.0)
        spec = tasks.recolour(spec, m["target_colour"], rng)
        ref = np.asarray(m["ref_point"], float)
        body = ob.create_body(pybullet, spec, ref[:2].tolist(), float(rng.uniform(-math.pi, math.pi)), env.pb._client)
        env.objects = [{"id": body, "descr": spec["label"], "shape": spec["shape"], "color": spec["colour"],
                        "size": float(2 * spec["half_height"]), "spec": spec}]
        env.set_joint_positions(env.from_lerobot(q_ready))
        for _ in range(30):
            env.pb.stepSimulation()
        sim = cv2.cvtColor(env.observe(render=True)["images"]["left"], cv2.COLOR_RGB2BGR)
        if real.shape[:2] != sim.shape[:2]:
            real = cv2.resize(real, (sim.shape[1], sim.shape[0]), interpolation=cv2.INTER_AREA)
        lab = lambda im, t: caption(im[..., ::-1], [t])[..., ::-1]
        row = np.hstack([lab(real, f"real, sample {n}: {m['instruction']}"),
                         lab(sim, "simulator (target placed at the recorded grasp point)"),
                         lab(cv2.addWeighted(real, 0.5, sim, 0.5, 0), "blend")])
        out = os.path.join(args.out, f"real_vs_sim_{n}.png")
        cv2.imwrite(out, row)
        print("wrote", out)
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("tasks", "compare"))
    ap.add_argument("--setup", required=True)
    ap.add_argument("--out", default="docs/assets/exp11")
    ap.add_argument("--tip-paint-mm", type=float, default=20.0)
    ap.add_argument("--types", default=",".join(TASK_TYPES))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--substeps", type=int, default=4, help="rendered frames per expert step")
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--real", default="data_real_v11b", help="compare: folder of a real recording")
    ap.add_argument("--samples", default="3,12,20,41", help="compare: sample indices")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.what == "tasks":
        for t in args.types.split(","):
            task_clip(args, t.strip())
    else:
        compare(args)


if __name__ == "__main__":
    main()
