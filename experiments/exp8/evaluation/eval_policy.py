"""
Closed-loop evaluation in the PyBullet simulator.

    # sanity check of the harness itself: the scripted expert on the same scenes
    python eval_policy.py --policy expert

    # a fine-tuned LeRobot policy (checkpoint directory containing config + weights)
    python eval_policy.py --policy outputs/train/smolvla_sim_v1/checkpoints/last/pretrained_model \
        --new-scenes 100 --video-dir outputs/eval_videos

Scenes:
  * held-out seeds from eval_seeds.json: the exact scenes of the held-out dataset
    episodes (never used for training), so the policy can be compared with the expert;
  * --new-scenes N: additional fresh seeds from a range the generator never uses.

An episode succeeds when the target is held between both jaws and lifted (>3 cm) for
--hold-steps consecutive control steps; it fails at --max-steps.
The policy's random noise is seeded per scene, so a run is reproducible and different
settings (e.g. --n-action-steps) can be compared scene by scene.
Latency is reported for policy calls that ran the network (a new action chunk), and
for all calls (most just pop the next action from the chunk queue).
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from so101_sim.env import EnvConfig, SO101GraspEnv, DEFAULT_TEXTURE_DIR
from so101_sim.expert import ExpertConfig, ScriptedGraspExpert

DEFAULT_CAMERA_MAP = ("front:observation.images.camera1,wrist:observation.images.camera2,"
                      "top:observation.images.camera3")
NEW_SCENE_SEED_BASE = 900_000_000  # far from the generator's shard seed ranges
OPEN_PCT = 40.0    # gripper command above this counts as "opened" (home pose holds ~20 %)
CLOSED_PCT = 10.0  # after opening, a command below this counts as "closing"
OUTCOMES = ["success", "never_closed", "grasped_not_lifted", "pushed_object", "closed_away_from_object",
            "closed_near_no_grip"]


def classify(success, tr):
    """Failure category from ground-truth traces (checked in this order)."""
    if success:
        return "success"
    if not tr["closed"]:
        return "never_closed"            # never opened-then-closed: hovered / never committed
    if tr["touched"]:
        return "grasped_not_lifted"      # both jaws touched the object, but no lasting lift
    if tr["max_disp"] > 0.02:
        return "pushed_object"           # knocked the object > 2 cm away
    if tr["dist_at_close"] is not None and tr["dist_at_close"] > 0.02:
        return "closed_away_from_object"  # positioning error at the moment of closing
    return "closed_near_no_grip"         # closed within 2 cm, but never held between both jaws


class ExpertPolicy:
    """Scripted expert behind the same interface, to validate the evaluation harness."""

    def __init__(self, env):
        self.expert = ScriptedGraspExpert(env, ExpertConfig())

    def reset(self, obs, info, seed):
        self.expert.rng = np.random.default_rng([seed, 1])
        return self.expert.reset(obs, info)

    def act(self, obs, task):
        a, _ = self.expert.act(obs)
        return a, False


class LeRobotPolicy:
    """Wraps a LeRobot checkpoint: same preprocessing as LeRobot's own inference loop."""

    def __init__(self, path, camera_map, available_cams, device="cuda", n_action_steps=None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch = torch
        self.device = torch.device(device)
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = device
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
        self.policy.to(self.device).eval()
        if n_action_steps:
            self.policy.config.n_action_steps = n_action_steps
        try:
            self.pre, self.post = make_pre_post_processors(
                self.policy.config, pretrained_path=path,
                preprocessor_overrides={"device_processor": {"device": device}})
        except TypeError:
            self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=path)

        # Which camera key the policy expects for each simulator camera.
        expected = [k for k in self.policy.config.input_features if "image" in k and "empty_camera" not in k]
        self.cam_keys = {}
        for cam in (c for c in camera_map if c in available_cams):  # only cameras the simulator renders
            direct = f"observation.images.{cam}"
            mapped = camera_map.get(cam)
            if direct in expected:
                self.cam_keys[cam] = direct
            elif mapped in expected:
                self.cam_keys[cam] = mapped
        missing = [k for k in expected if k not in self.cam_keys.values()]
        allowed = int(getattr(self.policy.config, "empty_cameras", 0) or 0)
        if len(missing) > allowed:
            raise SystemExit(f"policy expects camera inputs {missing} that are not mapped (for a model "
                             f"trained with the overhead camera, add --top-camera) "
                             f"(only {allowed} may be left empty); policy inputs: {expected}, "
                             f"--camera-map: {camera_map}")
        print("camera mapping:", self.cam_keys, "| left empty (padded by the policy, as in training):", missing)
        self.n_action_steps = self.policy.config.n_action_steps
        self.t = 0

    def reset(self, obs, info, seed):
        # Seed the flow-matching noise per scene: runs are reproducible, and two settings
        # evaluated on the same scene start from the same random numbers.
        self.torch.manual_seed(seed % 2 ** 32)
        self.torch.cuda.manual_seed_all(seed % 2 ** 32)
        self.policy.reset()
        for proc in (self.pre, self.post):
            if hasattr(proc, "reset"):
                proc.reset()
        self.t = 0
        return True

    def act(self, obs, task):
        torch = self.torch
        batch = {"observation.state": torch.from_numpy(obs["state"].astype(np.float32)).unsqueeze(0)}
        for cam, key in self.cam_keys.items():
            if cam not in obs["images"]:
                continue
            img = torch.from_numpy(obs["images"][cam]).float().div(255).permute(2, 0, 1).contiguous()
            batch[key] = img.unsqueeze(0)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        batch["task"] = task
        batch["robot_type"] = "so101_follower"
        new_chunk = self.t % self.n_action_steps == 0
        self.t += 1
        with torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(batch)))
        return action.squeeze(0).float().cpu().numpy(), new_chunk


def write_video(path, frames, fps):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help='"expert" or a LeRobot checkpoint directory')
    ap.add_argument("--urdf", default="assets_so101/so101_sim.urdf")
    ap.add_argument("--eval-seeds", default="datasets/so101_sim_grasp/eval_seeds.json")
    ap.add_argument("--new-scenes", type=int, default=0)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--max-steps", type=int, default=360)
    ap.add_argument("--hold-steps", type=int, default=15)
    ap.add_argument("--n-action-steps", type=int, default=0, help="override chunk execution length (0 = keep)")
    ap.add_argument("--camera-map", default=DEFAULT_CAMERA_MAP,
                    help="sim camera -> policy input; v7 default: left->camera1, right->camera2")
    ap.add_argument("--top-camera", action="store_true", help="render the overhead camera (models trained with it)")
    ap.add_argument("--scene-preset", default="v1", choices=["v1", "v7", "v8"])
    ap.add_argument("--camera-split", default="train", choices=["train", "unseen", "all"],
                    help="v8: viewpoints from the training range or the held-out ones")
    ap.add_argument("--camera-desc", action="store_true", help="v8 model B: append the camera description to the state")
    ap.add_argument("--no-actuator", action="store_true", help="v8: ideal servos (no dead band, no delay), same scenes")
    ap.add_argument("--held-out-limit", type=int, default=-1,
                    help="use only the first N held-out seeds (0 = none; -1 = all)")
    ap.add_argument("--desc-noise", default="", help='model B: wrong description, "cm,deg", e.g. "2,3"')
    ap.add_argument("--texture-dir", default=DEFAULT_TEXTURE_DIR)
    ap.add_argument("--texture-split", default="eval", choices=["train", "eval", "all"],
                    help="v7: evaluate on held-out textures (eval) or training textures (train)")
    ap.add_argument("--camera-preset", default="v1", choices=["v1", "v4"],
                    help="must match the preset of the training data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video-dir", default="")
    ap.add_argument("--results", default="")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    size = (args.width, args.height)
    if args.scene_preset in ("v7", "v8") and args.camera_map == DEFAULT_CAMERA_MAP:
        args.camera_map = "left:observation.images.camera1,right:observation.images.camera2"
    env = SO101GraspEnv(args.urdf, EnvConfig(front_size=size, wrist_size=size, top_camera=args.top_camera,
                                             camera_preset=args.camera_preset, scene_preset=args.scene_preset,
                                             texture_dir=args.texture_dir, texture_split=args.texture_split,
                                             v8_camera_split=args.camera_split, v8_actuator=not args.no_actuator,
                                             record_wrist=args.scene_preset != "v7" or "wrist" in args.camera_map),
                        gui=args.gui)
    if args.policy == "expert":
        policy = ExpertPolicy(env)
    else:
        cmap = dict(item.split(":", 1) for item in args.camera_map.split(",") if item)
        cams = env.camera_names()
        policy = LeRobotPolicy(args.policy, cmap, cams, args.device, args.n_action_steps or None)

    held = []
    if args.held_out_limit != 0:       # --held-out-limit 0: new scenes only, no seed file needed
        with open(args.eval_seeds) as f:
            held = json.load(f)["seeds"]
    scenes = [("held_out", s) for s in (held if args.held_out_limit < 0 else held[:args.held_out_limit])]
    scenes += [("new", NEW_SCENE_SEED_BASE + i) for i in range(args.new_scenes)]
    if args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)

    results, chunk_ms, step_ms = [], [], []
    for k, (kind, seed) in enumerate(scenes):
        obs, info = env.reset(seed=seed)
        task, tgt = info["instruction"], info["target"]
        desc = None
        if args.camera_desc:
            from so101_sim.scene_v8 import perturb_desc
            desc = np.array(info["camera_desc"], np.float32)
            if args.desc_noise:
                cm, deg = (float(x) for x in args.desc_noise.split(","))
                desc = perturb_desc(desc, np.random.default_rng(seed + 7), cm / 100, deg)
        if not policy.reset(obs, info, seed):
            print(f"[{k + 1}/{len(scenes)}] seed {seed}: expert could not plan, skipped")
            continue
        held, success, frames, step = 0, False, [], 0
        p0, local = np.array(tgt["pos"]), env.kin.grasp_local(tgt["size"])
        rel = p0[:2] - env.pan_axis[:2]
        tr = {"radius": float(np.linalg.norm(rel)), "size": float(tgt["size"]), "shape": tgt["shape"],
              "color": tgt["color"],
              "bearing_deg": float(np.degrees(np.arctan2(rel @ env.left[:2], rel @ env.fwd[:2]))),
              "min_dist": np.inf, "dist_at_close": None, "opened": False, "closed": False, "touched": False,
              "max_disp": 0.0, "max_lift": 0.0}
        for step in range(1, args.max_steps + 1):
            if args.video_dir:
                frames.append(np.concatenate([obs["images"][c] for c in env.camera_names()], axis=1))
            t0 = time.perf_counter()
            obs_in = obs if desc is None else {**obs, "state": np.r_[obs["state"], desc].astype(np.float32)}
            action, new_chunk = policy.act(obs_in, task)
            dt = 1000 * (time.perf_counter() - t0)
            step_ms.append(dt)
            if new_chunk:
                chunk_ms.append(dt)
            if action is None:  # expert finished its plan
                break
            obs = env.step(action)
            st = env.grasp_state(tgt["id"], tgt["pos"][2])
            g_pos, g_rot = env.gripper_frame()
            op = env.object_pose(tgt["id"])[0]
            d = float(np.linalg.norm(g_pos + g_rot @ local - op))  # grasp point -> object centre
            tr["min_dist"] = min(tr["min_dist"], d)
            tr["opened"] |= bool(action[5] > OPEN_PCT)
            if tr["opened"] and not tr["closed"] and action[5] < CLOSED_PCT:
                tr["closed"], tr["dist_at_close"] = True, d
            tr["touched"] |= st["jaw_contact"] and st["fixed_contact"]
            tr["max_disp"] = max(tr["max_disp"], float(np.linalg.norm(op[:2] - p0[:2])))
            tr["max_lift"] = max(tr["max_lift"], float(op[2] - p0[2]))
            held = held + 1 if st["grasped"] else 0
            if held >= args.hold_steps:
                success = True
                break
        cat = classify(success, tr)
        if env.v8:
            cam = env.stereo_cam
            tr.update({k: cam[k] for k in ("azimuth_deg", "distance_m", "height_m", "pitch_deg", "hfov_deg", "unseen")
                       if k in cam})
            tr["actuator_active"] = bool(env.act_params.get("active"))
            tr["actuator_k"] = env.act_params.get("k_deg_per_nm")
        results.append({"kind": kind, "seed": seed, "task": task, "success": success, "steps": step,
                        "outcome": cat, **tr})
        dac = "-" if tr["dist_at_close"] is None else f"{100 * tr['dist_at_close']:.1f}cm"
        print(f"[{k + 1}/{len(scenes)}] {kind:8s} seed {seed}: success={success} steps={step} '{task}' "
              f"| {cat}, r={100 * tr['radius']:.0f}cm, dist@close={dac}", flush=True)
        if args.video_dir and frames:
            write_video(os.path.join(args.video_dir, f"{kind}_{seed}_{cat}.mp4"), frames, env.cfg.control_hz)
            frames = []

    print("\n== summary ==")
    summary = {"policy": args.policy, "camera_preset": args.camera_preset, "image_size": list(size),
               "scene_preset": args.scene_preset, "texture_split": args.texture_split,
               "n_action_steps": args.n_action_steps or None}
    for kind in ("held_out", "new"):
        rs = [r for r in results if r["kind"] == kind]
        if rs:
            n_ok = sum(r["success"] for r in rs)
            summary[kind] = {"episodes": len(rs), "success": n_ok, "rate": n_ok / len(rs)}
            print(f"  {kind:8s}: {n_ok}/{len(rs)} = {100 * n_ok / len(rs):.1f}%")
    if results:
        print("\n  outcomes:")
        for cat in OUTCOMES:
            n = sum(r["outcome"] == cat for r in results)
            if n:
                print(f"    {cat:24s} {n:4d}  ({100 * n / len(results):.0f}%)")
        print("\n  success by condition:")
        for label, key, edges in (("distance (cm)", "radius", [0.12, 0.16, 0.20, 0.25]),
                                  ("bearing (deg)", "bearing_deg", [-51, -17, 17, 51]),
                                  ("object size (mm)", "size", [0.0, 0.028, 0.034, 0.05])):
            cells = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                rs = [r for r in results if lo <= r[key] < hi]
                if rs:
                    scale = 1 if key == "bearing_deg" else (100 if key == "radius" else 1000)
                    cells.append(f"{lo * scale:.0f}..{hi * scale:.0f}: {sum(r['success'] for r in rs)}/{len(rs)}")
            print(f"    {label:18s} " + "   ".join(cells))
        summary["outcomes"] = {c: sum(r["outcome"] == c for r in results) for c in OUTCOMES}
        if results and "hfov_deg" in results[0]:
            for label, key, edges in (("camera distance (m)", "distance_m", [0.45, 0.6, 0.75, 0.96]),
                                      ("camera HFOV (deg)", "hfov_deg", [42, 50, 58, 68.1]),
                                      ("camera azimuth (deg)", "azimuth_deg", [-45, -15, 15, 30, 45.1])):
                cells = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    rs = [r for r in results if lo <= r[key] < hi]
                    if rs:
                        cells.append(f"{lo:g}..{hi:g}: {sum(r['success'] for r in rs)}/{len(rs)}")
                print(f"    {label:20s} " + "   ".join(cells))
            for flag, name in ((True, "lagging servos"), (False, "ideal servos")):
                rs = [r for r in results if r.get("actuator_active") == flag]
                if rs:
                    print(f"    {name:20s} {sum(r['success'] for r in rs)}/{len(rs)}")
    if chunk_ms and args.policy != "expert":
        summary["latency_ms"] = {"chunk_inference_median": float(np.median(chunk_ms)),
                                 "chunk_inference_p95": float(np.percentile(chunk_ms, 95)),
                                 "all_calls_median": float(np.median(step_ms))}
        print(f"  chunk inference: median {np.median(chunk_ms):.1f} ms, p95 {np.percentile(chunk_ms, 95):.1f} ms; "
              f"all calls median {np.median(step_ms):.2f} ms")
    if args.results:
        with open(args.results, "w") as f:
            json.dump({"summary": summary, "episodes": results}, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
