"""
The automated calibration loop, independent of where the observations come from.

A rig provides:
    rig.read_joints()          -> real joint readings (deg, 5 arm joints + gripper)
    rig.move_to(q_real_deg)       moves the arm (slowly, torque on) and waits for it to settle
    rig.observe()              -> {"q": readings, "wrist": (obj, img) or None, "stereo": (obj, img) or None}
    rig.rest_pose              -> joint readings of the rest pose (real convention)

Loop: stereo frames at the start -> bootstrap (random safe poses until `boot_views` wrist detections) ->
initial estimate -> batches of planned poses -> re-estimate after every batch -> stop when converged
(calib.estimate.converged) or after `max_poses`. Planned poses are in the simulator convention and are sent as
real commands with the current offset estimate (real = simulator - d). Every transition is checked for safety;
unsafe ones go through the rig's safe pose.
"""
import time

import numpy as np

from calib.estimate import CellCalibration, converged


def _as_list(o):
    """Observations as [(board, obj, img)]: accepts None, a single (obj, img) or a list of (board, obj, img)."""
    if o is None:
        return []
    if isinstance(o, tuple) and len(o) == 2:
        return [(0, o[0], o[1])]
    return list(o)


def move_path(rig, waypoints_real):
    """One continuous motion through the waypoints (settling only at the end) if the rig supports it."""
    if hasattr(rig, "move_path"):
        rig.move_path(waypoints_real)
    else:
        for w in waypoints_real:
            rig.move_to(w)


def run_session(rig, cal: CellCalibration, planner, boot_views=8, boot_tries=80, batch=5, min_poses=14,
                max_poses=40, board_centre=None, log=print):
    t0 = time.time()
    for _ in range(3):                                           # stereo camera: the boards are static
        o = rig.observe()
        for b, obj, img in _as_list(o.get("stereo")):
            cal.add_stereo(obj, img, board=b)
    log(f"stereo frames with the board: {len(cal.stereo_obs)}"
        + ("" if cal.stereo_obs else "  (the stereo pose will not be estimated: is the board in the stereo view?)"))

    d = cal.d_prior.copy()
    planner.d = d.copy()
    to_real = lambda q_sim: np.r_[np.asarray(q_sim)[:5] - d, rig.gripper_pct]
    current = np.r_[rig.read_joints()[:5] + d]                    # simulator convention
    visited = []

    skipped = [0]
    rest_sim = np.asarray(current, float).copy()
    leave = planner.route_taught(current, np.asarray(rig.safe_pose_sim, float))
    if leave is None:
        ws = getattr(planner, "ws_raw", None) or planner.ws
        raise RuntimeError("no taught path from the rest pose to the safe pose"
                           + (f" (taught poses form {ws.components()} connected parts)" if ws is not None else ""))
    move_path(rig, [to_real(w) for w in leave])
    current = np.asarray(rig.safe_pose_sim, float).copy()
    log(f"left the rest pose along {len(leave)} taught waypoints")

    def go(q_sim):
        nonlocal current
        q_sim = np.asarray(q_sim, float)
        way = planner.route(current, q_sim)
        if way is None:                                        # no safe route: try via the safe pose
            w1 = planner.route(current, np.asarray(rig.safe_pose_sim, float))
            w2 = planner.route(np.asarray(rig.safe_pose_sim, float), q_sim)
            if w1 is None or w2 is None:                       # never move without a checked route
                skipped[0] += 1
                return False
            way = w1 + w2
        move_path(rig, [to_real(w) for w in way])
        current = q_sim.copy()
        o = rig.observe()
        visited.append(current.copy())
        seen = _as_list(o.get("wrist"))
        for b, obj, img in seen:
            cal.add_sample(o["q"][:5], obj, img, board=b)
        return bool(seen)

    # bootstrap: the mounting is unknown -> sweep the gripper's orientation above the board (systematic),
    # or look around at random if the board's rough position is not given
    if board_centre is not None:
        boot = planner.order(current, planner.bootstrap_poses(board_centre, report=log))[:boot_tries]
    else:
        boot = planner.random_safe(boot_tries)
    # after every hit, try varied poses right around it first: neighbours of a pose that sees the board
    # very likely see it too, from a different direction
    tries, queue = 0, list(boot)
    spread = np.array([8.0, 8.0, 10.0, 15.0, 35.0])                    # pan, lift, elbow, wrist_flex, roll
    while queue and tries < boot_tries and len(cal.samples) < boot_views:
        q = queue.pop(0)
        tries += 1
        if go(q):
            near = planner.random_safe(4, around=q, spread=spread)
            queue = near + queue
    log(f"bootstrap: {len(cal.samples)} wrist views of the board in {tries - skipped[0]} visited poses"
        + (f" ({skipped[0]} skipped: no checked route)" if skipped[0] else ""))
    if len(cal.samples) < 4:
        raise RuntimeError("the wrist camera saw the board in fewer than 4 bootstrap poses: check that the board "
                           "lies in front of the robot and the wrist camera works")
    res = cal.solve()
    d = res["d"].copy()
    planner.d = d.copy()                                          # safety checks and commands: same offsets
    log(_summary(res, len(cal.samples)))
    prev = None
    while len(cal.samples) < max_poses:
        poses = planner.plan(res["X"], res["Z"], [np.r_[s["q"] + d] for s in cal.samples], batch)
        if not poses:
            log("planner found no further informative poses")
            break
        for q in planner.order(current, poses):
            go(q)
        prev, res = res, cal.solve(init=(res["X"], res["Z"], res["S"], res["d"]))
        d = res["d"].copy()
        planner.d = d.copy()
        log(_summary(res, len(cal.samples)))
        if len(cal.samples) >= min_poses and converged(prev, res):
            log("converged")
            break
    home = planner.route(current, np.asarray(rig.safe_pose_sim, float))
    if home is not None:
        move_path(rig, [to_real(w) for w in home])
        current = np.asarray(rig.safe_pose_sim, float)
    back = planner.route_taught(current, rest_sim)
    if back is not None:
        move_path(rig, [to_real(w) for w in back[:-1]])
    else:
        log("WARNING: no taught path back to the rest pose; returning directly (slowly)")
    rig.move_to(rig.rest_pose)
    if skipped[0]:
        log(f"{skipped[0]} planned poses skipped (no route within the taught workspace)")
    res["poses"], res["visited"], res["minutes"] = len(cal.samples), len(visited), (time.time() - t0) / 60
    return res


def _summary(res, n):
    s = res["std"]
    return (f"[{n:2d} views] wrist {res['wrist_rms_px']:.2f} px"
            + (f", stereo {res['stereo_rms_px']:.2f} px" if res["stereo_rms_px"] is not None else "")
            + f" | mount +-{s['X_pos_mm']:.2f} mm/{s['X_rot_deg']:.3f} deg"
            + (f", stereo +-{s['S_pos_mm']:.1f} mm/{s['S_rot_deg']:.2f} deg" if s['S_pos_mm'] is not None else "")
            + " | views per board " + str(res.get("views_per_board", {}))
            + " | offsets " + " ".join(f"{k.split('_')[0]} {v:+.2f}" for k, v in
                                        zip([k for k in s["d_deg"]], [res['d'][i] for i in (1, 2, 3)]))
            + " (+-" + "/".join(f"{v:.2f}" for v in s["d_deg"].values()) + ")")


# ------------------------------------------------------------------------------------------------ replay mode
def with_revisits(stops, every=8, lag=5):
    """Insert revisits: after every `every`-th stop, the stop visited `lag` stops earlier is visited again
    (arriving from elsewhere in the workspace). Used to measure repeatability / play."""
    out = []
    for n, s_ in enumerate(stops):
        out.append(s_)
        if every and n >= lag and n % every == 0:
            out.append(stops[n - lag])
    return out


def choose_stops(points_real, n_stops, keep):
    """Indices (recording order) of ~n_stops stops spread evenly along the recording, among the poses where
    keep(index) is True (e.g. gripper above the board area and jaws above the clearance limit)."""
    ok = [i for i in range(len(points_real)) if keep(i)]
    if not ok:
        return []
    pick = np.linspace(0, len(ok) - 1, min(n_stops, len(ok))).round().astype(int)
    return sorted({ok[k] for k in pick})


def thin(points_real, i0, i1, max_step_deg=8.0):
    """Recorded poses from index i0 to i1 (either direction), thinned so that consecutive waypoints differ by at
    most max_step_deg in any joint (the motion stays a replay of the recorded one)."""
    step = 1 if i1 >= i0 else -1
    way, last = [], np.asarray(points_real[i0], float)
    idx = list(range(i0 + step, i1 + step, step))
    for k, i in enumerate(idx):
        p = np.asarray(points_real[i], float)
        nxt = np.asarray(points_real[idx[k + 1]], float) if k + 1 < len(idx) else None
        if nxt is None or np.max(np.abs(nxt[:5] - last[:5])) > max_step_deg:
            way.append(p)
            last = p
    return way


def make_router(points_real, jaw_ok=None):
    """route(q_from, j): waypoints (real readings) from q_from to recorded pose j through recorded poses that
    satisfy jaw_ok, every step checked; None if there is no such route. ok: per-pose mask of jaw_ok."""
    from calib.workspace import Workspace
    P = np.asarray(points_real, float)[:, :5]
    ok = np.array([True if jaw_ok is None else bool(jaw_ok(p)) for p in P])
    hi = Workspace(P[ok], order=np.nonzero(ok)[0])

    def step_ok(a, b):
        return jaw_ok is None or all(jaw_ok(a + (b - a) * t) for t in np.linspace(0, 1, 8))

    def route(q_from, j):
        way = hi.path(q_from, P[j])
        if way is None:
            return None
        way = way[1:] + [P[j]]
        prev = np.asarray(q_from, float)
        for w in way:
            if not step_ok(prev, w):
                return None
            prev = w
        return thin(np.asarray([np.asarray(q_from, float)] + way), 0, len(way))
    return route, ok


def run_replay(rig, cal: CellCalibration, points_real, stops, jaw_ok=None, gripper_pct=25.0, log=print):
    """Travel along the taught poses (real readings), stop at `stops` (recording indices), observe, estimate.
    jaw_ok(q_real) -> bool: the clearance rule. Leaving and returning to the rest pose replay the raw recording
    (the rest pose is low by nature); all travel between stops goes through recorded poses that satisfy the
    clearance rule, with every step checked (stops without such a route are skipped)."""
    from calib.workspace import Workspace
    t0 = time.time()
    P = np.asarray(points_real, float)[:, :5]
    ok = np.array([True if jaw_ok is None else bool(jaw_ok(p)) for p in P])
    hi = Workspace(P[ok], order=np.nonzero(ok)[0])                  # taught poses above the clearance limit

    def step_ok(a, b):
        return jaw_ok is None or all(jaw_ok(a + (b - a) * t) for t in np.linspace(0, 1, 8))

    def route(q_from, j):
        way = hi.path(q_from, P[j])
        if way is None:
            return None
        way = way[1:] + [P[j]]
        prev = np.asarray(q_from, float)
        for w in way:
            if not step_ok(prev, w):
                return None
            prev = w
        return thin(np.asarray([np.asarray(q_from, float)] + way), 0, len(way))

    for _ in range(3):                                           # stereo camera: the boards are static
        o = rig.observe()
        for b, obj, img in _as_list(o.get("stereo")):
            cal.add_stereo(obj, img, board=b)
    log(f"stereo frames with the board: {len(cal.stereo_obs)}"
        + ("" if cal.stereo_obs else "  (the stereo pose will not be estimated: is the board in the stereo view?)"))
    start = int(np.argmin(np.max(np.abs(P - rig.read_joints()[:5]), axis=1)))   # nearest recorded pose to rest
    after = next((i for i in range(start, len(P)) if ok[i]), None)
    before = next((i for i in range(start, -1, -1) if ok[i]), None)
    cands = [i for i in (after, before) if i is not None]
    if not cands:
        raise RuntimeError("no recorded pose is above the clearance limit")
    first_hi = min(cands, key=lambda i: abs(i - start))
    log(f"replay: leave rest along the recording (poses {start}..{first_hi}), then {len(stops)} stops "
        f"through {int(ok.sum())} poses above the clearance limit")
    move_path(rig, [np.r_[w, gripper_pct] for w in thin(P, start, first_hi)] or [np.r_[P[first_hi], gripper_pct]])
    cur, seen, skipped = P[first_hi].copy(), 0, 0
    for n, s_ in enumerate(stops):
        way = route(cur, s_)
        if way is None:
            skipped += 1
            log(f"  stop {n + 1:2d}/{len(stops)} (recorded pose {s_}): skipped, no route above the clearance limit")
            continue
        move_path(rig, [np.r_[w, gripper_pct] for w in way])
        cur = P[s_].copy()
        if hasattr(rig, "tag"):
            rig.tag = {"stop": n, "pose_index": int(s_)}
        o = rig.observe()
        hits = _as_list(o.get("wrist"))
        for b, obj, img in hits:
            cal.add_sample(o["q"][:5], obj, img, board=b)
        seen += bool(hits)
        log(f"  stop {n + 1:2d}/{len(stops)} (recorded pose {s_}): "
            + (", ".join(f"board {'AB'[b] if b < 2 else b} {len(obj)} corners" for b, obj, _ in hits) if hits else "no board"))
    back = route(cur, first_hi)
    if back is not None:
        move_path(rig, [np.r_[w, gripper_pct] for w in back])
        move_path(rig, [np.r_[w, gripper_pct] for w in thin(P, first_hi, start)] or [np.r_[P[start], gripper_pct]])
    else:
        log("WARNING: no route back above the clearance limit; returning to rest directly (slowly)")
    rig.move_to(rig.rest_pose)
    log(f"wrist views with a board: {seen} of {len(stops) - skipped} visited stops"
        + (f" ({skipped} skipped)" if skipped else ""))
    if len(cal.samples) < 4:
        raise RuntimeError(f"only {len(cal.samples)} wrist views of a board: not enough for the calibration "
                           f"(check the images in the session folder with calib/check_images.py)")
    res = cal.solve()
    log(_summary(res, len(cal.samples)))
    res["poses"], res["visited"], res["minutes"] = len(cal.samples), len(stops) - skipped, (time.time() - t0) / 60
    return res
