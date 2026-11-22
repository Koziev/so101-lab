"""
Stage 1 of the calibration: teach the arm's safe workspace by hand, the way LeRobot's calibration records
joint ranges. The calibration later moves ONLY within the recorded poses and along paths between nearby ones.

    python calib/workspace.py record --setup setups/<name> --port /dev/ttyACM0
    python calib/workspace.py show   --setup setups/<name>

Recording: the arm starts folded in its rest pose; after confirming that you support it, the torque goes off.
Move the arm through every pose the calibration may use: the gripper over the whole board area at different
heights (lowest = just above the board, never touching), and in each region with strongly varied wrist angles
(wrist_flex and especially wrist_roll). Stay away from the table edges, the camera, cables and the extremes
of the joints. A pose is stored when it differs from every stored pose by at least --min-step degrees in some
joint, so standing still adds nothing. 'q' + Enter or Ctrl+C saves <setup>/workspace.json; the torque then
comes back on at the current pose and the arm returns slowly to its rest pose.
Angles are the arm's own readings (real convention), so the workspace does not depend on any calibration.
"""
import argparse
import json
import os
import select
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class Workspace:
    """Recorded safe poses (real readings, deg, 5 arm joints)."""

    def __init__(self, points, tol_deg=3.0, step_deg=10.0, order=None):
        self.P = np.asarray(points, float)[:, :5]
        self.tol, self.step = tol_deg, step_deg
        # recording order: consecutive recorded poses are connected (the arm physically moved between them)
        self.order = np.arange(len(self.P)) if order is None else np.asarray(order)
        self._nbrs = None

    @classmethod
    def load(cls, path, **kw):
        return cls(json.load(open(path))["points"], **kw)

    def _dist(self, q):
        return np.max(np.abs(self.P - np.asarray(q, float)[:5]), axis=1)          # max-joint distance

    def contains(self, q_real):
        return bool(self._dist(q_real).min() <= self.tol)

    def nearest(self, q_real):
        return int(np.argmin(self._dist(q_real)))

    def filtered(self, keep):
        """A workspace with only the points where keep(point) is True (e.g. a height margin)."""
        mask = np.array([bool(keep(p)) for p in self.P])
        return Workspace(self.P[mask], self.tol, self.step, order=self.order[mask])

    def sample(self, n, rng, around=None, spread=None):
        """n poses from the taught region: recorded points (all, or those within +-spread of `around`), each
        moved a random fraction towards a recorded neighbour, i.e. between two nearby taught poses."""
        if around is None:
            cand = self.P
        else:
            cand = self.P[np.max(np.abs((self.P - np.asarray(around, float)[:5]) / np.asarray(spread, float)), axis=1) <= 1.0]
        if len(cand) == 0:
            return []
        out = []
        for q in cand[rng.integers(len(cand), size=n)]:
            j = self._neighbours()[self.nearest(q)]
            if len(j):
                q = q + rng.uniform(0, 0.5) * (self.P[rng.choice(j)] - q)
            out.append(q)
        return out

    def _neighbours(self):
        """Spatial neighbours (all joints within step_deg) plus the poses recorded just before and after
        (only where no pose between them was removed, e.g. by the clearance filter)."""
        if self._nbrs is None:
            pos = {int(o): i for i, o in enumerate(self.order)}
            self._nbrs = []
            for i, p in enumerate(self.P):
                near = set(np.nonzero((self._dist(p) <= self.step) & (self._dist(p) > 0))[0].tolist())
                for o in (int(self.order[i]) - 1, int(self.order[i]) + 1):
                    if o in pos:
                        near.add(pos[o])
                self._nbrs.append(np.array(sorted(near), dtype=int))
        return self._nbrs

    def components(self):
        """Number of connected parts of the neighbour graph (1 = every taught pose reachable from every other)."""
        nb, seen, parts = self._neighbours(), np.zeros(len(self.P), bool), 0
        for s in range(len(self.P)):
            if seen[s]:
                continue
            parts += 1
            stack = [s]
            seen[s] = True
            while stack:
                i = stack.pop()
                for j in nb[i]:
                    if not seen[j]:
                        seen[j] = True
                        stack.append(j)
        return parts

    def path(self, q_from, q_to):
        """Waypoints (recorded poses) from q_from to q_to through neighbours (shortest path, edges <= step deg).
        None if the two are not connected."""
        import heapq
        a, b = self.nearest(q_from), self.nearest(q_to)
        nb = self._neighbours()
        dist, prev, heap = {a: 0.0}, {}, [(0.0, a)]
        while heap:
            d, i = heapq.heappop(heap)
            if i == b:
                break
            if d > dist.get(i, 1e18):
                continue
            for j in nb[i]:
                nd = d + float(np.max(np.abs(self.P[i] - self.P[j])))
                if nd < dist.get(j, 1e18):
                    dist[j], prev[j] = nd, i
                    heapq.heappush(heap, (nd, j))
        if b not in dist:
            return None
        way, i = [], b
        while i != a:
            way.append(self.P[i])
            i = prev[i]
        return [self.P[a]] + way[::-1]


def record(args):
    from calib.lerobot_helpers import make_robot, move_to, read_joints, send
    from setup_io import Setup
    setup = Setup(args.setup)
    robot = make_robot(args.port, setup.info["arm_id"], 20.0)
    rest = read_joints(robot)
    points = []
    try:
        input("Arm folded at rest. SUPPORT THE ARM, then press Enter: the torque goes off and recording starts...")
        robot.bus.disable_torque()
        print("Recording. Move the arm through the safe workspace; 'q' + Enter (or Ctrl+C) saves.")
        last_print = 0.0
        while True:
            q = read_joints(robot)
            if not points or np.min(np.max(np.abs(np.asarray(points)[:, :5] - q[:5]), axis=1)) >= args.min_step:
                points.append(q.tolist())
            if time.time() - last_print > 1.0:
                print(f"\r  {len(points)} poses", end="", flush=True)
                last_print = time.time()
            if select.select([sys.stdin], [], [], 0)[0] and sys.stdin.readline().strip().lower() == "q":
                break
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n{len(points)} poses recorded")
        try:
            send(robot, read_joints(robot))
            robot.bus.enable_torque()
            move_to(robot, rest, seconds=5.0)
        except Exception as e:
            print("could not park the arm:", e)
        robot.disconnect()
    if points:
        P = np.asarray(points)
        data = {"arm_id": setup.info["arm_id"], "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
                "convention": "real readings (deg), LeRobot degrees mode", "min_step_deg": args.min_step,
                "points": points, "ranges": {"min": P.min(0).round(1).tolist(), "max": P.max(0).round(1).tolist()}}
        setup.save("workspace.json", data, note=f"workspace taught by hand ({len(points)} poses)")
        print("saved", setup.file("workspace.json"))


def show(args):
    from setup_io import JOINTS, Setup
    d = Setup(args.setup).load("workspace.json")
    if d is None:
        raise SystemExit("no workspace.json in this setup")
    P = np.asarray(d["points"])
    print(f"{len(P)} poses recorded {d['recorded']}")
    for j, lo, hi in zip(JOINTS, P.min(0), P.max(0)):
        print(f"  {j:14s} {lo:7.1f} .. {hi:7.1f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--setup", required=True)
    r.add_argument("--port", default="/dev/ttyACM0")
    r.add_argument("--min-step", type=float, default=3.0, help="store a pose when some joint differs by >= this (deg)")
    r.add_argument("--hz", type=float, default=20.0)
    s = sub.add_parser("show")
    s.add_argument("--setup", required=True)
    args = ap.parse_args()
    {"record": record, "show": show}[args.cmd](args)


if __name__ == "__main__":
    main()
