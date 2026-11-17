"""
Download the official SO-101 URDF + meshes and build a PyBullet-friendly variant.

    python prepare_assets.py --out assets_so101

Produces assets_so101/so101_sim.urdf. It is identical to so101_new_calib.urdf except
that the two gripper collision meshes are replaced by convex decompositions (VHACD).
PyBullet collides a single STL mesh as its convex hull, which would fill the space
between the jaws and make physical grasping impossible.
"""
import argparse
import os
import re
import struct
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

RAW_BASE = "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101/"
URDF_NAME = "so101_new_calib.urdf"
# Meshes that form the jaws: fixed jaw is part of wrist_roll_follower, moving jaw is separate.
VHACD_MESHES = ["assets/wrist_roll_follower_so101_v1.stl", "assets/moving_jaw_so101_v1.stl"]


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    print("downloading", url)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r, open(path, "wb") as f:
        f.write(r.read())


def load_stl(path):
    """Return (vertices, faces) of a binary or ASCII STL, with duplicate vertices merged."""
    data = open(path, "rb").read()
    tris = None
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            tris = np.frombuffer(data, dtype=dt, count=n, offset=84)["v"].astype(np.float64)
    if tris is None:
        vs = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data)
        tris = np.array([[float(x) for x in v] for v in vs]).reshape(-1, 3, 3)
    flat = tris.reshape(-1, 3)
    verts, inv = np.unique(np.round(flat, 9), axis=0, return_inverse=True)
    return verts, inv.reshape(-1, 3)


def write_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n")
        for t in faces:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def rewrite_urdf(src, dst, replacements):
    """Swap collision mesh filenames (visual meshes stay untouched)."""
    tree = ET.parse(src)
    n = 0
    for coll in tree.iter("collision"):
        mesh = coll.find("geometry/mesh")
        if mesh is not None and mesh.get("filename") in replacements:
            mesh.set("filename", replacements[mesh.get("filename")])
            n += 1
    tree.write(dst, xml_declaration=True, encoding="utf-8")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets_so101")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-vhacd", action="store_true")
    ap.add_argument("--vhacd-resolution", type=int, default=1_000_000)
    args = ap.parse_args()

    urdf_path = os.path.join(args.out, URDF_NAME)
    if not args.skip_download:
        download(RAW_BASE + URDF_NAME, urdf_path)
        meshes = {m.get("filename") for m in ET.parse(urdf_path).iter("mesh")}
        for fn in sorted(meshes):
            download(RAW_BASE + fn, os.path.join(args.out, fn))

    replacements = {}
    if not args.skip_vhacd:
        import pybullet
        for fn in VHACD_MESHES:
            stl = os.path.join(args.out, fn)
            obj = stl[:-4] + ".obj"
            out_rel = fn[:-4] + "_vhacd.obj"
            write_obj(obj, *load_stl(stl))
            pybullet.vhacd(obj, os.path.join(args.out, out_rel), os.path.join(args.out, "vhacd_log.txt"),
                           resolution=args.vhacd_resolution)
            replacements[fn] = out_rel
            print("decomposed", fn, "->", out_rel)

    dst = os.path.join(args.out, "so101_sim.urdf")
    n = rewrite_urdf(urdf_path, dst, replacements)
    print(f"wrote {dst} ({n} collision meshes replaced)")


if __name__ == "__main__":
    main()
