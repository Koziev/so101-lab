"""
Objects of the exp11 cell: the user's 3D-printed shapes. Sizes are the measured ranges widened by 1 mm at both
ends (measurement tolerance, print variation); all of them in SHAPES below.

Measurement conventions (change here if a measurement turns out to mean something else):
  hex_prism       "diameter" = across the CORNERS (circumscribed circle)
  dodecahedron    size = distance between two PARALLEL FACES (as a calliper grips it resting on a face)
  pyramid         square base (4 sides), base size = edge length

Every object is a single convex body (exact for these shapes) built from vertices; the same vertex list serves
the collision shape and the visual mesh. Colours: white with weight `white_weight` (the printed PLA), the rest
shared among the other colour names; white varies slightly in brightness and tint. The label is the colour name
plus a shape word (a synonym is drawn for training prompts when enabled).
"""
import math

import numpy as np

# size ranges in metres (measured ranges +- 1 mm)
MM = 0.001
SHAPES = {
    "cube":         {"edge": (17 * MM, 31 * MM)},
    "ball":         {"diameter": (22 * MM, 29 * MM)},
    "cylinder":     {"diameter": (18 * MM, 26 * MM), "height": (19 * MM, 31 * MM)},
    "hex_prism":    {"corners": (23 * MM, 29 * MM), "height": (23 * MM, 31 * MM)},
    "dodecahedron": {"faces": (22 * MM, 31 * MM)},
    "cone":         {"diameter": (24 * MM, 31 * MM), "height": (29 * MM, 41 * MM)},
    "pyramid":      {"base": (24 * MM, 31 * MM), "height": (29 * MM, 41 * MM)},
}
NAMES = {                      # label word first, then synonyms for training prompts
    "cube": ["cube", "box", "block"],
    "ball": ["ball", "sphere"],
    "cylinder": ["cylinder"],
    "hex_prism": ["hexagonal prism", "hex prism"],
    "dodecahedron": ["dodecahedron"],
    "cone": ["cone"],
    "pyramid": ["pyramid"],
}
COLOURS = {                    # name -> RGB (0..1); "white" is drawn with its own weight
    "white": (0.93, 0.93, 0.91), "red": (0.80, 0.12, 0.12), "green": (0.15, 0.62, 0.22),
    "blue": (0.15, 0.30, 0.80), "yellow": (0.92, 0.82, 0.12), "orange": (0.95, 0.50, 0.10),
    "purple": (0.50, 0.22, 0.65), "pink": (0.95, 0.50, 0.68), "black": (0.08, 0.08, 0.08),
    "gray": (0.50, 0.50, 0.50), "brown": (0.45, 0.28, 0.14), "cyan": (0.15, 0.70, 0.78),
}


# ------------------------------------------------------------------------------------------------ geometry
def _prism(n, r_corner, h, phase=0.0):
    a = phase + np.arange(n) * 2 * math.pi / n
    ring = np.c_[r_corner * np.cos(a), r_corner * np.sin(a)]
    return np.r_[np.c_[ring, np.full(n, -h / 2)], np.c_[ring, np.full(n, h / 2)]]


def _apex(n, r_corner, h, phase=0.0):
    """Pyramid / cone with the base centred at z = -h/4 below the centroid... built with the base at z = 0 and
    then shifted so that the body's origin is its bounding-box centre."""
    a = phase + np.arange(n) * 2 * math.pi / n
    base = np.c_[r_corner * np.cos(a), r_corner * np.sin(a), np.zeros(n)]
    v = np.r_[base, [[0, 0, h]]]
    v[:, 2] -= h / 2
    return v


def _dodecahedron(face_dist):
    """Regular dodecahedron, resting on a face (two faces normal to z), scaled to the given face-to-face distance."""
    phi = (1 + 5 ** 0.5) / 2
    v = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    for a, b in ((1 / phi, phi), ):
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                v += [(0, s1 * a, s2 * b), (s1 * a, s2 * b, 0), (s2 * b, 0, s1 * a)]
    v = np.array(v, float)
    # turn it so that one of its faces lies flat: take a face normal from the convex hull, rotate it onto -z
    from scipy.spatial import ConvexHull
    n = ConvexHull(v).equations[0, :3]
    n = n / np.linalg.norm(n)
    target = np.array([0, 0, -1.0])
    axis = np.cross(n, target)
    if np.linalg.norm(axis) < 1e-9:
        R = np.eye(3)
    else:
        k = axis / np.linalg.norm(axis)
        ang = math.acos(np.clip(n @ target, -1, 1))
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
    v = v @ R.T
    return v * (face_dist / (v[:, 2].max() - v[:, 2].min()))


def sample_object(rng, shapes=None, white_weight=0.5, lying_prob=0.25, synonyms=False):
    """A random object spec: shape, label, colour, sizes, vertices (body frame, origin = bounding-box centre,
    resting pose with z up), half height, grasp height above the board, radius for spacing."""
    shapes = shapes or list(SHAPES)
    shape = shapes[int(rng.integers(len(shapes)))]
    R = SHAPES[shape]
    u = lambda key: float(rng.uniform(*R[key]))
    lying = False
    if shape == "cube":
        e = u("edge")
        v = np.array([[x, y, z] for x in (-e / 2, e / 2) for y in (-e / 2, e / 2) for z in (-e / 2, e / 2)])
        dims = {"edge": e}
    elif shape == "ball":
        dd = u("diameter")
        v = None
        dims = {"diameter": dd}
    elif shape == "cylinder":
        dd, h = u("diameter"), u("height")
        v = _prism(32, dd / 2, h)
        dims = {"diameter": dd, "height": h}
        lying = rng.random() < lying_prob
    elif shape == "hex_prism":
        c, h = u("corners"), u("height")
        v = _prism(6, c / 2, h)
        dims = {"corners": c, "height": h}
        lying = rng.random() < lying_prob
    elif shape == "dodecahedron":
        f = u("faces")
        v = _dodecahedron(f)
        dims = {"faces": f}
    elif shape == "cone":
        dd, h = u("diameter"), u("height")
        v = _apex(32, dd / 2, h)
        dims = {"diameter": dd, "height": h}
    else:                                                       # pyramid, square base: corners at r = base/sqrt(2)
        b, h = u("base"), u("height")
        v = _apex(4, b / math.sqrt(2), h, phase=math.pi / 4)
        dims = {"base": b, "height": h}
    if lying and v is not None:                                  # turn the prism onto its side (axis along x)
        v = v[:, [2, 1, 0]] * np.array([1, 1, 1.0])
        if shape == "hex_prism":                                 # rest on a side FACE, not on an edge
            v = v @ np.array([[1, 0, 0], [0, math.cos(math.pi / 6), -math.sin(math.pi / 6)],
                              [0, math.sin(math.pi / 6), math.cos(math.pi / 6)]]).T
    if v is not None:
        v = v - (v.max(axis=0) + v.min(axis=0)) / 2                  # origin = bounding-box centre
        half_h = float(v[:, 2].max())
        radius = float(np.max(np.linalg.norm(v[:, :2], axis=1)))
    else:
        half_h = radius = dims["diameter"] / 2
    # grasp height above the board: the jaws close where the body is wide and held well
    if shape in ("cone", "pyramid"):
        grasp_h = 0.30 * (2 * half_h)                              # low, near the wide base
    else:
        grasp_h = half_h                                            # the middle
    # colour
    names = [c for c in COLOURS if c != "white"]
    colour = "white" if rng.random() < white_weight else names[int(rng.integers(len(names)))]
    rgb = np.array(COLOURS[colour])
    if colour == "white":                                           # PLA: slightly off-white, varies
        rgb = np.clip(rgb * rng.uniform(0.90, 1.0) + rng.normal(0, 0.015, 3), 0, 1)
    else:
        rgb = np.clip(rgb * rng.uniform(0.85, 1.1) + rng.normal(0, 0.03, 3), 0, 1)
    words = NAMES[shape]
    word = words[int(rng.integers(len(words)))] if synonyms else words[0]
    return {"shape": shape, "lying": lying, "dims": dims, "vertices": v, "half_height": half_h,
            "radius": radius, "grasp_height": float(grasp_h), "colour": colour, "rgb": rgb.tolist(),
            "label": f"{colour} {words[0]}", "prompt_word": f"{colour} {word}"}


def create_body(p, spec, position_xy, yaw, client, mass=0.012, friction=0.6):
    """PyBullet body for a spec (convex collision + visual from the same vertices; sphere primitive for balls),
    resting on the board (z = half height), turned by yaw. Returns the body id."""
    kw = {"physicsClientId": client}
    if spec["vertices"] is None:
        r = spec["dims"]["diameter"] / 2
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=r, **kw)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=list(spec["rgb"]) + [1], **kw)
    else:
        verts = np.asarray(spec["vertices"], float).tolist()
        col = p.createCollisionShape(p.GEOM_MESH, vertices=verts, **kw)
        from scipy.spatial import ConvexHull
        hull = ConvexHull(np.asarray(verts))
        idx = []
        c = np.mean(verts, axis=0)
        for s in hull.simplices:                                    # outward winding for the visual mesh
            a, b, d = (np.asarray(verts[i]) for i in s)
            if np.dot(np.cross(b - a, d - a), a - c) < 0:
                s = [s[0], s[2], s[1]]
            idx += [int(i) for i in s]
        vis = p.createVisualShape(p.GEOM_MESH, vertices=verts, indices=idx, rgbaColor=list(spec["rgb"]) + [1], **kw)
    q = p.getQuaternionFromEuler([0, 0, yaw])
    body = p.createMultiBody(mass, col, vis, [position_xy[0], position_xy[1], spec["half_height"] + 0.0005], q, **kw)
    p.changeDynamics(body, -1, lateralFriction=friction, rollingFriction=0.0005 if spec["shape"] == "ball" else 0.0,
                     spinningFriction=0.001, **kw)
    return body
