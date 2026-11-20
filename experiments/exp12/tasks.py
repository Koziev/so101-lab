"""
Tasks and prompts of experiment 11: every scene is built around its prompt, so that the prompt refers to
exactly one object on the board.

Task types (weights in DEFAULT_WEIGHTS, adjustable):
  full         colour + shape     "grab and lift the red cone"          (colour+shape pair unique)
  colour       colour only        "pick up a violet object"             (target colour unique)
  shape        shape only         "take the cylinder"                   (target shape unique)
  superlative  size among a shape "grab the biggest cube"               (2-3 objects of that shape, the
               group             "compare the balls on the table, ...    target's volume exceeds / falls
                                   choose the smallest, grab it, lift it"  short of all others by a margin;
                                                                          no other object has that shape)
Sizes are compared by volume (convex hull), with a margin (default 1.35x, about 10 % in linear size), so the
answer is visible, not a matter of millimetres.

sample_task(rng, ...) -> {"type", "specs" (objects, target first), "instruction", "canonical", ...};
valid(task, specs) re-checks the uniqueness rule on the objects actually placed (some may not fit).
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulator"))
from so101_sim import objects_v11 as ob  # noqa: E402

DEFAULT_WEIGHTS = {"full": 0.4, "colour": 0.2, "shape": 0.2, "superlative": 0.2}

OBJECT_WORDS = ["object", "item", "thing", "piece", "body"]
COLOUR_WORDS = {"purple": ["purple", "violet"], "gray": ["gray", "grey"]}
PLURALS = {"cube": "cubes", "box": "boxes", "block": "blocks", "ball": "balls", "sphere": "spheres",
           "cylinder": "cylinders", "hexagonal prism": "hexagonal prisms", "hex prism": "hex prisms",
           "dodecahedron": "dodecahedra", "cone": "cones", "pyramid": "pyramids"}
SUPERLATIVES = {"biggest": ["biggest", "largest"], "smallest": ["smallest", "tiniest"]}
VERBS = ["grab and lift {np}", "pick up {np}", "take {np}", "grasp {np}", "lift {np}", "grab {np}",
         "pick {np} up", "get {np}"]
SUPERLATIVE_VERBS = ["compare the {plural} on the table, choose the {sup}, grab it and lift it",
                     "find the {sup} of the {plural} and pick it up",
                     "look at the {plural}, take the {sup} one"]


# ------------------------------------------------------------------------------------------------ helpers
def volume(spec):
    if spec["vertices"] is None:
        return math.pi / 6 * spec["dims"]["diameter"] ** 3
    from scipy.spatial import ConvexHull
    return float(ConvexHull(np.asarray(spec["vertices"])).volume)


def recolour(spec, colour, rng):
    """Give a spec another colour (same jitter as objects_v11.sample_object)."""
    rgb = np.array(ob.COLOURS[colour])
    if colour == "white":
        rgb = np.clip(rgb * rng.uniform(0.90, 1.0) + rng.normal(0, 0.015, 3), 0, 1)
    else:
        rgb = np.clip(rgb * rng.uniform(0.85, 1.1) + rng.normal(0, 0.03, 3), 0, 1)
    spec.update({"colour": colour, "rgb": rgb.tolist(), "label": f"{colour} {ob.NAMES[spec['shape']][0]}"})
    return spec


def draw_colour(rng, white_weight, exclude=()):
    names = [c for c in ob.COLOURS if c != "white" and c not in exclude]
    if "white" not in exclude and rng.random() < white_weight:
        return "white"
    return names[int(rng.integers(len(names)))]


def pick(rng, words, vary):
    return words[int(rng.integers(len(words)))] if vary else words[0]


def article(word, definite):
    if definite:
        return "the"
    return "an" if word[0] in "aeiou" else "a"


# ------------------------------------------------------------------------------------------------ rules
def valid(task, specs):
    """Does the task still pick out exactly specs[0] among these objects?"""
    if not specs:
        return False
    t, tgt, others = task["type"], specs[0], specs[1:]
    if t == "full":
        return all((o["colour"], o["shape"]) != (tgt["colour"], tgt["shape"]) for o in others)
    if t == "colour":
        return all(o["colour"] != tgt["colour"] for o in others)
    if t == "shape":
        return all(o["shape"] != tgt["shape"] for o in others)
    if t == "superlative":
        group = [o for o in others if o["shape"] == tgt["shape"]]
        if not group:
            return False
        v, vg = volume(tgt), [volume(o) for o in group]
        m = task["margin"]
        return v >= m * max(vg) if task["superlative"] == "biggest" else v * m <= min(vg)
    raise ValueError(t)


# ------------------------------------------------------------------------------------------------ sampler
def sample_task(rng, weights=None, n_objects=(2, 5), white_weight=0.5, shapes=None, vary=True, margin=1.35,
                lying_prob=0.25, tries=100, verbs=None, superlative_verbs=None):
    """A scene (object specs, target first) and its instruction. vary: synonyms and template variety."""
    weights = weights or DEFAULT_WEIGHTS
    shapes = shapes or list(ob.SHAPES)
    types = list(weights)
    p = np.array([weights[k] for k in types], float)
    ttype = types[int(rng.choice(len(types), p=p / p.sum()))]
    n = int(rng.integers(n_objects[0], n_objects[1] + 1))
    obj = lambda shape_list, colour=None: _obj(rng, shape_list, white_weight, lying_prob, colour)
    task = {"type": ttype, "margin": margin}
    for _ in range(tries):
        if ttype == "superlative":
            shape = shapes[int(rng.integers(len(shapes)))]
            g = int(rng.integers(2, 4))                                  # 2-3 objects of that shape
            group = [obj([shape]) for _ in range(g)]
            sup = "biggest" if rng.random() < 0.5 else "smallest"
            vols = [volume(s) for s in group]
            k = int(np.argmax(vols) if sup == "biggest" else np.argmin(vols))
            specs = [group[k]] + [s for i, s in enumerate(group) if i != k]
            others = [s for s in shapes if s != shape]
            specs += [obj(others) for _ in range(max(0, n - g))]
            task.update({"superlative": sup, "shape": shape})
        else:
            tgt = obj(shapes)
            specs = [tgt]
            for _ in range(n - 1):
                if ttype == "colour":
                    specs.append(obj(shapes, draw_colour(rng, white_weight, exclude=(tgt["colour"],))))
                elif ttype == "shape":
                    specs.append(obj([s for s in shapes if s != tgt["shape"]]))
                else:
                    specs.append(obj(shapes))
        if valid(task, specs):
            task["specs"] = specs
            task.update(phrase(task, specs[0], rng, vary, verbs, superlative_verbs))
            return task
    raise RuntimeError(f"could not build a valid '{ttype}' scene")


def _obj(rng, shape_list, white_weight, lying_prob, colour=None):
    s = ob.sample_object(rng, shapes=shape_list, white_weight=white_weight, lying_prob=lying_prob)
    return recolour(s, colour, rng) if colour else s


# ------------------------------------------------------------------------------------------------ wording
def phrase(task, tgt, rng, vary=True, verbs=None, superlative_verbs=None):
    """Instruction (possibly varied) and the canonical form (first synonyms, first template). verbs /
    superlative_verbs replace the command templates (default: grab and lift)."""
    verbs = verbs or VERBS
    superlative_verbs = superlative_verbs or SUPERLATIVE_VERBS
    def words(v):
        colour = pick(rng, COLOUR_WORDS.get(tgt["colour"], [tgt["colour"]]), v)
        shape = pick(rng, ob.NAMES[tgt["shape"]], v)
        definite = (rng.random() < 0.6) if v else True
        t = task["type"]
        if t == "full":
            np_ = f"{article(colour, definite)} {colour} {shape}"
        elif t == "colour":
            w = pick(rng, OBJECT_WORDS, v)
            np_ = f"{article(colour, definite)} {colour} {w}"
        elif t == "shape":
            np_ = f"{article(shape, definite)} {shape}"
        else:
            sup = pick(rng, SUPERLATIVES[task["superlative"]], v)
            if v and rng.random() < 0.3:
                tmpl = superlative_verbs[int(rng.integers(len(superlative_verbs)))]
                return tmpl.format(plural=PLURALS[shape], sup=sup)
            np_ = f"the {sup} {shape}"
        tmpl = pick(rng, verbs, v)
        return tmpl.format(np=np_)
    canonical = words(False)
    return {"instruction": words(vary) if vary else canonical, "canonical": canonical}
