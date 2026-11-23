"""Dev diagnostic (Blender, read-only): collapsed triangles + arm/torso crossings for Corrective Smooth variants,
on theta_fit and on each P pose with the finger-curl op removed. Saves nothing."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))  # noqa
import json
from collections import Counter

import numpy as np

import bl_common as C
import posing
from pose_specs import POSES

from maximus_core import clearance, geometry


def main():
    body, rig = C.human_and_rig(); me = body.data
    T = C.mesh_arrays(me)["T"]; V0 = C.evaluated_positions(body)
    W, names = C.weights_matrix(body); bn = [n for n in names if not n.startswith("_")]
    Wb = W[:, [names.index(n) for n in bn]]; dom = np.array(bn)[Wb.argmax(1)]
    distal = np.array([d.startswith(("upperarm02", "lowerarm", "wrist", "finger", "metacarpal")) for d in dom])
    torso = np.array([d.startswith(("spine", "breast", "pelvis", "root", "upperleg")) for d in dom])
    E = geometry.unique_edges(T); Et = E[distal[E[:, 0]] & distal[E[:, 1]]]; tt = T[np.all(torso[T], axis=1)]
    fit = json.loads(C.THETA_FIT.read_text()); P = posing.Poser(C.bpy, rig)
    cs = body.modifiers["CorrectiveSmooth"]
    out = {}

    def measure():
        X = C.evaluated_positions(body); c = geometry.collapsed_triangles(V0, X, T, 0.25)
        n, _ = clearance.segment_triangle_hits(X[Et[:, 0]], X[Et[:, 1]], X, tt)
        return {"collapsed": c["count"], "crossings": n, "by_bone": Counter(dom[T[c["indices"]]].ravel().tolist()).most_common(4)}

    variants = {"cs_off": None, "simple_orco": ("SIMPLE", False), "length_orco": ("LENGTH_WEIGHTED", False),
                "length_orco_pin": ("LENGTH_WEIGHTED", True)}
    for pose in ("theta_fit", "P1", "P2", "P3"):
        P.reset(); P.apply_local(fit["bones"]); P.lock_clavicles()
        if pose != "theta_fit":
            for o in POSES[pose]["ops"]:
                if o["op"] != "finger_curl":
                    P.op(dict(o))
            P.lock_clavicles()
        for k, v in variants.items():
            if v is None:
                cs.show_viewport = cs.show_render = False
            else:
                cs.show_viewport = cs.show_render = True; cs.smooth_type, cs.use_pin_boundary = v
            out[f"{pose}:{k}"] = measure()
    return {}, out


C.run("diag_cs_variants", main)
