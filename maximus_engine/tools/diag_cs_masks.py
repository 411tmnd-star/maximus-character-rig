"""Dev diagnostic (Blender, read-only): collapsed triangles (< 0.25x rest) per P pose for Corrective Smooth mask /
smooth-type variants at the locked settings (repeat 5, factor 0.50). Poses are read from poses/*.json. Saves nothing."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))  # noqa
import json
from collections import Counter

import numpy as np

import bl_common as C
import posing

from maximus_core import geometry

MASKS = {
    "shoulder_cap": ("clavicle", "shoulder01"),
    "current": ("clavicle", "shoulder01", "upperarm01"),
    "creases": ("clavicle", "shoulder01", "upperarm01", "upperarm02", "spine05", "pelvis", "upperleg01", "finger", "metacarpal", "lowerleg01"),
    "creases_no_armpit": ("clavicle", "shoulder01", "spine05", "pelvis", "upperleg01", "finger", "metacarpal", "lowerleg01"),
}


def main():
    body, rig = C.human_and_rig(); me = body.data
    T = C.mesh_arrays(me)["T"]
    W, names = C.weights_matrix(body); bones = [i for i, n in enumerate(names) if not n.startswith("_")]
    dom = np.array(names)[bones][W[:, bones].argmax(1)]
    P = posing.Poser(C.bpy, rig); cs = body.modifiers["CorrectiveSmooth"]; vg = body.vertex_groups["_cs_mask_shoulders"]
    P.reset(); V0 = C.evaluated_positions(body)
    poses = {p: json.loads((C.ROOT / "poses" / f).read_text()) for p, f in
             (("P1", "theta_monolith.json"), ("P2", "theta_contrapposto.json"), ("P3", "theta_cropped_editorial.json"))}
    out = {}
    for mname, prefixes in MASKS.items():
        cols = [i for i, n in enumerate(names) if n.split(".")[0].startswith(prefixes)]
        wv = np.clip(W[:, cols].sum(1) * 1.5, 0, 1)
        vg.remove(list(range(len(me.vertices))))
        for i in np.nonzero(wv > 0.01)[0]:
            vg.add([int(i)], float(wv[i]), "REPLACE")
        for st in ("SIMPLE", "LENGTH_WEIGHTED", "OFF"):
            if st == "OFF":
                cs.show_viewport = cs.show_render = False
            else:
                cs.show_viewport = cs.show_render = True; cs.smooth_type = st
            row = {}
            for pid, pj in poses.items():
                P.reset(); P.apply_local(pj["bones"])
                X = C.evaluated_positions(body); c = geometry.collapsed_triangles(V0, X, T, 0.25)
                row[pid] = [c["count"], Counter(dom[T[c["indices"]]].ravel().tolist()).most_common(3)]
            out[f"{mname}:{st}"] = row
            if st == "OFF":
                cs.show_viewport = cs.show_render = True
    return {}, out


C.run("diag_cs_masks", main)
