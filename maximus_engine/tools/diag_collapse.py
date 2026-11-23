"""Dev diagnostic (Blender, read-only): collapsed-triangle counts for theta_fit alone, with/without Corrective Smooth,
and with candidate Corrective Smooth masks / finger curl scales. Prints one EC-02 envelope; saves nothing."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))  # noqa
import json

import numpy as np

import bl_common as C
import posing

from maximus_core import geometry


def main():
    body, rig = C.human_and_rig(); me = body.data
    T = C.mesh_arrays(me)["T"]; V0 = C.evaluated_positions(body)
    W, names = C.weights_matrix(body); dom = np.array(names)[W.argmax(1)]
    fit = json.loads(C.THETA_FIT.read_text()); P = posing.Poser(C.bpy, rig)
    cs = body.modifiers["CorrectiveSmooth"]; vg = body.vertex_groups["_cs_mask_shoulders"]
    out = {}

    def count(tag):
        X = C.evaluated_positions(body); c = geometry.collapsed_triangles(V0, X, T, 0.25)
        from collections import Counter
        out[tag] = {"count": c["count"], "by_bone": Counter(dom[T[c["indices"]]].ravel().tolist()).most_common(6)}

    P.reset(); P.apply_local(fit["bones"]); P.lock_clavicles()
    count("theta_fit_cs_on"); cs.show_viewport = False; cs.show_render = False; count("theta_fit_cs_off"); cs.show_viewport = True; cs.show_render = True
    for pref, sc in (("upperarm", 1.5), ("finger", 1.0), ("metacarpal", 1.0), ("spine05", 1.0), ("pelvis", 1.0)):
        cols = [i for i, n in enumerate(names) if n.startswith(pref)]
        wv = np.clip(W[:, cols].sum(1) * sc, 0, 1)
        for i in np.nonzero(wv > 0.01)[0]:
            vg.add([int(i)], float(max(wv[i], 0)), "ADD")
    count("theta_fit_cs_extended_mask")
    for it in (10, 20):
        cs.iterations = it; count(f"theta_fit_cs_ext_iter{it}")
    cs.iterations = 5
    P.op({"op": "finger_curl", "scale": 0.8}); count("theta_fit_plus_curl0.8_ext")
    return {}, out


C.run("diag_collapse", main)
