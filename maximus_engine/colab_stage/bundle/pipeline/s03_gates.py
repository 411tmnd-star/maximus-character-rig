"""Stage 03 (system Python): numeric quality gates on the rebuilt master and the P1-P3 poses.

Gates (owner thresholds, 2026-11-16):
  G1 vertex count          exactly 13,380 (rest and every pose)
  G2 invariants            topology, UVs, 152-bone weights identical to v1; shape-key deltas within 1 um; LBS only
  G3 zero drift            every vertex outside the head blend band is bit-identical to v1
  G4 floor contact         Z_sole >= 0.0 mm and <= 0.5 mm in every pose
  G5 collapsed triangles   <= 6 triangles below 0.25 x rest area per pose
  G6 body penetrations     0 crossings of distal-arm / hand / head edges through torso-leg triangles per pose
  G7 clavicle lock         clavicle.L/.R local rotation (0,0,0) in every P JSON
  G8 P3 wrist clearance    hand >= 15 mm off the torso surface
Reported (not gated): L_body (eps 3 mm) energy of hand/forearm vs torso, neck-band growth vs v1, egg fit quality,
the four armpit triangles flagged in FBL-006a.
"""
import json
from pathlib import Path

import numpy as np

from corpus_bridge import contracts
from maximus_core import clearance, geometry

ROOT = Path(__file__).resolve().parents[1]; CACHE = ROOT / "out" / "cache"; REP = ROOT / "out" / "reports"
DISTAL = ("upperarm02", "lowerarm", "wrist", "finger", "metacarpal")
TORSO = ("spine", "breast", "pelvis", "root", "upperleg")
ARMPIT_TRIS = (2607, 7120, 15844, 20357)


def main(**_):
    v1 = np.load(CACHE / "v1_arrays.npz"); v2 = np.load(CACHE / "v2_rest.npz")
    V1, T, W, names = v1["V"], v1["T"], v1["W"], list(v1["group_names"]); V2 = v2["V"]; w = v2["w_head"]
    s01 = json.loads((REP / "s01_master_report.json").read_text())
    Wb = W.copy(); Wb[:, [i for i, n in enumerate(names) if n.startswith("_")]] = -1
    dom = np.array(names)[Wb.argmax(1)]          # dominant BONE (helper groups ignored)
    distal = np.array([d.startswith(DISTAL) for d in dom]); torso = np.array([d.startswith(TORSO) for d in dom])
    head = w > 0.99
    E = geometry.unique_edges(T)
    E_test = E[(distal[E[:, 0]] & distal[E[:, 1]]) | (head[E[:, 0]] & head[E[:, 1]])]
    torso_tris = T[np.all(torso[T], axis=1)]
    hand_L = np.isin(dom, [n for n in names if n.startswith(("wrist", "finger", "metacarpal")) and n.endswith(".L")])
    gates, report = {}, {}

    g = lambda name, ok, value, threshold: gates.__setitem__(name, {"pass": bool(ok), "value": value, "threshold": threshold})  # noqa: E731
    g("G1_vertex_count_rest", len(V2) == 13380, int(len(V2)), 13380)
    inv = s01["invariants"]; g("G2_invariants", all(inv.values()), inv, "all true")
    moved = np.linalg.norm(V2 - V1, axis=1)
    outside = w <= 1e-4
    g("G3_zero_drift_outside_head_band", bool(np.all(V2[outside] == V1[outside])), {"vertices_checked": int(outside.sum()), "max_mm": float(moved[outside].max() * 1000)}, "bit-identical")

    base_idx = set(geometry.collapsed_triangles(V2, np.load(CACHE / "pose_BASE_theta_fit.npz")["V"], T, ratio=0.25)["indices"])
    for pid in ("P1", "P2", "P3"):
        X = np.load(CACHE / f"pose_{pid}.npz")["V"]
        pj = json.loads((ROOT / "poses" / {"P1": "theta_monolith.json", "P2": "theta_contrapposto.json", "P3": "theta_cropped_editorial.json"}[pid]).read_text())
        zmin = float(X[:, 2].min())
        col = geometry.collapsed_triangles(V2, X, T, ratio=0.25)
        n_hit, pairs = clearance.segment_triangle_hits(X[E_test[:, 0]], X[E_test[:, 1]], X, torso_tris)
        a0 = geometry.tri_areas(V2, T); a1 = geometry.tri_areas(X, T)
        armpit = {int(i): round(float(a1[i] / a0[i]), 4) for i in ARMPIT_TRIS if i < len(T)}
        e_body, _, info = clearance.l_body(X[distal & ~torso], X, torso_tris, eps=clearance.EPS_BODY)
        clav = pj["clavicle_quaternions"]
        clav_ok = all(abs(q[0] - 1) < 1e-12 and max(abs(x) for x in q[1:]) < 1e-12 for q in clav.values()) and pj["clavicle_lock_xyz_deg"] == [0.0, 0.0, 0.0]
        g(f"G1_vertex_count_{pid}", len(X) == 13380, int(len(X)), 13380)
        g(f"G4_floor_contact_{pid}", 0.0 <= zmin <= 0.0005, {"z_sole_mm": round(zmin * 1000, 6)}, ">= 0.0 mm and <= 0.5 mm")
        from collections import Counter
        by_bone = Counter(dom[T[col["indices"]]].max(axis=1).tolist() if False else [Counter(dom[T[i]].tolist()).most_common(1)[0][0] for i in col["indices"]])
        g(f"G5_collapsed_tris_{pid}", col["count"] <= 6, {"count": col["count"], "min_ratio": col["min_ratio"], "indices": col["indices"][:24],
                                                            "by_dominant_bone": dict(by_bone)}, "<= 6 under 0.25x rest area")
        g(f"G6_body_penetrations_{pid}", n_hit == 0, {"edge_triangle_crossings": n_hit, "edges_tested": int(len(E_test)), "torso_tris": int(len(torso_tris))}, 0)
        g(f"G7_clavicle_lock_{pid}", clav_ok, clav, "(0,0,0) on X, Y, Z")
        if pid == "P3":
            r = clearance.point_mesh_distance(X[hand_L], X, torso_tris, k=16)
            g("G8_P3_wrist_clearance", float(r["signed"].min()) >= 0.015, {"min_mm": round(float(r["signed"].min()) * 1000, 3)}, ">= 15 mm")
        report[pid] = {"armpit_tri_area_ratio": armpit, "l_body_eps3mm_energy": e_body, "l_body_points_within_3mm": info["n_within_eps"],
                       "l_body_points_penetrating": info["n_penetrating"], "min_signed_distance_mm": round(info["min_signed_m"] * 1000, 3),
                       "pose_checks": pj["checks"],
                       # G5 attribution (report only, the gate above is unchanged): which collapsed triangles the
                       # locked base pose theta_fit already has, and which ones this pose adds or recovers
                       "g5_breakdown": {"shared_with_theta_fit": sorted(base_idx & set(col["indices"])),
                                        "added_by_pose": sorted(set(col["indices"]) - base_idx),
                                        "recovered_vs_theta_fit": sorted(base_idx - set(col["indices"])),
                                        "guard": pj["checks"].get("g5_guard")}}

    XB = np.load(CACHE / "pose_BASE_theta_fit.npz")["V"]; cb = geometry.collapsed_triangles(V2, XB, T, ratio=0.25)
    from collections import Counter
    report["baseline_theta_fit_only"] = {"collapsed_tris": cb["count"], "indices": cb["indices"],
                                         "by_dominant_bone": dict(Counter(Counter(dom[T[i]].tolist()).most_common(1)[0][0] for i in cb["indices"]))}
    band = (w > 1e-4) & (w <= 0.99)
    r = clearance.point_mesh_distance(V2[band], V1, T, k=16)
    report["neck_band_vs_v1"] = {"vertices": int(band.sum()), "max_outward_mm": round(float(r["signed"].max()) * 1000, 3),
                                 "max_inward_mm": round(float(-r["signed"].min()) * 1000, 3),
                                 "lowest_moved_vertex_z_m": round(float(V1[moved > 1e-9, 2].min()), 4)}
    report["egg_fit"] = {k: s01["egg_fit"][k] for k in ("virtual_edges", "head_tris_facing_inward", "inward_area_fraction_of_head", "max_move_mm", "full_weight_vertices")}
    report["egg_fit"].update({k: s01["egg_fit"].get(k) for k in ("unfold", "fold_vertices_sunk", "normal_blend_rings")})
    report["corrective_smooth_mask"] = s01.get("corrective_smooth", {}).get("mask")
    status = "ok" if all(x["pass"] for x in gates.values()) else "fail"
    contracts.atomic_write_json(REP / "s03_gates.json", {"status": status, "gates": gates, "report": report})
    return contracts.envelope("s03_gates", status, outputs={"gates": str(REP / "s03_gates.json")},
                              metrics={"passed": sum(x["pass"] for x in gates.values()), "total": len(gates),
                                       "failed": [k for k, x in gates.items() if not x["pass"]]})
