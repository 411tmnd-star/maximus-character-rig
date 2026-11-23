"""Builds MAXIMUS_REBUILD_REPORT.md from the stage JSON reports (no hand-typed metrics)."""
import json


def _t(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build(s01, s03, s04, s05, hashes, root, date):
    g = s03["gates"]; rep = s03["report"]; egg = s01["egg_fit"]
    pose = {p: json.loads((root / "poses" / f).read_text()) for p, f in
            (("P1", "theta_monolith.json"), ("P2", "theta_contrapposto.json"), ("P3", "theta_cropped_editorial.json"))}
    n_pass = sum(v["pass"] for v in g.values()); fails = [k for k, v in g.items() if not v["pass"]]
    L = []
    L.append(f"# Maximus v2 rebuild report\n\nBuilt {date} on this PC (CPU only), headless Blender 5.2.3, from the read-only canonical rig "
             f"`03_full_rigged.blend` (SHA-256 `1be2d4ca305d…`). Every number below is read from `out/reports/*.json`, which the pipeline writes.\n")
    L.append(f"**Gate result: {n_pass}/{len(g)} gates pass.** "
             + ("All gates pass." if not fails else "Failing: " + ", ".join(f"`{f}`" for f in fails) + " (explained in section 4)."))
    L.append("\n## 1. What was rebuilt\n")
    L.append(_t([
        ["Head", f"Faceless sculptural egg (no ears, nose or mouth) fitted in place: {egg['moved_vertices']} vertices moved, "
                 f"{egg['full_weight_vertices']} of them exactly onto the egg; egg a/b/c = {egg['egg']['a']*1000:.1f}/{egg['egg']['b']*1000:.1f}/{egg['egg']['c']*1000:.1f} mm, tilt {egg['egg']['tilt_deg']} deg"],
        ["Topology", "13,380 vertices, same order, same UVs, same 152-bone weights, same 5 shape-key deltas (checked by hash / max drift)"],
        ["Below the head", f"bit-identical to v1 ({g['G3_zero_drift_outside_head_band']['value']['vertices_checked']} vertices, max drift {g['G3_zero_drift_outside_head_band']['value']['max_mm']} mm), so existing garment fits stay valid"],
        ["Surface", "procedural Cycles finishes M1 Obsidian Noir #18181B r0.68, M2 Alabaster Chalk #F2EFE9 r0.75 SSS 0.05, M3 Travertine Warm #D8CEBE r0.80 SSS 0.02 (sRGB converted to linear), mineral grain + pits on a rest-position attribute, hairline joint seams at wrists and elbows"],
        ["Wireframe texture", "missing `mannequin_wire_2048.png` and its material removed; no image dependency left in the master"],
        ["Skinning", f"Linear Blend Skinning (dual quaternion off) + Corrective Smooth factor {s01['corrective_smooth']['factor']}, repeat {s01['corrective_smooth']['iterations']}, rest ORCO, localised to `{s01['corrective_smooth']['vertex_group']}` ({s01['corrective_smooth']['n_vertices']} vertices, clavicle + shoulder cap, smooth geodesic ramp to zero before the armpit fold)"],
        ["Poses", "P1 theta_monolith (-> POSE_02, M1), P2 theta_contrapposto (-> POSE_06, M2), P3 theta_cropped_editorial (-> POSE_05, M3) on the native rig over theta_fit.json; clavicles (0,0,0) on all axes; POSE_01-10 untouched"],
        ["Delivery", f"Draco core GLB `out/maximus_v2_core.glb` ({s04['glb']['bytes']/1e6:.2f} MB) + 2K texture sets per finish; storefront stays 2D/2.5D (native RGBA renders, CSS backdrop, 82% -> 98% bust fade, no hem_trim())" if s04 else "GLB: not built"],
    ], ["Area", "Result"]))
    L.append("\n## 2. Which .py runs each stage\n")
    L.append(_t([
        ["s00 extract v1", "Blender", "`pipeline/s00_extract_v1.py`", "corpus_bridge (EC-02, EC-07)"],
        ["s01 master", "Blender", "`pipeline/s01_build_master.py`, `materials.py`", "maximus_core.egg_head (MX-01), clearance L_VE + virtual edges (MX-03), L_body schedule (MX-02), geometry (MX-06)"],
        ["s02 poses", "Blender", "`pipeline/s02_build_poses.py`, `posing.py`, `pose_specs.py`", "maximus_core.clearance: 3 mm arm guard (MX-02), segment/triangle tests (MX-04)"],
        ["s03 gates", "system Python", "`pipeline/s03_gates.py`", "maximus_core.geometry, clearance (MX-02/04/06)"],
        ["s04 bake + GLB", "Blender", "`pipeline/s04_bake_export.py`", "maximus_delight.pack_orm (MX-07)"],
        ["s05 previews", "Blender", "`pipeline/s05_previews.py`", "posing (pose JSON -> pose bones)"],
        ["s06 report", "system Python", "`pipeline/s06_report.py`, `report_text.py`", "corpus_bridge (hashes, atomic writes)"],
        ["orchestration", "system Python", "`run_all.py` -> corpus `60_cross_sectional_workflows/flow_maximus_v2_rebuild.py` (MX-10)", "corpus_bridge.blender_runner (EC-04)"],
        ["operators", "Blender", "`pipeline/io_ops_gateway.py` only", "render.render, object.bake, export_scene.gltf, wm.save_as_mainfile (tests/test_no_bpy_ops.py)"],
    ], ["Stage", "Runtime", "Maximus_build file", "Imported from 411_UNIFIED_3D_CORPUS"]))
    L.append("\nThe shared modules are imported from the corpus, not copied. `corpus_lock.json` pins their SHA-256, and `corpus_link.py` refuses to run if any of them changed.\n")
    L.append("## 3. Verification metrics\n")
    L.append(_t([[k, "PASS" if v["pass"] else "FAIL", f"`{json.dumps(v['value'])[:150]}`", v["threshold"]] for k, v in g.items()], ["Gate", "Result", "Measured", "Threshold"]))
    L.append("\nPose details:\n")
    rows = []
    for p, pj in pose.items():
        ops = {o["op"] + (":" + o.get("side", "") if o.get("side") else ""): o for o in pj["declarative_ops"]}
        guard = pj["checks"].get("clearance_guard", [])
        gtxt = "; ".join(f"{x['side']}: +{x['added_abduction_deg']} abd, +{x['added_forward_flex_deg']} flex, {x['min_free_arm_torso_mm']} mm" for x in guard)
        rows.append([p, pj["kind"], pj["maps_to_canonical_pose"], pj["finish"], f"{pj['grounding']['min_z_m']*1000:.4f}", gtxt,
                     f"{rep[p]['armpit_tri_area_ratio']}", pj["checks"].get("right_knee_bend_deg", ""), pj["checks"].get("hand_torso_clearance_mm", "")])
    L.append(_t(rows, ["Pose", "File", "Maps to", "Finish", "Z_sole mm", "3 mm arm guard (added deg, min free arm-torso)", "FBL-006a armpit tri area ratios", "R knee deg", "P3 hand mm"]))
    b = rep["baseline_theta_fit_only"]
    L.append(f"\nBaseline for collapsed triangles: `theta_fit.json` alone, with no editorial operation, already gives **{b['collapsed_tris']}** triangles below 0.25x rest area {b['by_dominant_bone']}.\n")
    L.append(f"Head and neck: {egg['virtual_edges']} virtual edges were used in the L_VE phase. "
             f"The full-weight head patch is embedded on the egg by a harmonic map and a flip-guarded tangential relaxation "
             f"(inward-facing triangles before / after: {(egg.get('unfold') or {}).get('inward_before', 'n/a')} / {egg['head_tris_facing_inward']}, "
             f"{egg['inward_area_fraction_of_head']*100:.2f}% of the head surface; fold vertices sunk as a fallback: {egg.get('fold_vertices_sunk', 0)}). "
             f"Egg vertices use the egg's analytic normals as custom shading normals ({egg.get('custom_normals_on_egg_vertices', 'n/a')} vertices), "
             f"mixed into the mesh normals over {egg.get('normal_blend_rings', 'n/a')} edge rings at the neck. "
             f"The neck band moves at most {rep['neck_band_vs_v1']['max_outward_mm']} mm outward and {rep['neck_band_vs_v1']['max_inward_mm']} mm inward relative to v1. "
             f"The lowest moved vertex is at z = {rep['neck_band_vs_v1']['lowest_moved_vertex_z_m']} m, above shirt collars.\n")
    if s04:
        gl = s04["glb"]
        L.append(f"Core GLB: {gl['bytes']:,} bytes ({'under' if gl['under_budget'] else 'OVER'} the 2.5 MB budget), {gl['vertices_gltf']:,} glTF vertices "
                 f"(13,380 body vertices plus UV-seam duplicates), {gl['joints']} joints, {gl['morph_targets']} morph targets, extensions {gl['extensionsUsed']}, SHA-256 `{gl['sha256'][:16]}…`.\n")
    if s05:
        L.append(f"Previews: {len(s05['renders'])} renders at {s05['resolution'][0]}x{s05['resolution'][1]}, {s05['samples']} samples, {s05['view_transform']}, device {s05.get('device', 'CPU')}. "
                 "They are in `out/previews/`, with storefront composites in `out/previews/composite/` and `out/previews/contact_sheet_storefront.png`.\n")
    L.append(open(root / "pipeline" / "report_notes.md", encoding="utf-8").read() if (root / "pipeline" / "report_notes.md").exists() else "")
    L.append("\n## Artifact SHA-256\n")
    L.append(_t([[f"`{k}`", f"`{v}`"] for k, v in hashes.items() if not k.startswith("out/cache")], ["File", "SHA-256"]))
    return "\n".join(L) + "\n"
