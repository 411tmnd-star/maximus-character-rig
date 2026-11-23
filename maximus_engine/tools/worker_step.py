"""Worker verification protocol for FABLE_52_MAXIMUS_HANDOFF.md. Three commands, run in order from Maximus_build:

    python -I tools/worker_step.py 1     unit tests + headless s01..s04, then assert 19/19 gates (G5 <= 6 on P1, P2, P3)
    python -I tools/worker_step.py 2     ONE fast preview (P1 / M1, 32 samples), then assert tone, headroom, clean head
    python -I tools/worker_step.py 3     pack colab_stage/maximus_v2_colab_bundle.zip for the Colab L4 batch

Each step stops at the first failing sub-step, prints a JSON summary, and exits non-zero on any failed assertion.
Thresholds are asserted here exactly as the gates define them; this file must not be edited to make a step pass.
"""
import os
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_TESTS = Path(os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/01_code_library/60_cross_sectional_workflows/tests')))
REP = ROOT / "out" / "reports"


def sh(*args):
    print("\n$ python -I " + " ".join(str(a) for a in args), flush=True)
    return subprocess.call([sys.executable, "-I", *[str(a) for a in args]], cwd=str(ROOT))


def load(name):
    p = REP / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def finish(summary, checks):
    summary["checks"] = checks; summary["pass"] = all(checks.values())
    print("\n" + json.dumps(summary, indent=1, default=str)); print("STEP", summary["step"], "PASS" if summary["pass"] else "FAIL")
    return 0 if summary["pass"] else 1


def step1():
    ran = {}
    for name, args in (("corpus_unit_tests", ("-m", "unittest", "discover", "-s", CORPUS_TESTS)),
                       ("bpy_ops_audit", ("-m", "unittest", "discover", "-s", ROOT / "tests")),
                       ("s01_build_master", ("run_stage.py", "s01_build_master")), ("s02_build_poses", ("run_stage.py", "s02_build_poses")),
                       ("s03_gates", ("run_stage.py", "s03_gates")), ("s04_bake_export", ("run_stage.py", "s04_bake_export"))):
        ran[name] = sh(*args)
        if ran[name] != 0:
            break
    s01, s03, s04 = load("s01_master_report.json"), load("s03_gates.json"), load("s04_bake_export.json")
    fresh = all(ran.get(k) == 0 for k in ("s01_build_master", "s02_build_poses")) and "s03_gates" in ran
    g = s03["gates"] if (s03 and fresh) else {}
    egg = s01["egg_fit"] if (s01 and ran.get("s01_build_master") == 0) else {}
    unf = egg.get("unfold") or {}
    g5 = {p: (g.get(f"G5_collapsed_tris_{p}") or {}).get("value", {}).get("count") for p in ("P1", "P2", "P3")}
    summary = {"step": 1, "exit_codes": ran, "gates_passed": sum(v["pass"] for v in g.values()), "gates_total": len(g),
               "gates_failed": [k for k, v in g.items() if not v["pass"]], "G5_counts": g5,
               "G5_guard_chosen": {p: ((s03 or {}).get("report", {}).get(p, {}).get("g5_breakdown", {}).get("guard") or {}).get("chosen") for p in ("P1", "P2", "P3")} if fresh else None,
               "G5_added_by_pose": {p: (s03 or {}).get("report", {}).get(p, {}).get("g5_breakdown", {}).get("added_by_pose") for p in ("P1", "P2", "P3")} if fresh else None,
               "egg": {"head_tris_facing_inward": egg.get("head_tris_facing_inward"), "fold_vertices_sunk": egg.get("fold_vertices_sunk"),
                       "max_move_mm": egg.get("max_move_mm"), "unfold": unf},
               "invariants": (s01 or {}).get("invariants") if egg else None,
               "cs_mask": (s01 or {}).get("corrective_smooth") if egg else None,
               "glb": ((s04 or {}).get("glb") if ran.get("s04_bake_export") == 0 else None)}
    glb = summary["glb"] or {}
    checks = {"unit_tests_ok": ran.get("corpus_unit_tests") == 0 and ran.get("bpy_ops_audit") == 0,
              "all_stages_exit_0": len(ran) == 6 and all(v == 0 for v in ran.values()),
              "gates_19_of_19": len(g) == 19 and all(v["pass"] for v in g.values()),
              "G5_le_6_on_P1_P2_P3": all(isinstance(c, int) and c <= 6 for c in g5.values()),
              "invariants_all_true": bool(summary["invariants"]) and all(summary["invariants"].values()),
              "egg_no_inward_triangles": egg.get("head_tris_facing_inward") == 0 and unf.get("inward_after_harmonic") == 0 and unf.get("inward_after") == 0,
              "egg_no_sunk_folds_no_islands": egg.get("fold_vertices_sunk") == 0 and unf.get("island_vertices") == 0,
              "cs_mask_not_empty": ((summary["cs_mask"] or {}).get("n_vertices") or 0) > 0,
              "glb_152_joints_5_morphs_under_budget": glb.get("joints") == 152 and glb.get("morph_targets") == 5 and bool(glb.get("under_budget"))}
    return finish(summary, checks)


def step2():
    png = ROOT / "out" / "previews" / "maximus_v2_P1_M1_fast.png"
    rc = sh("run_stage.py", "s05_previews", "poses=P1", "finishes=M1", "samples=32", "tag=fast")
    rc2 = sh("tools/check_preview.py", png, "M1") if rc == 0 else 1
    s05 = load("s05_previews_fast.json") if rc == 0 else None
    r = ((s05 or {}).get("renders") or {}).get("P1_M1", {})
    summary = {"step": 2, "exit_codes": {"s05_previews": rc, "check_preview": rc2}, "png": str(png),
               "s05_stats": r.get("stats"), "exposure": r.get("exposure"), "auto_exposure": r.get("auto_exposure"),
               "camera": ((s05 or {}).get("camera") or {}).get("P1"), "head_crop": str(png)[:-4] + "_head_crop.png"}
    st = r.get("stats") or {}
    checks = {"render_ok": rc == 0, "check_preview_pass": rc2 == 0,
              "auto_exposure_converged": (r.get("auto_exposure") or {}).get("converged") is True,
              "s05_top_margin_ge_6_percent": isinstance(st.get("top_margin"), float) and 0.06 <= st["top_margin"] <= 0.08}
    return finish(summary, checks)


def step3():
    rc = sh("run_stage.py", "s06_colab_pack")
    z = ROOT / "colab_stage" / "maximus_v2_colab_bundle.zip"; man = ROOT / "colab_stage" / "bundle" / "MANIFEST.json"
    m = json.loads(man.read_text(encoding="utf-8")) if (rc == 0 and man.exists()) else {}
    summary = {"step": 3, "exit_codes": {"s06_colab_pack": rc}, "zip": str(z), "zip_mb": round(z.stat().st_size / 1e6, 2) if z.exists() else None,
               "files": len(m.get("files", {})), "gates": f"{m.get('gates_passed')}/{m.get('gates_total')}", "master_sha256": m.get("master_sha256"),
               "upload_to": "My Drive/COLAB_HEAVY/maximus_v2/", "then_run": str(ROOT / "colab_stage" / "COLAB_CELL.txt")}
    checks = {"pack_ok": rc == 0 and z.exists(), "bundle_built_from_passing_gates": m.get("gates_all_pass") is True}
    return finish(summary, checks)


if __name__ == "__main__":
    sys.exit({"1": step1, "2": step2, "3": step3}[sys.argv[1]]())
