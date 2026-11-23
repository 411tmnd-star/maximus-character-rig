"""MX-10  Maximus v2 rebuild flow: run every Maximus_build stage in order and stop at the first failing stage.

  [10] corpus_bridge.blender_runner (EC-04)  ->  each Blender stage prints one EC-02 envelope
  s00 extract v1 (read-only, hash 1be2d4ca...)         Blender
  s01 master: egg head (MX-01, MX-03 L_VE -> L_body), M1/M2/M3, LBS + Corrective Smooth     Blender
  s02 P1/P2/P3 poses, clavicles (0,0,0), L_body 3 mm arm guard (MX-02)                      Blender
  s03 gates (MX-02, MX-04, MX-06)                                                           system Python
  s04 bake M1/M2/M3 + Draco core GLB (MX-07 ORM)                                            Blender
  s05 2K previews (P x M) + CSS backdrop composites at 82% bust fade                        Blender + Python
  s06 MAXIMUS_REBUILD_REPORT.md with SHA-256 of every artefact                              system Python

Usage: python -I flow_maximus_v2_rebuild.py [--project ~\\Desktop\\4041_MechMix\\02_engines\\maximus_engine] [--from s02] [--only s03]
Writes <project>/out/reports/flow_run.json (all envelopes, EC-05 atomic).
"""
import argparse, importlib, json, os, sys, time
from pathlib import Path

STAGES = ["s00_extract_v1", "s01_build_master", "s02_build_poses", "s03_gates", "s04_bake_export", "s05_previews", "s06_report"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--project", default=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/maximus_engine')))
    ap.add_argument("--from", dest="start", default="s00"); ap.add_argument("--only"); ap.add_argument("--continue-on-fail", action="store_true")
    a = ap.parse_args()
    proj = Path(a.project).resolve(); sys.path.insert(0, str(proj)); sys.path.insert(0, str(proj / "pipeline"))
    rs = importlib.import_module("run_stage")                 # links the corpus (strict lock unless MAXIMUS_STRICT=0)
    from corpus_bridge import contracts
    todo = [s for s in STAGES if s.startswith(a.only)] if a.only else STAGES[[s[:3] for s in STAGES].index(a.start[:3]):]
    run = {"flow": "MX-10 flow_maximus_v2_rebuild", "project": str(proj), "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "stages": []}
    for s in todo:
        t0 = time.time(); env = rs.run(s); env.setdefault("metrics", {})["flow_seconds"] = round(time.time() - t0, 1)
        run["stages"].append(env); print(f"{s:20s} {env['status']:5s} {env['metrics'].get('flow_seconds')}s", flush=True)
        if env["status"] != "ok" and not a.continue_on_fail:
            break
    run["status"] = "ok" if all(e["status"] == "ok" for e in run["stages"]) and len(run["stages"]) == len(todo) else "fail"
    contracts.atomic_write_json(proj / "out" / "reports" / "flow_run.json", run)
    print("flow", run["status"]); sys.exit(0 if run["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
