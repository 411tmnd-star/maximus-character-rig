"""Stage 06b (system Python): pack everything the Colab L4 batch needs into one verified archive.

    python -I run_stage.py s06_colab_pack            refuses unless out/reports/s03_gates.json says every gate passed
    python -I run_stage.py s06_colab_pack force=1    pack anyway (the manifest records gates_all_pass = false)

Writes
  colab_stage/bundle/                      mirror of Maximus_build that s05 needs: out/maximus_v2_master.blend, pipeline/,
                                           poses/, corpus_link.py, corpus_lock.json and corpus/ (only the locked modules,
                                           so corpus_link verifies the same hashes on Colab in strict mode)
  colab_stage/bundle/MANIFEST.json         SHA-256 of every staged file + the gate summary the bundle was built from
  colab_stage/maximus_v2_colab_bundle.zip  maximus_v2/bundle/..., maximus_v2/run_batch_4k.py, maximus_v2/COLAB_CELL.txt
The zip is the only thing to upload. s06_report.py (composites + MAXIMUS_REBUILD_REPORT.md) is a separate stage.
"""
import json, shutil, zipfile
from pathlib import Path

import corpus_link
from corpus_bridge import contracts

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "colab_stage"; B = STAGE / "bundle"; ZIP = STAGE / "maximus_v2_colab_bundle.zip"
DRIVE = "/content/drive/MyDrive/COLAB_HEAVY/maximus_v2"
CELL = f"""# Colab cell (GPU runtime: L4). Upload maximus_v2_colab_bundle.zip to My Drive/COLAB_HEAVY/maximus_v2/ first.
from google.colab import drive; drive.mount('/content/drive')
!rm -rf /content/maximus_v2 && unzip -q -o {DRIVE}/maximus_v2_colab_bundle.zip -d /content/
!python /content/maximus_v2/run_batch_4k.py --bundle /content/maximus_v2/bundle --samples 512 --out {DRIVE}/results
# Results: {DRIVE}/results/previews_4k/*.png (9 renders), reports/s05_previews_4k.json, RESULTS_4K.json
# Copy previews_4k back into Maximus_build/out/previews_4k, then run:  python -I run_stage.py s06_report
"""


def put(src, rel):
    dst = B / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst); return rel


def main(**kv):
    gates_p = ROOT / "out" / "reports" / "s03_gates.json"
    gates = json.loads(gates_p.read_text(encoding="utf-8")) if gates_p.exists() else None
    all_pass = bool(gates) and gates["status"] == "ok"
    failed = [k for k, v in gates["gates"].items() if not v["pass"]] if gates else ["s03_gates.json missing"]
    if not all_pass and str(kv.get("force", "0")) != "1":
        return contracts.envelope("s06_colab_pack", "fail", errors=[f"gates not all passing: {failed}. Fix them, or pass force=1."])
    master = ROOT / "out" / "maximus_v2_master.blend"
    s01 = json.loads((ROOT / "out" / "reports" / "s01_master_report.json").read_text(encoding="utf-8"))
    if B.exists():
        shutil.rmtree(B)                    # the bundle is a derived folder owned by this stage
    files = [put(master, "out/maximus_v2_master.blend"), put(ROOT / "corpus_link.py", "corpus_link.py"), put(ROOT / "corpus_lock.json", "corpus_lock.json")]
    files += [put(p, f"pipeline/{p.name}") for p in sorted((ROOT / "pipeline").glob("*.py"))]
    files += [put(p, f"poses/{p.name}") for p in sorted((ROOT / "poses").glob("*.json"))]
    lock = json.loads((ROOT / "corpus_lock.json").read_text(encoding="utf-8"))["files"]
    drift = [rel for rel, h in lock.items() if corpus_link.sha256(corpus_link.CORPUS / rel) != h]
    if drift:
        return contracts.envelope("s06_colab_pack", "fail", errors=[f"corpus files differ from corpus_lock.json: {drift}"])
    files += [put(corpus_link.CORPUS / rel, f"corpus/{rel}") for rel in lock]
    manifest = {rel: corpus_link.sha256(B / rel) for rel in files}
    meta = {"files": manifest, "master_sha256": manifest["out/maximus_v2_master.blend"],
            "gates_all_pass": all_pass, "gates_failed": failed if not all_pass else [],
            "gates_passed": sum(v["pass"] for v in gates["gates"].values()) if gates else 0, "gates_total": len(gates["gates"]) if gates else 0,
            "egg_head_tris_facing_inward": s01["egg_fit"].get("head_tris_facing_inward"),
            "render_plan": {"poses": ["P1", "P2", "P3"], "finishes": ["M1", "M2", "M3"], "resolution": [3277, 4096], "samples": 512, "device": "GPU"}}
    (B / "MANIFEST.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (STAGE / "COLAB_CELL.txt").write_text(CELL, encoding="utf-8")
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in files + ["MANIFEST.json"]:
            z.write(B / rel, f"maximus_v2/bundle/{rel}")
        z.write(STAGE / "run_batch_4k.py", "maximus_v2/run_batch_4k.py"); z.write(STAGE / "COLAB_CELL.txt", "maximus_v2/COLAB_CELL.txt")
    with zipfile.ZipFile(ZIP) as z:         # read back: the archive must hold exactly what the manifest lists
        bad = z.testzip(); names = set(z.namelist())
    missing = [rel for rel in files if f"maximus_v2/bundle/{rel}" not in names]
    if bad or missing:
        return contracts.envelope("s06_colab_pack", "fail", errors=[f"archive check failed: corrupt={bad} missing={missing}"])
    return contracts.envelope("s06_colab_pack", "ok",
                              outputs={"zip": str(ZIP), "zip_sha256": contracts.sha256_file(ZIP), "bundle": str(B), "cell": str(STAGE / "COLAB_CELL.txt")},
                              metrics={"files": len(files), "zip_mb": round(ZIP.stat().st_size / 1e6, 2), "gates_all_pass": all_pass,
                                       "master_sha256": meta["master_sha256"]})
