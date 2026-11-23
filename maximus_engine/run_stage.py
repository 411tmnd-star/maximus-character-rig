"""Run one Maximus pipeline stage headless through the corpus EC-04 runner and print its EC-02 envelope.

  python -I run_stage.py s00_extract_v1 [blend=<path>] [key=value ...]
Blender stages (s00, s01, s02, s04, s05) run in Blender 5.2; system-Python stages (s03, s06) run in-process.
"""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "pipeline"))
import corpus_link  # noqa: E402

STRICT = os.environ.get("MAXIMUS_STRICT", "1") == "1"
corpus_link.link(strict=STRICT)
from corpus_bridge.blender_runner import run_blender  # noqa: E402

V1 = os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/02_engine_3d/isolated_clones/better_style_catalog_collar/out_fbl_008/inputs/03_full_rigged.blend'))
MASTER = str(ROOT / "out" / "maximus_v2_master.blend")
BLEND_FOR = {"s00_extract_v1": V1, "s01_build_master": V1, "s02_build_poses": MASTER, "s04_bake_export": MASTER, "s05_previews": MASTER}


def run(stage, extra=()):
    kv = dict(a.split("=", 1) for a in extra if "=" in a)
    if stage in BLEND_FOR:
        blend = kv.pop("blend", BLEND_FOR[stage])
        return run_blender(ROOT / "pipeline" / f"{stage}.py", [f"{k}={v}" for k, v in kv.items()], blend=blend,
                           log_path=ROOT / "out" / "logs" / f"{stage}.log", timeout=6 * 3600,
                           env={"MAXIMUS_STRICT": "1" if STRICT else "0"})
    mod = __import__(stage)
    return mod.main(**kv)


if __name__ == "__main__":
    env = run(sys.argv[1], sys.argv[2:])
    print(json.dumps(env, indent=1, default=str)[:6000])
    sys.exit(0 if env["status"] == "ok" else 1)
