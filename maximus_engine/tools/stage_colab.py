"""Build the Colab bundle + archive. Thin wrapper kept for the old command; the work is in pipeline/s06_colab_pack.py.
    python -I tools/stage_colab.py [force=1]        same as:  python -I run_stage.py s06_colab_pack [force=1]
"""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "pipeline"))
import corpus_link  # noqa: E402

corpus_link.link(strict=os.environ.get("MAXIMUS_STRICT", "1") == "1")
import s06_colab_pack  # noqa: E402

if __name__ == "__main__":
    env = s06_colab_pack.main(**dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x))
    print(json.dumps(env, indent=1, default=str)); sys.exit(0 if env["status"] == "ok" else 1)
