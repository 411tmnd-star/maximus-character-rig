"""Rebuild Maximus v2 end to end by calling the corpus flow MX-10 (411_UNIFIED_3D_CORPUS/60_cross_sectional_workflows).
  python -I run_all.py                 all stages (s00..s06), strict corpus lock
  python -I run_all.py --from s04      resume
"""
import subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I does not add the script folder
import corpus_link  # noqa: E402

ROOT = Path(__file__).resolve().parent
flow = corpus_link.CORPUS / "60_cross_sectional_workflows" / "flow_maximus_v2_rebuild.py"
sys.exit(subprocess.call([sys.executable, "-I", str(flow), "--project", str(ROOT)] + sys.argv[1:]))
