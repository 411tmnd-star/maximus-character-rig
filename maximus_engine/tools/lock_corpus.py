"""Pin the SHA-256 of every corpus module Maximus_build imports (maximus_core, maximus_delight, corpus_bridge).
Run deliberately after reviewing a corpus change:  python -I tools/lock_corpus.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import corpus_link  # noqa: E402

files = corpus_link.write_lock()
print(json.dumps(files, indent=1)); print(f"locked {len(files)} files -> {corpus_link.LOCK}")
