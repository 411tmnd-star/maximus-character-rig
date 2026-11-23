"""Link Maximus_build to 411_UNIFIED_3D_CORPUS (zero-drift imports).

Maximus_build does not keep its own copies of the shared maths. It imports these packages from the corpus:
  maximus_core       20_3d_rig_topology_and_physics/primitives/owned   geometry, clearance (L_body, L_VE, winding),
                                                                    face binding, egg-head fit, gates
  maximus_delight    30_vision_audio_and_media/primitives/owned AO/normal delighting, sRGB/linear, ORM packing
  corpus_bridge      10_execution_substrate/primitives/owned        EC-02 envelopes, EC-05 atomic writes, EC-04 runner

corpus_lock.json pins the SHA-256 of every imported module file. link() refuses to import if a file drifted.
Regenerate the lock deliberately with:  python -I tools/lock_corpus.py
Works in system Python and in Blender's bundled Python (standard library only).
"""
import hashlib, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = Path(os.environ.get("MAXIMUS_CORPUS", os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/01_code_library'))))
PACKAGES = {
    "maximus_core": "20_3d_rig_topology_and_physics/primitives/owned",
    "maximus_delight": "30_vision_audio_and_media/primitives/owned",
    "corpus_bridge": "10_execution_substrate/primitives/owned",
}
LOCK = HERE / "corpus_lock.json"


def module_files():
    for pkg, base in PACKAGES.items():
        for f in sorted((CORPUS / base / pkg).rglob("*.py")):
            yield pkg, f


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def link(strict=True):
    if not CORPUS.exists():
        raise FileNotFoundError(f"411_UNIFIED_3D_CORPUS not found at {CORPUS} (set MAXIMUS_CORPUS)")
    if LOCK.exists():
        lock = json.loads(LOCK.read_text(encoding="utf-8"))["files"]
        drift = []
        for pkg, f in module_files():
            rel = f.relative_to(CORPUS).as_posix()
            if lock.get(rel) != sha256(f):
                drift.append(rel)
        missing = [r for r in lock if not (CORPUS / r).exists()]
        if (drift or missing) and strict:
            raise RuntimeError(f"corpus drift vs corpus_lock.json: changed={drift} missing={missing}")
    elif strict:
        raise RuntimeError("corpus_lock.json missing; run tools/lock_corpus.py")
    for base in set(PACKAGES.values()):
        p = str(CORPUS / base)
        if p not in sys.path:
            sys.path.insert(0, p)
    return CORPUS


def write_lock():
    files = {f.relative_to(CORPUS).as_posix(): sha256(f) for _, f in module_files()}
    LOCK.write_text(json.dumps({"corpus": str(CORPUS), "files": files}, indent=1), encoding="utf-8")
    return files
