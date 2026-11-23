"""EC-02 result envelopes, EC-05 atomic writes and EC-07 hashing. Standard library only (system and Blender Python).

An EC-02 envelope is one JSON object on one line, printed to stdout prefixed with "EC02 " so a supervisor (EC-04)
can find it among Blender's log lines:
  {"schema": "411.ec02/1", "stage": ..., "status": "ok"|"fail", "inputs": {...}, "outputs": {...},
   "metrics": {...}, "errors": [...], "utc": ...}
"""
import datetime, hashlib, json, os, tempfile
from pathlib import Path

SCHEMA = "411.ec02/1"
PREFIX = "EC02 "


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def envelope(stage, status="ok", inputs=None, outputs=None, metrics=None, errors=None):
    if status not in ("ok", "fail"):
        raise ValueError(status)
    return {"schema": SCHEMA, "stage": stage, "status": status, "inputs": inputs or {}, "outputs": outputs or {},
            "metrics": metrics or {}, "errors": list(errors or []),
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def validate(env):
    req = {"schema", "stage", "status", "inputs", "outputs", "metrics", "errors", "utc"}
    missing = req - set(env)
    if missing or env.get("schema") != SCHEMA or env["status"] not in ("ok", "fail"):
        raise ValueError(f"invalid EC-02 envelope: missing={sorted(missing)} schema={env.get('schema')}")
    return env


def emit(env):
    """Print the envelope as one line (EC-02)."""
    print(PREFIX + json.dumps(validate(env), separators=(",", ":"), default=str), flush=True)


def parse_stdout(text):
    """Last EC-02 envelope found in a process's stdout, or None."""
    found = None
    for line in text.splitlines():
        if line.startswith(PREFIX):
            found = json.loads(line[len(PREFIX):])
    return validate(found) if found else None


def atomic_write_bytes(path, data):
    """EC-05: write to a temp file in the same folder, fsync, then os.replace (no half-written files)."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def atomic_write_text(path, text):
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path, obj, indent=1):
    return atomic_write_text(path, json.dumps(obj, indent=indent, default=str) + "\n")
