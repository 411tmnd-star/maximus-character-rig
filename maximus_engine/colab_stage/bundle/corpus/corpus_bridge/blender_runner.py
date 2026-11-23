"""EC-04 supervised subprocess bridge: run a Blender script headless and return its EC-02 envelope.

  blender.exe --background [file.blend] --factory-startup --python <script.py> -- <args>

--factory-startup keeps user add-ons (for example the BlenderMCP add-on) out of the run, so results do not depend
on this PC's preferences. stdout/stderr are saved next to the outputs for audit.
"""
import os, subprocess, time
from pathlib import Path

from .contracts import envelope, parse_stdout

BLENDER = Path(os.environ.get("BLENDER_EXE", r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"))


def run_blender(script, args=(), blend=None, timeout=3600, log_path=None, env=None):
    cmd = [str(BLENDER), "--background"]
    if blend:
        cmd.append(str(blend))
    cmd += ["--factory-startup", "--python-exit-code", "3", "--python", str(script), "--"] + [str(a) for a in args]
    t0 = time.time()
    e = dict(os.environ); e.update(env or {}); e.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=e)
    dt = round(time.time() - t0, 2)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(f"$ {' '.join(cmd)}\n# exit {proc.returncode}  {dt}s\n\n{proc.stdout}\n--- stderr ---\n{proc.stderr}", encoding="utf-8")
    env_out = parse_stdout(proc.stdout)
    if env_out is None:
        tail = (proc.stdout + proc.stderr)[-3000:]
        return envelope(Path(script).stem, "fail", inputs={"cmd": cmd}, metrics={"seconds": dt, "exit": proc.returncode},
                        errors=[f"no EC-02 envelope; exit={proc.returncode}", tail])
    env_out.setdefault("metrics", {})["seconds_wall"] = dt
    if proc.returncode != 0 and env_out["status"] == "ok":
        env_out["status"] = "fail"; env_out["errors"].append(f"exit code {proc.returncode}")
    return env_out
