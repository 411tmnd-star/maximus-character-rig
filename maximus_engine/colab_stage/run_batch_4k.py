"""Colab 4K batch renderer for Maximus v2 (self-contained; run in a Colab cell with a GPU runtime, e.g. L4).

    !unzip -q -o /content/drive/MyDrive/COLAB_HEAVY/maximus_v2/maximus_v2_colab_bundle.zip -d /content/
    !python /content/maximus_v2/run_batch_4k.py --bundle /content/maximus_v2/bundle --out /content/drive/MyDrive/COLAB_HEAVY/maximus_v2/results

1. verifies bundle/MANIFEST.json (SHA-256 of every staged file; refuses on any mismatch, and refuses a bundle that was
   packed with failing gates unless --allow-failed-gates is given)
2. installs the shared libraries headless Blender needs and downloads Blender 5.2.3 (Linux x64) into /content/blender
3. renders P1-P3 x M1-M3 at 3277x4096 with Cycles on the GPU (OptiX, else CUDA), 512 samples + denoise, via the same
   s05_previews.py stage (strict corpus lock, operators only through io_ops_gateway). Framing and the M1 exposure trim
   are solved inside s05, so the 4K frames match the verified 2K preview.
4. writes bundle/out/previews_4k/*.png and bundle/out/reports/s05_previews_4k.json, then RESULTS_4K.json with hashes,
   and copies all of it to --out (a Drive folder) when given
Copy previews_4k back into Maximus_build/out/previews_4k and run s06_report locally to composite + report.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, tarfile, urllib.request
from pathlib import Path

BLENDER_URL = "https://download.blender.org/release/Blender5.2/blender-5.2.3-linux-x64.tar.xz"
APT_LIBS = ["libxi6", "libxxf86vm1", "libxfixes3", "libxrender1", "libgl1", "libxkbcommon0", "libsm6", "libegl1"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def blender(dest=Path("/content/blender")):
    exe = next(dest.glob("blender-*/blender"), None)
    if exe:
        return exe
    if shutil.which("apt-get"):                      # best effort: a missing library shows up in the stage log
        subprocess.run(["apt-get", "install", "-y", "-qq"] + APT_LIBS, capture_output=True, text=True)
    dest.mkdir(parents=True, exist_ok=True); tar = dest / "blender.tar.xz"
    print("downloading", BLENDER_URL); urllib.request.urlretrieve(BLENDER_URL, tar)
    with tarfile.open(tar) as t:
        t.extractall(dest)
    return next(dest.glob("blender-*/blender"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bundle", required=True); ap.add_argument("--samples", default="512")
    ap.add_argument("--rx", default="3277"); ap.add_argument("--ry", default="4096"); ap.add_argument("--poses", default="P1,P2,P3")
    ap.add_argument("--finishes", default="M1,M2,M3"); ap.add_argument("--blender"); ap.add_argument("--out")
    ap.add_argument("--device", default="GPU"); ap.add_argument("--allow-failed-gates", action="store_true")
    a = ap.parse_args(); B = Path(a.bundle).resolve()
    man = json.loads((B / "MANIFEST.json").read_text())
    bad = [r for r, h in man["files"].items() if not (B / r).exists() or sha256(B / r) != h]
    if bad:
        sys.exit(f"bundle hash mismatch: {bad}")
    if not man.get("gates_all_pass", False) and not a.allow_failed_gates:
        sys.exit(f"bundle was packed with failing gates {man.get('gates_failed')}; rebuild it or pass --allow-failed-gates")
    exe = Path(a.blender) if a.blender else blender()
    out_dir = B / "out" / "previews_4k"
    env = dict(os.environ, MAXIMUS_CORPUS=str(B / "corpus"), MAXIMUS_STRICT="1")
    cmd = [str(exe), "--background", str(B / "out" / "maximus_v2_master.blend"), "--factory-startup", "--python-exit-code", "3",
           "--python", str(B / "pipeline" / "s05_previews.py"), "--", f"samples={a.samples}", f"rx={a.rx}", f"ry={a.ry}", f"device={a.device}",
           "tag=4k", f"out_dir={out_dir}", f"poses={a.poses}", f"finishes={a.finishes}"]
    print(" ".join(cmd)); p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (B / "out" / "logs").mkdir(parents=True, exist_ok=True); (B / "out" / "logs" / "s05_4k.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    env_line = [l for l in p.stdout.splitlines() if l.startswith("EC02 ")]
    res = json.loads(env_line[-1][5:]) if env_line else {"status": "fail", "errors": ["no EC-02 envelope", p.stderr[-2000:]]}
    results = {f.name: sha256(f) for f in sorted(out_dir.glob("*.png"))}
    (B / "out" / "RESULTS_4K.json").write_text(json.dumps({"envelope": res, "renders": results, "master_sha256": man.get("master_sha256")}, indent=1))
    if a.out:                                        # persist on Drive: the /content disk is lost when the runtime stops
        dst = Path(a.out); dst.mkdir(parents=True, exist_ok=True)
        for sub in ("previews_4k", "reports", "logs"):
            if (B / "out" / sub).exists():
                shutil.copytree(B / "out" / sub, dst / sub, dirs_exist_ok=True)
        shutil.copy2(B / "out" / "RESULTS_4K.json", dst / "RESULTS_4K.json")
    print(res["status"], len(results), "renders", "->", a.out or out_dir)
    if res["status"] != "ok":
        print(json.dumps(res.get("errors"), indent=1)[:4000])
    sys.exit(0 if res["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
