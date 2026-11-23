"""Shared pose library for M4 / M5 / M6 and the runtime_v3 GPU checks (Blender side; bpy.data / pose-bone matrices only, no bpy.ops).

The 16 static poses every Maximus v2 gate is run on:

  theta_fit                 avatar_engine/poses/theta_fit.json local quaternions, grounded (the locked base pose)
  P1, P2, P3                the editorial layer (poses/theta_*.json, authored by s02 on theta_fit; root translation stored in the file)
  POSE_01 .. POSE_10        the canonical library (vendor/canonical_poses.json, sha 856fc331...) applied with the ENGINE's own retargeter
                            (vendor/pose_retarget.py, byte-identical to human_engine/src_catalog/.../engine/components/pose_retarget.py):
                            NEUTRAL stance, full anatomical angles, hand contacts by 2-bone IK, pelvis offset, finger curl profile.
                            One deliberate change, made BEFORE retargeting: clavicle_L / clavicle_R angles are dropped from the spec, because
                            the Maximus invariant locks both clavicles at (0, 0, 0) on every static pose. Dropping them first (instead of zeroing
                            afterwards) keeps the child chain and the IK hand contacts consistent with the locked clavicle.
  arm_elev_90, arm_elev_120 theta_fit, clavicles locked, both humeri raised to 90 / 120 deg from vertical in the frontal plane (posing.Poser
                            'arm_abduction', the s02 operator), grounded. No finger relief or other G5 dodge is applied to any pose.

Every pose is grounded so the lowest evaluated vertex sits in [0, 0.01 mm] (Poser.ground), except P1-P3 whose stored root translation already
grounds them (s02/s03 measured 0.002 mm).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P_FILES = {"P1": "theta_monolith.json", "P2": "theta_contrapposto.json", "P3": "theta_cropped_editorial.json"}
ALL_POSES = ["theta_fit", "P1", "P2", "P3"] + [f"POSE_{i:02d}" for i in range(1, 11)] + ["arm_elev_90", "arm_elev_120"]
CLAVICLE_JOINTS = ("clavicle_L", "clavicle_R")


def _vendor(name):
    for base in (HERE / "vendor", HERE.parent / "colab_stage" / "vendor"):
        if (base / name).exists():
            return base / name
    raise FileNotFoundError(f"vendor/{name} not found next to {HERE}")


def canonical_specs():
    return json.loads(_vendor("canonical_poses.json").read_text(encoding="utf-8"))["poses"]


def retargeter_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pose_retarget_vendor", str(_vendor("pose_retarget.py")))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class PoseLibrary:
    """apply(pid) poses the rig in place and returns a small facts dict. `poser` is pipeline/posing.Poser, `ev` returns evaluated world
    vertices (bl_common.evaluated_positions(body)), `root` is the folder that holds poses/theta_*.json (bundle root or project root)."""

    def __init__(self, rig, poser, ev, fit_bones, root):
        self.rig = rig; self.P = poser; self.ev = ev; self.fit = fit_bones; self.root = Path(root)
        self.specs = canonical_specs(); self.RT = retargeter_module(); self._rt = None

    def _retargeter(self):
        if self._rt is None:
            self._rt = self.RT.Retargeter(self.rig)
        return self._rt

    def clavicle_quats(self):
        return {n: [round(float(x), 12) for x in self.rig.pose.bones[n].matrix_basis.to_quaternion()] for n in ("clavicle.L", "clavicle.R")}

    def apply(self, pid):
        P = self.P; facts = {"pose": pid}
        if pid == "theta_fit":
            P.reset(); P.apply_local(self.fit); facts["z_sole_m"] = P.ground(self.ev)
        elif pid in P_FILES:
            P.reset(); P.apply_local(json.loads((self.root / "poses" / P_FILES[pid]).read_text(encoding="utf-8"))["bones"])
            facts["z_sole_m"] = float(self.ev()[:, 2].min())
        elif pid.startswith("arm_elev_"):
            deg = float(pid.rsplit("_", 1)[1])
            P.reset(); P.apply_local(self.fit); P.lock_clavicles()
            for s in ("L", "R"):
                P.op({"op": "arm_abduction", "side": s, "deg": deg})
            facts["z_sole_m"] = P.ground(self.ev)
        elif pid in self.specs:
            spec = json.loads(json.dumps(self.specs[pid]))
            dropped = {j: spec["bone_rotations_deg"].pop(j) for j in CLAVICLE_JOINTS if j in spec["bone_rotations_deg"]}
            self._retargeter().apply(spec, t=1.0)
            P.lock_clavicles()                      # already identity (reset + no clavicle entry); asserts the invariant explicitly
            facts["clavicle_angles_dropped_deg"] = dropped
            facts["hand_contacts"] = {s: c["target"] for s, c in (spec.get("hand_contacts") or {}).items()}
            facts["z_sole_m"] = P.ground(self.ev)
        else:
            raise KeyError(pid)
        facts["clavicles"] = self.clavicle_quats()
        facts["clavicle_lock_ok"] = all(abs(q[0] - 1) < 1e-9 and max(abs(x) for x in q[1:]) < 1e-9 for q in facts["clavicles"].values())
        return facts

    def local_quats(self, names):
        """matrix_basis rotation of each named pose bone as (w, x, y, z), canonical hemisphere w >= 0."""
        out = np.zeros((len(names), 4))
        for i, n in enumerate(names):
            q = np.array(self.rig.pose.bones[n].matrix_basis.to_quaternion(), float)
            out[i] = -q if q[0] < 0 else q
        return out
