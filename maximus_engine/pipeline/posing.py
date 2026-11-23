"""Declarative pose engine for the native 03_full_rigged rig (Blender data API: pose_bone.matrix, no bpy.ops).

Body frame (rig world = identity, verified in s00): L = +X (model's left), F = -Y (forward), U = +Z (up).
Each op is a small dict; the same list is written to the pose JSON as the declarative spec, next to the resolved
local quaternions. Angles are ABSOLUTE anatomical targets where the brief gives an end state ("upper arms at 5.5 deg",
"18 deg elbow flexion") and DELTAS where it gives a motion ("+4 deg thoracic curve").
"""
import math

import numpy as np
from mathutils import Matrix, Quaternion

L = np.array([1.0, 0.0, 0.0]); F = np.array([0.0, -1.0, 0.0]); U = np.array([0.0, 0.0, 1.0])
AXES = {"L": L, "F": F, "U": U}
CLAVICLES = ("clavicle.L", "clavicle.R")
FINGER_CURL = {1: (8.0, 14.0, 18.0), 2: (10.0, 14.0, 10.0), 3: (16.0, 22.0, 14.0), 4: (22.0, 28.0, 18.0), 5: (28.0, 35.0, 22.0)}  # pose_retarget.py


def rot(axis, deg):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a); t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


class Poser:
    def __init__(self, bpy, rig):
        self.bpy = bpy; self.rig = rig; self.pb = rig.pose.bones; self.log = []

    def update(self):
        self.bpy.context.view_layer.update()

    def J(self, name, tail=False):
        b = self.pb[name]; return np.array(b.tail if tail else b.head, float)

    def rot_about_head(self, name, R):
        pb = self.pb[name]; h = pb.head.copy()
        M = Matrix(np.asarray(R).tolist()).to_4x4(); T = Matrix.Translation(h)
        pb.matrix = T @ M @ T.inverted() @ pb.matrix; self.update()

    def align(self, name, v_from, v_to):
        a = v_from / np.linalg.norm(v_from); b = v_to / np.linalg.norm(v_to)
        ax = np.cross(a, b); s = np.linalg.norm(ax)
        if s < 1e-10:
            return 0.0
        ang = math.degrees(math.atan2(s, float(a @ b))); self.rot_about_head(name, rot(ax, ang)); return ang

    # ----- base
    def reset(self):
        for b in self.pb:
            b.rotation_mode = "QUATERNION"; b.matrix_basis.identity()
        self.update()

    def apply_local(self, bones):
        for n, v in bones.items():
            b = self.pb.get(n)
            if b is None:
                continue
            b.rotation_mode = "QUATERNION"; b.rotation_quaternion = Quaternion(v["q_wxyz"]); b.location = v.get("t", (0, 0, 0))
        self.update()

    def lock_clavicles(self):
        for n in CLAVICLES:
            b = self.pb[n]; b.rotation_mode = "QUATERNION"; b.rotation_quaternion = (1, 0, 0, 0); b.location = (0, 0, 0); b.scale = (1, 1, 1)
        self.update()

    # ----- ops
    def op(self, o):
        k = o["op"]; self.log.append(dict(o))
        if k == "rotate":                           # delta about a body axis through the bone head
            self.rot_about_head(o["bone"], rot(AXES[o["axis"]], o["deg"]))
        elif k == "shoulder_roll":                  # protraction: lateral end of shoulder01 moves forward
            for s, sx in (("L", 1), ("R", -1)):
                if s in o.get("sides", "LR"):
                    self.rot_about_head(f"shoulder01.{s}", rot(U, -sx * o["deg"]))
        elif k == "arm_abduction":                  # absolute frontal-plane angle of the humerus from vertical
            s = o["side"]; sx = 1 if s == "L" else -1
            h = self.J(f"lowerarm01.{s}") - self.J(f"upperarm01.{s}")
            q = math.hypot(h[0], h[2]); a = math.radians(o["deg"])
            o["resolved_from_deg"] = round(math.degrees(math.atan2(sx * h[0], -h[2])), 3)
            self.align(f"upperarm01.{s}", h, np.array([sx * math.sin(a) * q, h[1], -math.cos(a) * q]))
        elif k == "arm_flex":                       # delta: humerus swings forward (toward F) about the shoulder
            s = o["side"]; h = self.J(f"lowerarm01.{s}") - self.J(f"upperarm01.{s}")
            p = F - (F @ h) / (h @ h) * h; ax = np.cross(h, p); self.rot_about_head(f"upperarm01.{s}", rot(ax, o["deg"]))
        elif k == "elbow_flex":                     # absolute elbow flexion; plane toward F (+ optional medial)
            s = o["side"]; sx = 1 if s == "L" else -1
            h = self.J(f"lowerarm01.{s}") - self.J(f"upperarm01.{s}"); f = self.J(f"wrist.{s}") - self.J(f"lowerarm01.{s}")
            hn = h / np.linalg.norm(h)
            o["resolved_from_deg"] = round(math.degrees(math.acos(np.clip(hn @ (f / np.linalg.norm(f)), -1, 1))), 3)
            D = F + o.get("medial", 0.0) * (-sx * L); p = D - (D @ hn) * hn; p /= np.linalg.norm(p)
            t = math.radians(o["deg"]); target = math.cos(t) * hn + math.sin(t) * p
            self.align(f"lowerarm01.{s}", f, target * np.linalg.norm(f))
        elif k == "wrist_pronation":                # rotate hand about the forearm axis; thumb moves medially
            s = o["side"]; sx = 1 if s == "L" else -1
            ax = self.J(f"wrist.{s}") - self.J(f"lowerarm01.{s}")
            thumb0 = self.J(f"finger1-3.{s}", tail=True)[0]
            self.rot_about_head(f"wrist.{s}", rot(ax, o["deg"]))
            if sx * self.J(f"finger1-3.{s}", tail=True)[0] > sx * thumb0:      # moved laterally: wrong sign
                self.rot_about_head(f"wrist.{s}", rot(ax, -2 * o["deg"]))
        elif k == "finger_curl":                    # extra flexion on top of the base pose (local X, + = curl)
            for s in o.get("sides", "LR"):
                for fi, curls in FINGER_CURL.items():
                    for j, deg in enumerate(curls, start=1):
                        b = self.pb.get(f"finger{fi}-{j}.{s}")
                        if b:
                            b.rotation_mode = "QUATERNION"
                            b.rotation_quaternion = b.rotation_quaternion @ Quaternion((1, 0, 0), math.radians(deg * o["scale"]))
            self.update()
        elif k == "finger_relief":                  # delta on local X of the listed joints (+ = curl, - = open), as finger_curl
            for s in o.get("sides", "LR"):
                for fi in o["fingers"]:
                    for j in o["joints"]:
                        b = self.pb.get(f"finger{fi}-{j}.{s}")
                        if b:
                            b.rotation_mode = "QUATERNION"
                            b.rotation_quaternion = b.rotation_quaternion @ Quaternion((1, 0, 0), math.radians(o["deg"]))
            self.update()
        else:
            raise KeyError(k)

    def ground(self, eval_positions, iters=4, target=2e-6):
        """Translate root so the lowest evaluated vertex sits on the floor: z in [0, 0.01 mm] (never below 0)."""
        for _ in range(iters):
            z = float(eval_positions()[:, 2].min())
            if 0.0 <= z <= 1e-5:
                break
            pb = self.pb["root"]; M = pb.matrix.copy(); M.translation.z -= (z - target); pb.matrix = M; self.update()
        return float(eval_positions()[:, 2].min())

    def local_quats(self):
        out = {}
        for b in self.pb:
            q = b.rotation_quaternion if b.rotation_mode == "QUATERNION" else b.matrix_basis.to_quaternion()
            t = b.location
            if abs(q.w - 1) > 1e-9 or any(abs(x) > 1e-12 for x in t):
                out[b.name] = {"q_wxyz": [round(x, 9) for x in q], "t": [round(x, 9) for x in t]}
        return out
