"""Declarative P1/P2/P3 specs (angles from 411chrone.pdf). Shared by s02_build_poses and the diagnostics."""

POSES = {
    "P1": {"file": "theta_monolith.json", "name": "theta_monolith", "title": "Heavy Forward Slouch (Monolith)", "maps_to": "POSE_02",
           "finish": "M1", "ops": [
               {"op": "rotate", "bone": "spine02", "axis": "L", "deg": 4.0, "why": "+4 deg thoracic forward curve"},
               {"op": "rotate", "bone": "head", "axis": "L", "deg": 3.0, "why": "-3 deg chin tuck (head nods down 3 deg)"},
               {"op": "shoulder_roll", "deg": 3.5, "why": "+3.5 deg forward shoulder roll on shoulder01 (clavicle stays 0)"},
               {"op": "arm_abduction", "side": "L", "deg": 5.5, "why": "upper arms dropped to 5.5 deg"},
               {"op": "arm_abduction", "side": "R", "deg": 5.5, "why": "upper arms dropped to 5.5 deg"},
               {"op": "elbow_flex", "side": "L", "deg": 18.0, "why": "18 deg elbow flexion"},
               {"op": "elbow_flex", "side": "R", "deg": 18.0, "why": "18 deg elbow flexion"}]},
    "P2": {"file": "theta_contrapposto.json", "name": "theta_contrapposto", "title": "Weight-Shift S-Curve (Contrapposto)", "maps_to": "POSE_06",
           "finish": "M2", "ops": None},   # filled below by p2_ops(0.0)
    "P3": {"file": "theta_cropped_editorial.json", "name": "theta_cropped_editorial", "title": "High-Fashion Half-Look (Cropped Editorial)", "maps_to": "POSE_05",
           "finish": "M3", "ops": [
               {"op": "rotate", "bone": "spine03", "axis": "U", "deg": 2.0, "why": "+6 deg axial torso twist toward the lens (1/3)"},
               {"op": "rotate", "bone": "spine02", "axis": "U", "deg": 2.0, "why": "+6 deg axial torso twist toward the lens (2/3)"},
               {"op": "rotate", "bone": "spine01", "axis": "U", "deg": 2.0, "why": "+6 deg axial torso twist toward the lens (3/3)"},
               {"op": "arm_flex", "side": "L", "deg": 8.0, "why": "left humerus forward so the forearm can cross to the hem"},
               {"op": "elbow_flex", "side": "L", "deg": 28.0, "medial": 0.8, "why": "28 deg inward forearm flexion toward the hem"},
               {"op": "elbow_flex", "side": "R", "deg": 10.0, "why": "right arm relaxed"}]},
}


def p2_ops(g):
    """P2 op list. The 4 deg pelvic tilt needs the legs turned back 4 deg to stay vertical. g in [0, 1] is the share
    of that counter-rotation carried by the mid-thigh joint (upperleg02) instead of the hip joint (upperleg01).
    g = 0 is the pose as first authored: all 4 deg of relative rotation land on the hip crease, where theta_fit has
    already used up the G5 budget (2 collapsed triangles per side), and the left side goes from 2 to 5.
    g = 1 leaves the hip crease exactly as theta_fit has it and puts a 4 deg bend in the thigh shaft, where the
    triangles are far from collapse. Pelvis tilt, chest counter-rotation and vertical shins are the same for every g."""
    hip, thigh = -4.0 * (1.0 - g), -4.0 * g
    ops = [{"op": "rotate", "bone": "root", "axis": "F", "deg": 4.0, "why": "+4 deg pelvic tilt, left hip up"}]
    for bone, deg in (("upperleg01", hip), ("upperleg02", thigh)):
        if abs(deg) > 1e-9:
            ops += [{"op": "rotate", "bone": f"{bone}.{s}", "axis": "F", "deg": deg, "why": "legs counter-rotated to stay vertical"} for s in "LR"]
    return ops + [
        {"op": "rotate", "bone": "spine02", "axis": "F", "deg": -2.0, "why": "-4 deg chest counter-rotation (half on spine02)"},
        {"op": "rotate", "bone": "spine01", "axis": "F", "deg": -2.0, "why": "-4 deg chest counter-rotation (half on spine01)"},
        {"op": "elbow_flex", "side": "L", "deg": 12.0, "why": "12 deg left elbow"},
        {"op": "elbow_flex", "side": "R", "deg": 16.0, "why": "16 deg right elbow"},
        {"op": "wrist_pronation", "side": "R", "deg": 10.0, "why": "10 deg right wrist pronation"}]


def _ring_relief(deg):
    return {"op": "finger_relief", "fingers": [4], "joints": [2], "deg": -deg,
            "why": f"open the ring-finger middle joint {deg} deg (theta_fit collapses one knuckle triangle per hand there)"}


POSES["P2"]["ops"] = p2_ops(0.0)

# G5 relief ladders, used by the s02 guard only when the spec ops leave more than 6 triangles under 0.25 x rest area.
# Ordered from the smallest departure from the authored pose to the largest; the first variant that passes is kept.
def _with_roll(ops, deg):
    """P1 op list with the forward shoulder roll set to `deg` (0 drops the op)."""
    return [dict(o, deg=deg) if o["op"] == "shoulder_roll" else o for o in ops if not (o["op"] == "shoulder_roll" and deg == 0)]


# P1: the 4 armpit triangles (2622, 2625, 15861, 15862) fall to ~0.23 x rest area under the spec 3.5 deg shoulder roll
# (theta_fit alone: 0.31-0.34). A per-op ablation (20 variants, 2026-11-17) found the roll is the driver; the
# arm abduction is irrelevant because the clearance guard returns the humerus to ~10 deg whatever the spec says (and
# without the guard the spec arm crosses the torso, 5 edges per side). Roll 1.75 deg: weakest armpit triangle 0.26,
# 4 collapsed in total, 0 arm/torso crossings, >= 11 mm clearance. Roll 0: weakest 0.29.
POSES["P1"]["g5_relief"] = (
    [{"label": f"ring_relief_{d}", "ops": POSES["P1"]["ops"] + [_ring_relief(d)], "why": f"ring-finger middle joint opened {d} deg"} for d in (8.0, 16.0)]
    + [{"label": f"shoulder_roll_{r}+ring_relief_8.0", "ops": _with_roll(POSES["P1"]["ops"], r) + [_ring_relief(8.0)],
        "why": f"forward shoulder roll reduced from 3.5 to {r} deg (armpit triangles), ring-finger middle joint opened 8.0 deg"} for r in (1.75, 0.0)])
POSES["P2"]["g5_relief"] = (
    [{"label": f"thigh_share_{g:.2f}", "ops": p2_ops(g), "why": f"{g:.0%} of the leg counter-rotation moved from the hip to the mid-thigh joint"} for g in (0.5, 1.0)]
    + [{"label": f"thigh_share_1.00+ring_relief_{d}", "ops": p2_ops(1.0) + [_ring_relief(d)],
        "why": f"leg counter-rotation on the mid-thigh joint, ring-finger middle joint opened {d} deg"} for d in (8.0, 16.0)])
POSES["P3"]["g5_relief"] = [
    {"label": f"ring_relief_{d}", "ops": POSES["P3"]["ops"] + [_ring_relief(d)], "why": f"ring-finger middle joint opened {d} deg"} for d in (8.0, 16.0)]

# Finger curl: theta_fit already carries a relaxed inward curl (10-35 deg per joint). Extra curl on top collapsed the
# ring/pinky knuckle triangles (119 below 0.25x rest area at 0.8x; tools/diag_collapse.py), so no extra curl is added.
