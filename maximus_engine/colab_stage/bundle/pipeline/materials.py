"""Procedural M1 / M2 / M3 mineral finishes (Cycles), built with the bpy.data node API only (no bpy.ops).

Texture space: the 'rest_pos' point attribute (rest-pose positions in metres) instead of Generated/Object
coordinates, so grain and seams stay fixed to the surface when the rig poses the mesh.
Joint seams: hairline grooves where a plane through each wrist / elbow joint (normal = limb direction) meets the
limb, limited to arm vertices by the 'seam_region' attribute and to the limb radius. Seams darken the base colour
slightly and cut into the bump, which reads as the display-mannequin joint line in MAXIMUS_LOOK.
"""
import bpy

from bl_common import hex_linear

SEAM_HALF_WIDTH = 0.0004   # 0.8 mm groove
SEAM_SOFT = 0.0008         # soft shoulder of the groove profile
GRAIN = {  # per-finish procedural character (scales are 1/metre on rest_pos)
    "M1": {"col_scale": 140.0, "col_spread": 0.06, "rough_scale": 60.0, "rough_spread": 0.04, "grit_scale": 1400.0, "grit": 0.15, "pits": 0.00, "bump": 0.35},
    "M2": {"col_scale": 120.0, "col_spread": 0.03, "rough_scale": 55.0, "rough_spread": 0.05, "grit_scale": 1100.0, "grit": 0.45, "pits": 0.08, "bump": 0.45},
    "M3": {"col_scale": 90.0, "col_spread": 0.07, "rough_scale": 50.0, "rough_spread": 0.05, "grit_scale": 900.0, "grit": 0.40, "pits": 0.35, "bump": 0.55},
}


def _n(nt, typ, x, y, **props):
    n = nt.nodes.new(typ); n.location = (x, y)
    for k, v in props.items():
        setattr(n, k, v)
    return n


def seam_group(planes):
    """Node group MX_JointSeams: output 'Seam' in [0,1]. planes: list of dicts {point, normal, radius}."""
    g = bpy.data.node_groups.get("MX_JointSeams") or bpy.data.node_groups.new("MX_JointSeams", "ShaderNodeTree")
    g.nodes.clear(); g.interface.clear()
    g.interface.new_socket("Seam", in_out="OUTPUT", socket_type="NodeSocketFloat")
    out = _n(g, "NodeGroupOutput", 1600, 0)
    rest = _n(g, "ShaderNodeAttribute", -800, 0, attribute_name="rest_pos", attribute_type="GEOMETRY")
    region = _n(g, "ShaderNodeAttribute", -800, -300, attribute_name="seam_region", attribute_type="GEOMETRY")
    acc = None
    for i, p in enumerate(planes):
        y = 400 - i * 260
        sub = _n(g, "ShaderNodeVectorMath", -600, y, operation="SUBTRACT"); sub.inputs[1].default_value = p["point"]
        g.links.new(rest.outputs["Vector"], sub.inputs[0])
        dot = _n(g, "ShaderNodeVectorMath", -400, y, operation="DOT_PRODUCT"); dot.inputs[1].default_value = p["normal"]
        g.links.new(sub.outputs["Vector"], dot.inputs[0])
        ab = _n(g, "ShaderNodeMath", -200, y, operation="ABSOLUTE"); g.links.new(dot.outputs["Value"], ab.inputs[0])
        prof = _n(g, "ShaderNodeMapRange", 0, y, clamp=True)
        prof.inputs["From Min"].default_value = SEAM_HALF_WIDTH; prof.inputs["From Max"].default_value = SEAM_HALF_WIDTH + SEAM_SOFT
        prof.inputs["To Min"].default_value = 1.0; prof.inputs["To Max"].default_value = 0.0
        g.links.new(ab.outputs["Value"], prof.inputs["Value"])
        crs = _n(g, "ShaderNodeVectorMath", -400, y - 120, operation="CROSS_PRODUCT"); crs.inputs[1].default_value = p["normal"]
        g.links.new(sub.outputs["Vector"], crs.inputs[0])
        ln = _n(g, "ShaderNodeVectorMath", -200, y - 120, operation="LENGTH"); g.links.new(crs.outputs["Vector"], ln.inputs[0])
        rad = _n(g, "ShaderNodeMapRange", 0, y - 120, clamp=True)
        rad.inputs["From Min"].default_value = p["radius"]; rad.inputs["From Max"].default_value = p["radius"] + 0.01
        rad.inputs["To Min"].default_value = 1.0; rad.inputs["To Max"].default_value = 0.0
        g.links.new(ln.outputs["Value"], rad.inputs["Value"])
        mul = _n(g, "ShaderNodeMath", 200, y, operation="MULTIPLY"); g.links.new(prof.outputs["Result"], mul.inputs[0]); g.links.new(rad.outputs["Result"], mul.inputs[1])
        if acc is None:
            acc = mul
        else:
            mx = _n(g, "ShaderNodeMath", 400 + i * 40, y, operation="MAXIMUM"); g.links.new(acc.outputs[0], mx.inputs[0]); g.links.new(mul.outputs[0], mx.inputs[1]); acc = mx
    reg = _n(g, "ShaderNodeMapRange", 900, -300, clamp=True)
    reg.inputs["From Min"].default_value = 0.35; reg.inputs["From Max"].default_value = 0.65
    g.links.new(region.outputs["Fac"], reg.inputs["Value"])
    fin = _n(g, "ShaderNodeMath", 1200, 0, operation="MULTIPLY"); g.links.new(acc.outputs[0], fin.inputs[0]); g.links.new(reg.outputs["Result"], fin.inputs[1])
    g.links.new(fin.outputs[0], out.inputs["Seam"])
    return g


def finish_material(key, spec, seams):
    name = f"MX_{key}_{spec['name'].replace(' ', '')}"
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True; nt = m.node_tree; nt.nodes.clear(); m.use_fake_user = True
    gp = GRAIN[key]; base = hex_linear(spec["hex"])
    out = _n(nt, "ShaderNodeOutputMaterial", 1400, 0)
    bsdf = _n(nt, "ShaderNodeBsdfPrincipled", 1000, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    rest = _n(nt, "ShaderNodeAttribute", -1200, 0, attribute_name="rest_pos", attribute_type="GEOMETRY")
    seam = _n(nt, "ShaderNodeGroup", -400, -500); seam.node_tree = seams
    # colour variation: two close tones around the spec colour
    nc = _n(nt, "ShaderNodeTexNoise", -800, 300); nc.inputs["Scale"].default_value = gp["col_scale"]; nc.inputs["Detail"].default_value = 6.0
    nt.links.new(rest.outputs["Vector"], nc.inputs["Vector"])
    ramp = _n(nt, "ShaderNodeValToRGB", -550, 300)
    lo = [c * (1 - gp["col_spread"]) for c in base]; hi = [min(1.0, c * (1 + gp["col_spread"])) for c in base]
    ramp.color_ramp.elements[0].position = 0.3; ramp.color_ramp.elements[0].color = (*lo, 1)
    ramp.color_ramp.elements[1].position = 0.7; ramp.color_ramp.elements[1].color = (*hi, 1)
    nt.links.new(nc.outputs["Fac"], ramp.inputs["Fac"])
    dark = _n(nt, "ShaderNodeMix", 600, 300, data_type="RGBA", blend_type="MULTIPLY")
    dark.inputs["B"].default_value = (0.55, 0.55, 0.55, 1)
    fac = _n(nt, "ShaderNodeMath", 300, 450, operation="MULTIPLY"); fac.inputs[1].default_value = 0.8
    nt.links.new(seam.outputs["Seam"], fac.inputs[0]); nt.links.new(fac.outputs[0], dark.inputs["Factor"])
    nt.links.new(ramp.outputs["Color"], dark.inputs["A"]); nt.links.new(dark.outputs["Result"], bsdf.inputs["Base Color"])
    # roughness around the spec value
    nr = _n(nt, "ShaderNodeTexNoise", -800, 0); nr.inputs["Scale"].default_value = gp["rough_scale"]; nt.links.new(rest.outputs["Vector"], nr.inputs["Vector"])
    mr = _n(nt, "ShaderNodeMapRange", -550, 0, clamp=True)
    mr.inputs["To Min"].default_value = spec["roughness"] - gp["rough_spread"]; mr.inputs["To Max"].default_value = spec["roughness"] + gp["rough_spread"]
    nt.links.new(nr.outputs["Fac"], mr.inputs["Value"]); nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])
    # micro relief: fine grit + mineral pits - seam groove
    ng = _n(nt, "ShaderNodeTexNoise", -800, -250); ng.inputs["Scale"].default_value = gp["grit_scale"]; ng.inputs["Detail"].default_value = 2.0
    nt.links.new(rest.outputs["Vector"], ng.inputs["Vector"])
    gm = _n(nt, "ShaderNodeMath", -550, -250, operation="MULTIPLY"); gm.inputs[1].default_value = gp["grit"]; nt.links.new(ng.outputs["Fac"], gm.inputs[0])
    vor = _n(nt, "ShaderNodeTexVoronoi", -800, -700); vor.inputs["Scale"].default_value = 260.0; nt.links.new(rest.outputs["Vector"], vor.inputs["Vector"])
    pit = _n(nt, "ShaderNodeMapRange", -550, -700, clamp=True)
    pit.inputs["From Min"].default_value = 0.0; pit.inputs["From Max"].default_value = 0.12
    pit.inputs["To Min"].default_value = gp["pits"]; pit.inputs["To Max"].default_value = 0.0
    nt.links.new(vor.outputs["Distance"], pit.inputs["Value"])
    h1 = _n(nt, "ShaderNodeMath", -250, -350, operation="SUBTRACT"); nt.links.new(gm.outputs[0], h1.inputs[0]); nt.links.new(pit.outputs["Result"], h1.inputs[1])
    h2 = _n(nt, "ShaderNodeMath", 0, -350, operation="SUBTRACT"); nt.links.new(h1.outputs[0], h2.inputs[0]); nt.links.new(seam.outputs["Seam"], h2.inputs[1])
    bump = _n(nt, "ShaderNodeBump", 600, -300); bump.inputs["Strength"].default_value = gp["bump"]; bump.inputs["Distance"].default_value = 0.0006
    nt.links.new(h2.outputs[0], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    if spec["sss"] > 0:
        bsdf.inputs["Subsurface Weight"].default_value = spec["sss"]
        bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.82, 0.65)
        bsdf.inputs["Subsurface Scale"].default_value = 0.01
    # specular response per finish (socket names looked up, so a renamed socket is skipped instead of raising)
    applied = {}
    for sock, val in (("Specular IOR Level", spec.get("specular")), ("Coat Weight", spec.get("coat")),
                      ("Coat Roughness", spec.get("coat_roughness"))):
        if val is not None and sock in bsdf.inputs:
            bsdf.inputs[sock].default_value = float(val); applied[sock] = float(val)
    m["mx_spec"] = {"hex_srgb": spec["hex"], "base_linear": list(base), "roughness": spec["roughness"], "sss": spec["sss"],
                    "specular_response": applied}
    return m
