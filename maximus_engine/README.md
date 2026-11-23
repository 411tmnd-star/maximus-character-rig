# Maximus_build

> **2026-11-16: Maximus v2 has been rebuilt.** Read `MAXIMUS_REBUILD_REPORT.md` first. Rebuild everything with `python -I run_all.py`, which runs the corpus flow MX-10 in `411_UNIFIED_3D_CORPUS`.
>
> | Path | Contents |
> |---|---|
> | `out/maximus_v2_master.blend` | the master |
> | `out/maximus_v2_core.glb` | the Draco core GLB |
> | `poses/` | P1, P2, P3 and the mapping to the canonical poses |
> | `out/textures/` | M1, M2 and M3 bakes |
> | `out/previews/` | 2K RGBA renders plus storefront composites |
> | `pipeline/` | stage scripts; `io_ops_gateway.py` is the only file allowed to use `bpy.ops` |
> | `colab_stage/` | the 4K batch |
>
> The sections below are the original pre-rebuild analysis and are kept for history.

Project folder for **Maximus v2**: a new, visually refined version of the canonical 152-bone mannequin, built *from* the current one without editing it.

Status: **analysis / pre-handoff.** A deeper-analysis model reviews this folder first (see `HANDOFF_DEEP_ANALYSIS.md`), then work is handed to a GPU for baking and rendering.

---

## 1. Goal

Make the mannequin in the storefront renders look like the **target renders** (`refs/target/`) instead of the **current renders** (`refs/current/`).

- Priorities, in the owner's order: **1) visual look, 2) pose, 3) flexibility.**
- The owner stated poses are *not* the main problem. The **mannequin surface/texture** is.
- Approach decided: **do not edit the current Maximus.** Build a new mannequin (Maximus v2) that copies the vertex layout / inner structure (rig, bones, weights, body volume) from the current one, then fully redo the visual layer.
- Compute and rendering should run **on a GPU** (see section 5; the local machine has no usable GPU).

## 2. The gap (target vs. current)

| | Target (PDF renders) | Current (storefront screenshots) |
|---|---|---|
| Head | Faceless egg: no nose, mouth, ears | Bald avatar with brow, nose, lips, ears |
| Surface | Matte, visible pits/grain (clearest on the travertine shots) | Flat, uniform, smooth |
| M1 Obsidian Noir `#18181B` | Near-black, soft sheen | Mid-grey (likely a hex-as-linear colour bug; unverified) |
| M2 Alabaster `#F2EFE9` / M3 Travertine `#D8CEBE` | M2 chalk-white; M3 warm cream with pores | Both read as near-identical pale white/grey; M3 is not warm |
| Joints | Hairline seams at wrists, upper arms, neck (display-mannequin look) | None |
| Hands | Relaxed, curled fingers | Splayed "starfish" fingers |
| Garment (polo) | Burgundy, set-in sleeves, small left-chest collegiate "69" with stitched outline, soft rugby collar with buttoned placket, pique texture | Bright red, raglan seams, large centred "69" in a condensed serif with drop shadow, flat collar with plain strip, no buttons |
| Viewer bug | | All three panels show the same caption "FRONT, ON THE TRAVERTINE WARM MANNEQUIN" regardless of finish |

## 3. Key findings so far

1. **The current renders come from a real-time viewer** (carousel UI; the agentic PDF describes a WebGL/glTF pipeline). Procedural Blender nodes (Noise/Voronoi) **do not export to glTF**. The look must be **baked** (albedo, roughness, normal, AO) onto the new mannequin's UVs.
2. **The target images are probably AI-generated reference art** (the Gemini spec PDF treats them as a design spec with CSS backdrops). The claim in the Google AI Mode PDF that they "look like Cycles renders" is unsupported. Use them as art direction, not as proof of a render setup.
3. **Colour space bug risk.** Any script that feeds `#RRGGBB` straight into a Blender colour socket treats sRGB as linear: `#18181B` renders roughly `#57`, and light finishes wash out. Convert sRGB to linear first. (The Google AI Mode script does this; the original chat scripts did not.)
4. **Procedural noise should use Object coordinates**, not the default Generated coordinates (these stretch across a tall bounding box). Grain at scale ~400 is sub-pixel at full-body framing and only reads in close-ups.
5. **Don't voxel-remesh the old mesh**: it destroys UVs and vertex groups. Build a clean new base mesh, Shrinkwrap it to v1, and Data-Transfer the 152-bone weights (nearest-face interpolated).
6. **Keep garments fitting**: outside head and hands, v2's torso/arm surface should stay within about 2 mm of v1, so existing garment fits (e.g. BS-POLO-001) don't need redoing.
7. **Google AI Mode script in the how-to PDF is not runnable as printed**: the PDF print stripped all bracketed indices (`inputs[0]` became `inputs`) and truncated the right edge of lines. The original script text is needed.
8. **Unverified / low confidence** (from the first long pasted chat, which is *not saved on disk*): the Gaussian-splat research section (accounts, arXiv IDs, "official repo" code excerpts, hyperparameters). It is largely irrelevant to this goal. One internal inconsistency: eps_body 3 mm in the text vs `eps=1e-3` in the pasted code.
9. Possible trademark issue: the green "Monogram Camp Shirt" (target_05) shows an interlocking-GG jacquard that resembles Gucci's monogram. Check before commercial use.

## 4. Plan for Maximus v2

1. **Non-destructive base:** new clean quad mesh, Shrinkwrap to v1, copy the 152-bone armature, bone names and vertex groups, then transfer weights.
2. **Visual layer from scratch:** faceless egg head; wrist/upper-arm/neck seams cut as grooves in one continuous mesh (not separate parts); relaxed curled-finger hands; clean 4K UVs.
3. **Fit check:** automatic distance check v2 vs v1 (about 2 mm on torso/arms).
4. **Bake the mineral PBR set** for M1, M2, M3 (albedo, roughness, normal, AO) from procedural shaders, then export GLB.
5. **Viewer:** IBL plus a raking key light so the grain shows; fix the repeated-caption bug; check M1 is not treated as linear.
6. **Garment alignment (if the target look is wanted):** burgundy, set-in sleeves, small left-chest "69", rugby collar. Confirm with the owner whether the red raglan version is meant to stay.
7. **Tests:** POSE_01, POSE_06, POSE_02 first (the spec's "safest" poses): no armpit collapse, no weight stretching. Then compare side by side with `refs/target/`.

## 5. Environment facts (checked 2026-11-16)

- Blender **5.2** at `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`.
- Local GPU: **Intel UHD Graphics, ~1 GB shared**: not usable for Cycles renders or bakes. RAM 15.8 GB. No NVIDIA GPU found.
- **GPU must come from elsewhere** (cloud GPU, Colab, or another machine). The Colab MCP server failed to connect this session. Not yet decided by the owner.
- Blender MCP was **not connected** (addon not running), so the live scene and the real Maximus mesh/rig have **not been inspected**.
- Location of the v1 Maximus file(s) and `canonical_poses.json` is **unknown** (a disk search timed out). Needed from the owner.
- Plan for GPU stage: headless `blender -b` with OptiX/CUDA, one pose per GPU, cloth baked first.

## 6. File index

### `refs/target/`: look to match (from the Gemini poses PDF)
Pose labels are *likely* matches inferred from the spec, not confirmed.

| File | Content |
|---|---|
| `target_01_denim_overshirt_M2_likely_POSE_10.jpg` | Denim overshirt, white mannequin, both hands at pockets |
| `target_02_ringer_tee_BS-69_M2_likely_POSE_06.jpg` | Cream ringer tee "BS-69", hand at hem |
| `target_03_burgundy_polo_69_M3_hand_at_hem.jpg` | Burgundy polo, warm porous mannequin: **best view of the pitted stone surface** |
| `target_04_black_plaid_camp_M1_likely_POSE_07.jpg` | Black tartan camp shirt, black mannequin, hand at neck |
| `target_05_green_monogram_camp_M2_likely_POSE_04.jpg` | Green monogram camp shirt, hand at collar |
| `target_06_black_polo_69_M1_likely_POSE_02.jpg` | Black polo "69", black mannequin, slouch |
| `target_07_burgundy_polo_walking_M3_likely_POSE_03.jpg` | Burgundy polo plus trousers, barefoot walking stride |
| `target_08_back_view_Betterstyle_M1_likely_POSE_09.jpg` | Back view, script "Betterstyle" plus horse-and-jockey graphic |
| `target_09_..._near_duplicate_of_03.jpg` | Same image as 03 (not byte-identical) |

### `refs/current/`: what we have now (storefront viewer)
- `current_01_three_panel_storefront_viewer.png`: three finishes (dark grey / white / white) with caption bar and thumbnails
- `current_02_three_panel_with_side_thumbnails.png`: same, wider layout

### `docs/`
- `canonical_poses_spec_10_poses.pdf`: **Spec.** The 10 poses, finishes M1/M2/M3 with hex values, backdrops, bone angles, garment pairings, production matrix. Text in `extracted_text/`.
- `google_ai_mode_mannequin_texture_howto.pdf`: Chat transcript with screenshots of the current renders, material recipe, remesh/seam steps, lighting, and (truncated) Blender scripts. Text in `extracted_text/`.
- `3D_Garment_Agentic_Workflow_webgl_collar_rhinestones.pdf`: The WebGL/glTF pipeline: collar-seating fix (13.6 mm float, KDTree snap, weight transfer), rhinestone instancing, Draco/quantization settings. Shows what the viewer's asset pipeline expects.
- `perplexity_thread_goal_priorities_gpu.md`: Priorities (look > pose > flexibility), Maximus definition, "one mesh vs many", GPU batch-render advice.
- `extracted_text/*.txt`: plain-text extractions of the three PDFs (bracketed indices missing in code blocks).

### `tools/`
- `extract_pdf.py`: PyMuPDF text extractor used for the above (`python extract_pdf.py in.pdf out.txt`).

## 7. Canonical values (from the spec PDF)

| Finish | Base | Roughness | Subsurface | Backdrop (radial) |
|---|---|---|---|---|
| M1 Obsidian Noir | `#18181B` | 0.68 | 0 | Warm Architectural Stone `#f4f1ea` to `#cbc4b8` |
| M2 Alabaster Chalk | `#F2EFE9` | 0.75 | 0.05 | Deep Charcoal Gallery `#34322f` to `#151413` |
| M3 Travertine Warm | `#D8CEBE` | 0.80 | 0.02 | Muted Putty / Limestone `#e6e0d6` to `#b5ab9c` |

Other spec numbers: 152-bone rig; clavicles locked at 0.0 deg pitch for shirt poses; upper arms lowered 5.5 to 9.0 deg; 81.5% thigh fade on bust shots; wrist seam bevel about 0.5 mm.
