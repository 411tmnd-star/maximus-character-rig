
## 4. Deviations, failures and why

1. **G5 collapsed triangles: the threshold is unchanged (≤ 6 under 0.25× rest area); the gate table above is the measured result.**
   - **The base pose is already at the limit.** The locked base pose `theta_fit.json` gives 6 collapsed triangles by itself (hip crease on spine05, and the ring-finger middle phalanx). Every editorial pose built on it starts at the ≤ 6 limit.
   - **P1: the Corrective Smooth mask no longer reaches the armpit fold.** The first mask (`clip(1.5·s)` cut at 0.01) had a 1–5 % skirt over the inner-arm fold and collapsed 3 extra triangles per side there. The mask is now `smoothstep` on the shoulder weight times a `smoothstep` of geodesic distance from the upperarm02 fold (zero inside 20 mm, full beyond 60 mm), so the fold stays pure LBS. Parameters are in `s01_master_report.json → corrective_smooth.mask`.
   - **P2: the leg counter-rotation can move off the hip crease.** The 4° pelvic tilt was undone entirely at the hip joints, which pushed the left hip crease from 2 to 5 collapsed triangles. `s02` now measures G5 while it authors each pose and, only when the authored ops exceed the limit, tries the declared relief variants in `pose_specs.py` (share of the counter-rotation moved to the mid-thigh joint; ring-finger middle joint opened 8° or 16°). The variant chosen and every trial are recorded in each pose JSON under `checks.g5_guard` and in `s03_gates.json → report.<pose>.g5_breakdown`.
   - **Not done:** dual-quaternion skinning, `object.correctivesmooth_bind`, and any edit of `theta_fit.json` or of the bone weights.
2. **P1 arms open slightly wider than 5.5° to avoid the ribcage (3 mm L_body rule).** 411chrone warns about exactly this ("verify sleeve-to-ribcage clearance at 5.5°"). At 5.5° the inner upper arms crossed the ribcage triangles, so the clean-room L_body guard opened each arm in 0.5° steps until there were 0 crossings and every free arm vertex was ≥ 3 mm off the torso. The added angles are in the pose table, and also in each pose JSON under `declarative_ops` / `clearance_guard`.
3. **No extra finger curl.** `theta_fit` already carries a relaxed inward curl. Adding 411chrone's curl on top collapsed 119 knuckle triangles (`tools/diag_collapse.py`), so the P layer keeps `theta_fit`'s fingers.
4. **"Refined facial planes and ears" was not built.** MAXIMUS_LOOK (and the earlier failed M-01B head) set the look: a faceless egg with no ears, and that is what was built.
5. **P2 contrapposto needed a 16.3° right-knee bend.** The 4° pelvic tilt lifts the left foot by about 15 mm. The free leg bends until both soles sit on z = 0, which is solved and recorded per pose.
6. **Gaussian-Garments ideas are clean-room.** L_body (ε = 3 mm, cubic), L_VE (virtual edges by inward ray casts, first half of the egg optimisation), face binding μ′ = kRμ + τ and AO/normal delighting were written from the paper's formulas in `411_UNIFIED_3D_CORPUS` (`maximus_core`, `maximus_delight`), with 14 unit tests. No code from the unlicensed repository was used.
7. **Light and exposure are set per finish.** M2 and M3 keep the rig and the −1.0 view exposure they were calibrated with. M1 read mid-grey for two reasons: the default Principled specular (F0 = 0.04 at roughness 0.68) reflects about five times more light than the 0.9 % base colour, and AgX shows a 0.9 % reflectance at sRGB ≈ 60 under a rig exposed for chalk. M1 now has a lower rough specular (0.22), a thin sharp clear coat for the rim highlights, a lower key and fill, a stronger rim, and its view exposure is trimmed after each render until the lit-torso median reaches sRGB 36 ± 3. The measured medians and the trim history are in `s05_previews*.json → renders.<pose>_<finish>`.
8. **The head has no folded flaps any more.** Radial projection left the mouth, nostril, eye-socket and ear surfaces folded over the egg; the earlier fix sank the fold-only vertices 1.5 mm, but the triangles at the fold line stayed on the skin, back-facing, and Cycles shades a back-facing triangle with a flipped normal: the marks at the chin and brow. `maximus_core.egg_head.unfold` now embeds the whole full-weight head patch on the egg without overlap (harmonic map in a stereographic chart, then a flip-guarded tangential relaxation). The inward-triangle count is in the head paragraph above.
11. **Preview framing is solved, not fixed.** The camera distance and height are computed per pose from the posed mesh so the crown has 7 % headroom and world height 0.852 m stays on the 82 % fade line (`s05_previews*.json → camera`).
9. **The GLB normal map is nearly flat.** The 0.8 mm seams and sub-millimetre grain fall below the 2K texel size on the body UVs (about 1 mm). The GLB therefore reads smoother than the Cycles renders. A dedicated detail-normal tile (a tiling 512² grain map) would carry the grain to the web viewer; it is not built yet.
10. **The core GLB carries the M2 bake only.** glTF cannot carry procedural nodes. M1 and M3 are delivered as 2K texture sets (`out/textures/M1`, `M3`) for the same UVs, and the 2D renders use the full procedural shaders.

## 5. How to run

```
python -I run_all.py                         # s00..s06 via the corpus flow MX-10, strict corpus lock
python -I run_all.py --from s05              # resume
python -I run_stage.py s03_gates             # one stage
python -I -m unittest discover -s tests      # bpy.ops audit
python -I run_stage.py s06_colab_pack        # verified archive for the 4K batch -> colab_stage/maximus_v2_colab_bundle.zip
```
**4K on Colab:**
1. Upload `colab_stage/maximus_v2_colab_bundle.zip` to `My Drive/COLAB_HEAVY/maximus_v2/`.
2. In a GPU runtime (L4), run the cell in `colab_stage/COLAB_CELL.txt`.
3. Copy `results/previews_4k` back to `out/previews_4k` and run `s06_report`.
