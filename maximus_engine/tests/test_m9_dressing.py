"""Unit & truth-gate tests for M9 Universal Dressing, Body Measurement & Coupled Refit.

Run: python -I -m unittest tests/test_m9_dressing.py
"""
import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "colab_stage", ROOT / "pipeline"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import corpus_link  # noqa: E402
import body_measure_engine as BME  # noqa: E402
import garment_transport as GT  # noqa: E402
import coupled_shirt_fitter as CSF  # noqa: E402
import m6_dual_lod_game_export as M6  # noqa: E402


class TestM9Dressing(unittest.TestCase):
    def test_corpus_link_and_zero_bpy_ops(self):
        cpath = corpus_link.link(strict=True)
        self.assertTrue(Path(cpath).is_dir())
        for pyf in (
            ROOT / "garment_transport.py",
            ROOT / "body_measure_engine.py",
            ROOT / "coupled_shirt_fitter.py",
            ROOT / "colab_stage" / "m9_dressed_reels.py",
        ):
            code = [l for l in pyf.read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("#")]
            hits = re.findall(r"bpy\.ops\.([a-z_]+\.[a-z_]+)\(", "\n".join(code))
            self.assertEqual(hits, [], f"bpy.ops found in {pyf.name}: {hits}")

    def test_body_measure_and_ridge_solver(self):
        bme = BME.BodyMeasureEngine()
        for build, h, w in (("lean", 173.0, 62.0), ("average", 173.1, 72.0), ("athletic", 178.0, 82.0), ("stocky", 175.0, 95.0)):
            res = bme.measure_from_profile(h, w, build, pose="rest")
            self.assertGreater(res["stature_cm"], 165.0)
            self.assertGreater(res["chest_girth_cm"], 75.0)
            self.assertGreater(res["waist_girth_cm"], 50.0)
            self.assertGreater(res["hip_girth_cm"], 70.0)
            self.assertGreater(res["neck_girth_cm"], 29.0)
            for k, val in res["beta"].items():
                self.assertTrue(0.0 <= val <= 1.0, (k, val))
        # Monotonicity: stocky chest > lean chest
        lean_c = bme.measure_from_profile(173.0, 62.0, "lean")["chest_girth_cm"]
        stk_c = bme.measure_from_profile(175.0, 95.0, "stocky")["chest_girth_cm"]
        self.assertGreater(stk_c, lean_c + 15.0)

    def test_shell_cache_and_o7_collar_truth(self):
        shell_npz = ROOT / "out" / "m9" / "hero_ringer_shell.npz"
        self.assertTrue(shell_npz.exists(), "Run garment_transport.py first to build hero_ringer_shell.npz")
        shell, gbind = GT.load_shell_cache(shell_npz)
        self.assertTrue(shell["collar_mask"].any())
        self.assertIn("UVMap_Art_Front", shell["uv_names"])
        o7 = GT.check_o7_collar_seating(gbind["V_fit"], gbind["F"], gbind["collar_mask"], gbind["body_ref_V"], gbind["body_F"])
        self.assertEqual(o7["inside_count"], 0)
        self.assertTrue(o7["o7_seated_le_1mm"])

        # Check collar weight isolation strictly to neck01 / spine03
        keep_24 = list(M6.SPEC_TO_RIG.values())
        W24_body = np.zeros((len(gbind["body_ref_V"]), 24), dtype=np.float32)
        W24_body[:, keep_24.index("spine03")] = 0.7
        W24_body[:, keep_24.index("clavicle.L")] = 0.3
        W_shirt = GT.transfer_body_weights_to_garment(gbind["V_fit"], gbind["body_ref_V"], W24_body, keep_24, shell["collar_mask"])
        clav_i = keep_24.index("clavicle.L")
        self.assertEqual(float(W_shirt[shell["collar_mask"], clav_i].max()), 0.0)

    def test_dressed_manifest_and_glb_budget(self):
        m_path = ROOT / "out" / "m9" / "m9_dressed_manifest.json"
        glb_path = ROOT / "out" / "m9" / "maximus_v2_dressed_ringer_lod1.glb"
        self.assertTrue(m_path.exists() and glb_path.exists(), "Run coupled_shirt_fitter.py first")
        man = json.loads(m_path.read_text(encoding="utf-8"))
        self.assertLess(man["glb_bytes"], 2 * 1024 * 1024)
        for bname, sizes in man["verification_matrix"].items():
            for sz, rep in sizes.items():
                self.assertEqual(rep["inside_gwn_gt_05"], 0, f"{bname}/{sz} had body penetrations")
                self.assertGreaterEqual(rep["min_signed_mm"], 2.9, f"{bname}/{sz} < 3mm barrier")
                CSF.FitReport(**rep)  # Validate Pydantic schema

        summary = M6.glb_summary(glb_path)
        self.assertEqual(summary["joints"], 24)
        self.assertEqual(summary["morph_targets"], 7)
        self.assertEqual(summary["morph_target_names"], ["gt_lean", "gt_stocky_nat", "gt_stocky_ext", "leg_len_m5cm", "leg_len_p5cm", "size_XL", "size_XXL"])


if __name__ == "__main__":
    unittest.main()
