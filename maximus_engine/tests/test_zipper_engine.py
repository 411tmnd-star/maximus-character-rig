"""Unit tests for colab_stage/zipper_engine.py (pure NumPy + GG-30 binding, no Blender required)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "colab_stage"))
import canon_geometry_kernels as K   # noqa: E402
import zipper_engine as ZE           # noqa: E402


def _straight_spine(length_m=0.35, n=20):
    z = np.linspace(0.0, length_m, n)
    return np.column_stack([np.zeros(n), np.zeros(n), z])


def _curved_jacket_seam(length_m=0.45, n=32):
    t = np.linspace(0.0, 1.0, n)
    x = 0.015 * np.sin(np.pi * t)
    y = -0.11 - 0.025 * np.sin(np.pi * t)
    z = 0.90 + length_m * t
    return np.column_stack([x, y, z])


class TestZipperEngine(unittest.TestCase):
    def test_config_validation_all_8_params(self):
        cfg = ZE.ZipperConfig(
            size_multiplier=1.35,
            tooth_index=1,
            head_index=2,
            open_factor=0.35,
            flip=True,
            tooth_material="polished_nickel",
            head_material="antique_brass",
            textile_uv_attribute="textile UV",
        )
        d = cfg.to_dict()
        self.assertEqual(set(d.keys()), {
            "size_multiplier", "tooth_index", "head_index", "open_factor",
            "flip", "tooth_material", "head_material", "textile_uv_attribute",
        })
        with self.assertRaises(ValueError):
            ZE.ZipperConfig(open_factor=1.2)
        with self.assertRaises(ValueError):
            ZE.ZipperConfig(tooth_index=5)
        with self.assertRaises(ValueError):
            ZE.ZipperConfig(head_index=4)
        with self.assertRaises(ValueError):
            ZE.ZipperConfig(size_multiplier=0.0)

    def test_all_5_tooth_types_and_4_head_types_manifold(self):
        curve = _curved_jacket_seam()
        for ti in range(5):
            res = ZE.build_zipper_numpy(curve, ZE.ZipperConfig(tooth_index=ti, open_factor=0.25))
            self.assertEqual(res["stats"]["degenerate_triangles"], 0, f"tooth_index={ti}")
            self.assertTrue(res["stats"]["all_parts_positive_volume"], f"tooth_index={ti}: {res['stats']}")
            self.assertGreater(res["slider_state"]["teeth_count_left"], 50)
        for hi in range(4):
            res = ZE.build_zipper_numpy(curve, ZE.ZipperConfig(head_index=hi, open_factor=0.40))
            self.assertEqual(res["stats"]["degenerate_triangles"], 0, f"head_index={hi}")
            self.assertTrue(res["stats"]["all_parts_positive_volume"], f"head_index={hi}")
            if hi == 3:
                self.assertIsNotNone(res["slider_state"]["s_lower_slider_m"])
            else:
                self.assertIsNone(res["slider_state"]["s_lower_slider_m"])

    def test_open_factor_single_source_of_truth(self):
        curve = _straight_spine(0.40)
        r0 = ZE.build_zipper_numpy(curve, ZE.ZipperConfig(open_factor=0.0))
        r5 = ZE.build_zipper_numpy(curve, ZE.ZipperConfig(open_factor=0.5))
        r1 = ZE.build_zipper_numpy(curve, ZE.ZipperConfig(open_factor=1.0))
        # Slider moves monotonically downward as open_factor goes 0 -> 0.5 -> 1.0
        self.assertGreater(r0["slider_state"]["s_slider_m"], r5["slider_state"]["s_slider_m"])
        self.assertGreater(r5["slider_state"]["s_slider_m"], r1["slider_state"]["s_slider_m"])
        # Top Y-split gap is 0 when closed and widens monotonically as zipper opens
        self.assertAlmostEqual(r0["slider_state"]["top_split_half_gap_mm"], 0.0, places=6)
        self.assertGreater(r5["slider_state"]["top_split_half_gap_mm"], 10.0)
        self.assertGreater(r1["slider_state"]["top_split_half_gap_mm"], r5["slider_state"]["top_split_half_gap_mm"])

    def test_size_multiplier_and_flip_and_textile_uv(self):
        curve = _straight_spine(0.30)
        r5 = ZE.build_zipper_numpy(curve, ZE.ZipperConfig(size_multiplier=1.0))
        r8 = ZE.build_zipper_numpy(curve, ZE.ZipperConfig(size_multiplier=1.35))
        # #8 gauge has larger tooth pitch -> fewer teeth on same length, larger tooth volume
        self.assertLess(r8["slider_state"]["teeth_count_left"], r5["slider_state"]["teeth_count_left"])
        self.assertGreater(r8["stats"]["signed_volume_mm3_by_part"]["teeth"], r5["stats"]["signed_volume_mm3_by_part"]["teeth"])
        # Flip reverses slider position from top toward bottom in world Z
        rf = ZE.build_zipper_numpy(curve, ZE.ZipperConfig(open_factor=0.0, flip=True))
        self.assertLess(rf["slider_state"]["slider_pos_m"][2], 0.05)
        self.assertGreater(r5["slider_state"]["slider_pos_m"][2], 0.25)
        # Continuous textile UV covers [0, 1] across tape width and > 10 repeats along 30 cm length
        uv = r5["parts"]["tape"]["uv"]
        self.assertGreater(uv[:, 0].max() - uv[:, 0].min(), 0.8)
        self.assertGreater(uv[:, 1].max() - uv[:, 1].min(), 20.0)

    def test_gg30_rigid_metal_transport_under_cloth_stretch(self):
        # Host garment strip of 2 triangles
        Vh = np.array([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.0, 0.4], [-0.1, 0.0, 0.4]], dtype=np.float64)
        Fh = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        curve = _straight_spine(0.30)
        z = ZE.build_zipper_numpy(curve, ZE.ZipperConfig(open_factor=0.3))
        b = ZE.bind_zipper_to_mesh(K, z, Vh, Fh)
        # Apply rigid rotation + translation + 15% cloth stretch along X
        R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        t = np.array([0.5, -0.3, 1.0])
        Vh_rigid = Vh @ R.T + t
        out_rigid = ZE.transport_zipper(K, b, Vh_rigid, Fh)
        for p in ZE.PART_NAMES:
            err = np.abs(out_rigid[p] - (z["parts"][p]["V"] @ R.T + t)).max()
            self.assertLess(err, 1e-12, f"rigid invariance failed for {p}: {err}")
        # Under 20% cloth stretch, slider head edge lengths stay strictly unchanged (rigid metal)
        Vh_stretch = Vh.copy(); Vh_stretch[:, 0] *= 1.20
        out_s = ZE.transport_zipper(K, b, Vh_stretch, Fh)
        fh = z["parts"]["head"]["F"][:20]
        e0 = np.linalg.norm(z["parts"]["head"]["V"][fh[:, 1]] - z["parts"]["head"]["V"][fh[:, 0]], axis=1)
        e1 = np.linalg.norm(out_s["head"][fh[:, 1]] - out_s["head"][fh[:, 0]], axis=1)
        self.assertLess(np.abs(e1 - e0).max(), 1e-12)

    def test_rigid_half_pitch_clearance_gate_and_instanced_glb(self):
        import tempfile
        C0, N0 = ZE.default_jacket_centerline(0.420)
        z = ZE.build_zipper_numpy(C0, ZE.ZipperConfig(open_factor=0.18), curve_normals=N0)
        cg = z["clearance_gate"]
        self.assertEqual(cg["pitch_mm"], 3.0)
        self.assertEqual(cg["half_pitch_stagger_mm"], 1.5)
        self.assertGreaterEqual(cg["min_closed_intertooth_clearance_mm"], 0.05)
        self.assertTrue(cg["rigid_se3_only"])
        self.assertTrue(cg["pass"])
        with tempfile.TemporaryDirectory() as d:
            glb_path = Path(d) / "zipper_hero_instanced.glb"
            info = ZE.export_instanced_zipper_glb(glb_path, C0, ZE.ZipperConfig(open_factor=0.18), curve_normals=N0)
            self.assertTrue(glb_path.exists())
            self.assertTrue(info["under_400KB"], info)
            self.assertIn("EXT_mesh_gpu_instancing", info["extensionsUsed"])
            self.assertEqual(info["animation"], "Zip_Scrub_0_to_1")
            self.assertGreater(info["instanced_teeth_count"], 200)


    def test_isometric_y_throat_tape_stretch_and_exact_pitch(self):
        """Milestone 1: isometric layout keeps tape stretch <= 1.006 and tooth pitch exact through the Y-throat."""
        C0, N0 = ZE.default_jacket_centerline(0.420)
        for O in (0.0, 0.3, 0.58, 1.0):
            z = ZE.build_zipper_numpy(C0, ZE.ZipperConfig(open_factor=O), curve_normals=N0, isometric=True)
            g = z["G_ZIP_CLEAR"]
            self.assertLessEqual(g["tape_stretch_max"], 1.006, (O, g))
            self.assertGreaterEqual(g["min_closed_intertooth_clearance_mm"], 0.05)
            self.assertGreater(g["tooth_pitch_ratio_min"], 0.999, (O, g))
            self.assertLess(g["tooth_pitch_ratio_max"], 1.001, (O, g))
            self.assertTrue(g["pass"])
            self.assertEqual(z["stats"]["degenerate_triangles"], 0)
            self.assertTrue(z["stats"]["all_parts_positive_volume"])
            te = z["tape_edges"]
            self.assertEqual(te["side_plus"].shape, te["side_minus"].shape)
            self.assertEqual(len(te["s_material"]), len(te["side_plus"]))
        # opening widens to the ISO half-gap (26 mm) and the legacy layout is unchanged and still reports its stretch
        z_open = ZE.build_zipper_numpy(C0, ZE.ZipperConfig(open_factor=0.58), curve_normals=N0, isometric=True)
        self.assertAlmostEqual(z_open["slider_state"]["top_split_half_gap_mm"], 26.0, places=3)
        z_leg = ZE.build_zipper_numpy(C0, ZE.ZipperConfig(open_factor=0.58), curve_normals=N0)
        self.assertGreater(z_leg["G_ZIP_CLEAR"]["tape_stretch_max"], 1.3)
        self.assertFalse(z_leg["G_ZIP_CLEAR"]["isometric"])


if __name__ == "__main__":
    unittest.main()
