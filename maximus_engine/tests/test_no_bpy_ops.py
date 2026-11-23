"""Owner rule: bpy.ops may appear only in pipeline/io_ops_gateway.py (render, bake, glTF export, save).
python -I -m unittest discover -s tests"""
import re, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"render.render", "object.bake", "export_scene.gltf", "wm.save_as_mainfile"}


class NoBpyOps(unittest.TestCase):
    def test_only_gateway_calls_operators(self):
        bad = []
        for p in list((ROOT / "pipeline").glob("*.py")) + list((ROOT / "colab_stage").glob("*.py")) + list((ROOT / "runtime_v3").glob("*.py")):
            code = [l for l in p.read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("#")]
            hits = re.findall(r"bpy\.ops\.([a-z_]+\.[a-z_]+)\(", "\n".join(code))
            if p.name == "io_ops_gateway.py":
                self.assertTrue(set(hits) <= ALLOWED, f"gateway uses non-whitelisted ops: {set(hits) - ALLOWED}")
                self.assertTrue(hits, "gateway should contain the whitelisted calls")
            elif hits:
                bad.append((p.name, hits))
        self.assertEqual(bad, [], f"bpy.ops outside the gateway: {bad}")


if __name__ == "__main__":
    unittest.main()
