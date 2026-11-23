"""Stage 00 (Blender, read-only): verify the canonical v1 rig hash and dump its arrays for the rebuild and the gates.

Input  : 03_full_rigged.blend (sha256 1be2d4ca...), opened read-only from the command line. Nothing is saved.
Output : out/cache/v1_arrays.npz, out/cache/v1_fingerprint.json
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401,E702

import numpy as np

import bl_common as C

from corpus_bridge import contracts


def main():
    src = C.bpy.data.filepath
    got = contracts.sha256_file(src)
    if got != C.V1_SHA256:
        raise RuntimeError(f"v1 hash mismatch: {got} != {C.V1_SHA256}")
    body, rig = C.human_and_rig()
    M = np.array(body.matrix_world, float)
    if not np.allclose(M, np.eye(4), atol=1e-9) or not np.allclose(np.array(rig.matrix_world), np.eye(4), atol=1e-9):
        raise RuntimeError("expected identity world matrices on Human and Human.rig")
    a = C.mesh_arrays(body.data); W, names = C.weights_matrix(body); keys = C.shape_key_arrays(body.data)
    bones = C.bone_table(rig)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(C.CACHE / "v1_arrays.npz", V=a["V"], T=a["T"], TP=a["TP"], uv=a["uv"], loop_vert=a["loop_vert"],
                        W=W, group_names=np.array(names), key_names=np.array(list(keys)),
                        keys=np.stack(list(keys.values())))
    fp = C.fingerprint(body)
    contracts.atomic_write_json(C.CACHE / "v1_fingerprint.json", {"source": src, "sha256": got, "fingerprint": fp, "bones": bones})
    return ({"npz": str(C.CACHE / "v1_arrays.npz"), "fingerprint": str(C.CACHE / "v1_fingerprint.json")},
            {"n_verts": fp["n_verts"], "n_tris": int(len(a["T"])), "n_bones": fp["n_bones"], "shape_keys": fp["shape_keys"],
             "min_z": float(a["V"][:, 2].min())})


C.run("s00_extract_v1", main)
