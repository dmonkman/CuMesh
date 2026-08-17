"""
cumesh GPU correctness test.

Runs from anywhere, including inside the CuMesh source tree, by removing the
script directory and CWD from sys.path so the *installed* cumesh package is
imported rather than the local `cumesh/` source folder (which has no compiled
`_C` extension and would shadow it).

Usage:
    python gpu_correctness_test.py
"""

import os
import sys

# --- Ensure we import the INSTALLED cumesh, not a local source folder --------
# Python adds the script's own directory (and sometimes "") to the front of
# sys.path. If the script lives in / is run from the CuMesh repo, a `cumesh/`
# source folder there shadows the installed package. Remove ONLY the script dir,
# the repo root, and the CWD/empty entries (never real site-packages paths)
# so torch and other installed packages still import normally.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)  # test/ -> repo root
_strip = {os.path.abspath(p) for p in (_script_dir, _repo_root, os.getcwd())}
sys.path = [p for p in sys.path if p not in ("", ".") and os.path.abspath(p) not in _strip]

import torch
import cumesh

# Sanity: confirm we got the installed build with the compiled extension.
_src = os.path.dirname(cumesh.__file__)
if not any(fn.startswith("_C") for fn in os.listdir(_src)):
    print(f"WARNING: imported cumesh from {_src} -- no compiled _C found here.")
    print("You are importing the source folder. Install the package first:")
    print("    pip install . --no-build-isolation")
    sys.exit(1)

print(f"cumesh:  {_src}")
print(f"device:  {torch.cuda.get_device_name(0)}")
print(f"has remesh_narrow_band_dc_quad: "
      f"{hasattr(cumesh.remeshing, 'remesh_narrow_band_dc_quad')}")

# --- Correctness: unit cube, nearest-surface distance from a known point -----
# Cube corners at +/-0.5. Query point (1.5, 0, 0) is 1.0 outside the +x face.
v = torch.tensor(
    [[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
     [-.5, -.5,  .5], [.5, -.5,  .5], [.5, .5,  .5], [-.5, .5,  .5]],
    dtype=torch.float32, device="cuda")
f = torch.tensor(
    [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
     [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]],
    dtype=torch.int32, device="cuda")

bvh = cumesh.cuBVH(v, f)
q = torch.tensor([[1.5, 0.0, 0.0]], dtype=torch.float32, device="cuda")
dist, face_idx, _ = bvh.unsigned_distance(q)
torch.cuda.synchronize()

d = dist.item()
print(f"unsigned_distance((1.5,0,0)) = {d:.6f}  (expected 1.0)")

ok = abs(d - 1.0) < 1e-4
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)