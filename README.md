# CuMesh: High-Performance Geometry Processing for PyTorch

**CuMesh** is a GPU-accelerated library designed for high-performance 3D geometry processing directly within the PyTorch ecosystem. It provides efficient primitives for mesh cleaning, decimation, remeshing, and UV unwrapping.

Key features include:
*   **CUDA/HIP Accelerated Mesh Operations**: Fast topology queries, simplification, hole filling, and cleaning.
*   **Remeshing**: Remesh arbitrary meshes using narrow-band UDF and Dual Contouring.
*   **UV Unwrapping**: Efficient UV parameterization with `xatlas` enhenced by a fast mesh clustering on the GPU.

---

## 🔴 NEW (August 2026): AMD ROCm / HIP Support on Windows (including RDNA2)

**This fork adds a working build for AMD GPUs (including RX 6000 series) on Windows**.
The upstream sources are CUDA-only; here they are hipified to
`.hip` files (kept alongside the original `.cu`), and `setup.py` routes to the
correct variant per platform. NVIDIA users can use the upstream repo unchanged.

**Validated on:** AMD Radeon RX 6800 XT (`gfx1030`, RDNA2), Windows 11,
ROCm 7.14, PyTorch 2.13, Python 3.12. It should work on other RDNA2/RDNA3 cards
whose `gfx` target is supported by your ROCm SDK (the build script will auto-detect the
architecture), but only `gfx1030` is tested.

### Prerequisites (AMD)

*   An **AMD Radeon GPU (RDNA2 tested; RDNA1/3/4 should work — see table below)**
*   **Python 3.12** with a recent **pip** (`python -m pip install --upgrade pip`)
*   **Visual Studio 2022** with the C++ workload (the build needs `link.exe`).
*   A **ROCm 7.14 SDK + ROCm PyTorch** for your `gfx` target, installed into your
    pip environment. ROCm 7.14 is required specifically because `torch_hip.dll` ships
    the HIP `MasqueradingAsCUDA` symbols that earlier Windows builds lack.

The required wheels can be found at `https://rocm.nightlies.amd.com/v2-staging/`

| Card series | Arch | Channel |
|---|---|---|
| RX 5000 (RDNA1) | gfx1010–1012 | [`gfx101X-dgpu`](https://rocm.nightlies.amd.com/v2-staging/gfx101X-dgpu/) |
| RX 6000 (RDNA2) | gfx1030–1036 | [`gfx103X-all`](https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/) |
| RX 7000 (RDNA3) | gfx1100–1103 | [`gfx110X-all`](https://rocm.nightlies.amd.com/v2-staging/gfx110X-all/) |
| Ryzen AI / Strix Halo | gfx1150–1153 | [`gfx1151`](https://rocm.nightlies.amd.com/v2-staging/gfx1151/) (etc.) |
| RX 9000 (RDNA4) | gfx1200/1201 | [`gfx120X-all`](https://rocm.nightlies.amd.com/v2-staging/gfx120X-all/) |


**Verify that you downloaded the correct wheels:**
- All filenames should contain `win_amd64` (Windows x64 builds).
- torch, torchaudio, and torchvision should contain `cp312` (Python 3.12 builds).

**Example**: The wheel set I used (gfx103x / RDNA2 / RX 6800XT, Windows 11, Python 3.12), from
`https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/`:

```
rocm-7.14.0a20260612.tar.gz
rocm_sdk_core-7.14.0a20260612-py3-none-win_amd64.whl
rocm_sdk_devel-7.14.0a20260612-py3-none-win_amd64.whl
rocm_sdk_libraries_gfx103x_all-7.14.0a20260612-py3-none-win_amd64.whl
torch-2.13.0a0+rocm7.14.0a20260612-cp312-cp312-win_amd64.whl
torchaudio-2.11.0a0+rocm7.14.0a20260612-cp312-cp312-win_amd64.whl
torchvision-0.28.0a0+rocm7.14.0a20260612-cp312-cp312-win_amd64.whl
```

### Build (AMD / Windows)

From any PowerShell:

**Step 1:** Clone Repo
```powershell
git clone https://github.com/dmonkman/CuMesh.git
cd CuMesh
```

**Step 2:** Activate the python env where you want to install CuMesh (ensuring that the pre-requisite wheels are installed)
```powershell
# Replace the path with your own venv. This is required as the build reads
# torch and the ROCm SDK from this environment.
& "C:\path\to\your\venv\Scripts\Activate.ps1"
```

**Step 3:** Run setup script to prepare the build environment (determine hardware, find Visual Studio compiler, etc.)
```powershell
# sets ROCm paths, clang-cl, and auto-detects GPU_ARCHS from your GPU
.\setup_build_env.ps1
```
`setup_build_env.ps1` auto-detects your GPU architecture via PyTorch and
validates it against the installed ROCm SDK targets. If needed, override with
`$env:GPU_ARCHS = "gfxXXXX"` before running.

**Step 4:** Use pip to install the package. This compiles the extension from source for your GPU. If it fails, create a git issue.
```powershell
pip install . --no-build-isolation --no-cache-dir
```


### Validate

Run this command from **outside** the source directory so the installed package is imported
(not the local `cumesh/` source folder):

Run the GPU correctness check (unit cube; nearest-surface distance from a known point,
expected `1.0`):

```powershell
python test/gpu_correctness_test.py
```
Excpected output (device will depend on your system):
```
cumesh: ...\site-packages\cumesh
device: AMD Radeon RX 6800 XT
has remesh_narrow_band_dc_quad: True
unsigned_distance((1.5,0,0)) = 1.000000 (expected 1.0)
PASS
```

### What was changed for HIP

Mostly mechanical (`hipify-perl` over `src/**` and `third_party/cubvh/**`),
plus manual cleanup that hipify does not cover:

*   **Source router** in `setup.py` — compiles `.hip` on HIP, `.cu` on CUDA.
*   **`::cuda::std::` → `::std::`** (libcu++ constructs are not remapped).
*   **rocPRIM radix-sort decomposer** returns `::rocprim::tuple<...>` (not
    `std::tuple`) for `hipcub::DeviceRadixSort::SortPairs`.
*   **`thrust::cuda` → `thrust::hip`** (thrust execution-policy namespace).
*   Reverted hipify's over-rename of the project's own `cubvh` namespace.
*   **Eigen include path** points at the repo root (Eigen is vendored there).

---

## Installation (NVIDIA / CUDA)

### Prerequisites
*   Python >= 3.8
*   PyTorch >= 2.4 (with CUDA support)
*   CUDA Toolkit >= 12.4 (matching your PyTorch version)

### Build from Source

```bash
git clone https://github.com/JeffreyXiang/CuMesh.git --recursive
pip install CuMesh --no-build-isolation
```


## Quick Start & Modules

See the [examples](examples) directory for more detailed usage.


## API Reference

### `cumesh.CuMesh`

*   **`init(vertices, faces)`**: Initialize mesh with `[V,3]` and `[F,3]` CUDA tensors.
*   **`read()`**: Return current `(vertices, faces)` tensors.
*   **`simplify(target_num_faces, verbose=False, options={})`**: Fast GPU-accelerated mesh decimation.
*   **`uv_unwrap(verbose=False, ...)`**: Generate UVs using accelerated clustering and Xatlas.
*   **`fill_holes(max_hole_perimeter)`**: Triangulate and close boundary loops.
*   **`repair_non_manifold_edges()`**: Split edges to resolve non-manifold geometry.
*   **`remove_degenerate_faces()`**: Remove zero-area faces or those with NaN normals.
*   **`remove_duplicate_faces()`**: Remove faces with identical vertex indices.
*   **`remove_small_connected_components(min_area)`**: Delete isolated components below area threshold.
*   **`unify_face_orientations()`**: Reorient faces to have consistent winding order.
*   **`compute_face_normals()`**, **`compute_vertex_normals()`**: Trigger normal calculation (access via `read_*_normals`).
*   **`get_connected_components()`**: Compute connectivity (access via `read_connected_components`).
*   **`get_boundary_loops()`**: Detect boundaries (access via `read_boundary_loops`).
*   **Properties**: `num_vertices`, `num_faces`, `num_edges`, `num_boundaries`.

### `cumesh.remeshing`
*   `remesh_narrow_band_dc(...)`: Performs Dual Contouring reconstruction based on the UDF of the input mesh.

### `cumesh.cuBVH`

*NOTE: This is a wrapper around the [`cubvh`](https://github.com/ashawkey/cubvh) library.*

### `cumesh.Atlas`

*NOTE: This is a wrapper around the [`xatlas`](https://github.com/jpcy/xatlas) library.*

*   **`add_mesh(vertices, faces, normals=None, uvs=None)`**: Register mesh geometry (Must be **CPU** tensors).
*   **`compute_charts(max_chart_area, ...)`**: Segment mesh into UV charts (parameterization).
*   **`pack_charts(resolution, padding, ...)`**: Pack generated charts into a texture atlas.
*   **`get_mesh(index)`**: Retrieve processed data as `(vertex_map, faces, uvs)`.
    *   `vertex_map`: Maps new vertex indices to original input indices.


## Acknowledgements

This package builds upon and integrates code from several excellent open-source libraries. We would like to express our gratitude to the authors of:

*   **[cubvh](https://github.com/ashawkey/cubvh)**: For the high-performance CUDA BVH acceleration toolkit.
*   **[xatlas](https://github.com/jpcy/xatlas)**: For the robust UV parameterization and atlas packing library.
*   **[Eigen](https://eigen.tuxfamily.org/)**: For the C++ template library for linear algebra, used by the cubvh backend.
*   **[pamo](https://github.com/SarahWeiii/pamo)**: For the reference implementation of the GPU parallel edge collapse algorithm used in our mesh simplification module.

## License

[MIT License](LICENSE)