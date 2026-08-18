from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

def src(path):
    """Route .cu -> .hip when building for HIP."""
    if IS_HIP and path.endswith(".cu"):
        return path[:-3] + ".hip"
    return path

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        IS_HIP = True
    else:
        IS_HIP = False
else:
    if BUILD_TARGET == "cuda":
        IS_HIP = False
    elif BUILD_TARGET == "rocm":
        IS_HIP = True

if not IS_HIP:
    cc_flag = ["-allow-unsupported-compiler"]
    hip_cxx = ["-O3", "-std=c++20"]
else:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    _rocm = os.getenv("ROCM_PATH", "")
    cc_flag = [f"--offload-arch={a}" for a in archs] + [
        f"--hip-path={_rocm}", f"--rocm-path={_rocm}",
        f"--rocm-device-lib-path={os.path.join(_rocm,'lib','llvm','amdgcn','bitcode')}",
        "-fms-extensions", "-Wno-c++11-narrowing",
    ]

# specify the version of torch that we are building CuMesh against
import torch

def _rocm_local():
    v = torch.__version__            # e.g. "2.12.0+rocm7.14.0a20260624"
    return "+" + v.split("+", 1)[1] if "+" in v else ""

setup(
    name="cumesh-rocm",
    version="1.0" + _rocm_local(),
    packages=[
        'cumesh',
    ],
    ext_modules=[
        CUDAExtension(
            name="cumesh._C",
            include_dirs=[ROOT],
            sources=[
                # Hashmap functions
                src("src/hash/hash.cu"),

                # CuMesh
                src("src/atlas.cu"),
                src("src/clean_up.cu"),
                src("src/cumesh.cu"),
                src("src/connectivity.cu"),
                src("src/geometry.cu"),
                src("src/io.cu"),
                src("src/simplify.cu"),
                src("src/shared.cu"),

                # Remeshing
                src("src/remesh/simple_dual_contour.cu"),
                src("src/remesh/svox2vert.cu"),

                # main
                "src/ext.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": ["-O3","-std=c++20"] + cc_flag,
            }
        ),
        CUDAExtension(
            name='cumesh._cubvh',
            sources=[
                src('third_party/cubvh/src/bvh.cu'),
                src('third_party/cubvh/src/api_gpu.cu'),
                'third_party/cubvh/src/bindings.cpp',
            ],
            include_dirs=[
                os.path.join(ROOT, "third_party/cubvh/include"),
                ROOT,
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": ["-O3","-std=c++20"] + cc_flag + ([] if IS_HIP else ["--extended-lambda","--expt-relaxed-constexpr"]) + [
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ]
            }
        ),
        CUDAExtension(
            name='cumesh._xatlas',
            sources=[
                'third_party/xatlas/xatlas_mod.cpp',
                'third_party/xatlas/binding.cpp',
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)