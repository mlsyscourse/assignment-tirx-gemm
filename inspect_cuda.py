"""Inspect generated CUDA/PTX source for any hgemm kernel step.

Compiles a TIRX kernel and prints the generated CUDA source code.
Useful for debugging barrier logic, memory layout, and PTX instructions.

Usage:
    python inspect_cuda.py <step> [size]

Arguments:
    step    Kernel version (1-10)
    size    Matrix dimension M=N=K (default: 1024)

Examples:
    python inspect_cuda.py 7           # v7, 1024x1024
    python inspect_cuda.py 8 2048      # v8, 2048x2048
    python inspect_cuda.py 10 4096     # v10, 4096x4096

    # Save to file for easier reading:
    python inspect_cuda.py 9 1024 > v9_cuda.cu

    # Search for specific instructions:
    python inspect_cuda.py 7 | grep tcgen05
    python inspect_cuda.py 9 | grep mbarrier
"""
import tvm
import sys

sys.path.insert(0, ".")
from gemm_kernels import *

step = int(sys.argv[1]) if len(sys.argv) > 1 else 7
size = int(sys.argv[2]) if len(sys.argv) > 2 else 1024

kernels = {
    1: hgemm_v1, 2: hgemm_v2, 3: hgemm_v3, 4: hgemm_v4,
    5: hgemm_v5, 6: hgemm_v6, 7: hgemm_v7, 8: hgemm_v8,
    9: hgemm_v9, 10: hgemm_v10,
}

if step not in kernels:
    print(f"Error: step must be 1-10, got {step}")
    sys.exit(1)

print(f"// Compiling hgemm_v{step}(M={size}, N={size}, K={size})...", file=sys.stderr)

kernel = kernels[step](size, size, size)
target = tvm.target.Target("cuda")
with target:
    mod = tvm.IRModule({"main": kernel})
    lib = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = lib.mod.imports[0].inspect_source()
    print(src)
