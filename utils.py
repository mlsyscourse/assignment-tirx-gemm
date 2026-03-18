import torch
import tvm


def prepare_data(M, N, K, dtype="fp16"):
    """Create random A(M,K), B(N,K), C(M,N) on GPU."""
    torch_dev = torch.device("cuda")
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    A = torch.randn(M, K, dtype=torch_dtype, device=torch_dev)
    B = torch.randn(N, K, dtype=torch_dtype, device=torch_dev)
    C = torch.zeros(M, N, dtype=torch_dtype, device=torch_dev)
    return A, B, C


def compile_and_run(kernel, A, B, C):
    """Compile a TIRX kernel and execute it. Returns the output tensor."""
    C_out = torch.zeros_like(C, device="cuda")
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")
        ex(A, B, C_out)
    return C_out


def verify(C_tir, A, B, rtol=1e-3, atol=1e-2):
    """Verify TIRX result against torch.matmul (cuBLAS)."""
    C_ref = torch.matmul(A, B.T)
    torch.testing.assert_close(C_tir, C_ref, rtol=rtol, atol=atol)


def benchmark(kernel, M, N, K, dtype="fp16", warmup=10, repeat=30):
    """Compile and benchmark a kernel. Returns avg time in ms."""
    A, B, C = prepare_data(M, N, K, dtype)
    C_out = torch.zeros_like(C, device="cuda")
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")
        for _ in range(warmup):
            ex(A, B, C_out)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(repeat):
            ex(A, B, C_out)
        end_event.record()
        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event) / repeat
    return elapsed


def _compute_tflops(M, N, K, time_ms):
    """Compute TFLOP/S from dimensions and time in ms."""
    flops = 2 * M * N * K
    return flops / (time_ms * 1e-3) / 1e12


def benchmark_flops(kernel, M, N, K, dtype="fp16", warmup=10, repeat=30):
    """Benchmark a kernel and report FLOP/S."""
    avg_ms = benchmark(kernel, M, N, K, dtype, warmup, repeat)
    tflops = _compute_tflops(M, N, K, avg_ms)
    print(f"M={M}, N={N}, K={K}: {tflops:.2f} TFLOP/S")
    return avg_ms, tflops


# averaged over multiple runs on modal
REFERENCE_TIMES = {
    (1, 128, 128, 64): 0.105,
    (2, 128, 128, 64): 0.105,
    (2, 128, 128, 512): 0.839,
    (2, 128, 128, 1024): 1.678,
    (2, 128, 128, 4096): 6.711,
    (3, 256, 256, 256): 0.373,
    (3, 512, 512, 512): 0.746,
    (3, 1024, 1024, 1024): 1.502,
    (3, 2048, 2048, 2048): 6.007,
    (4, 256, 256, 256): 0.009,
    (4, 512, 512, 512): 0.011,
    (4, 1024, 1024, 1024): 0.017,
    (4, 2048, 2048, 2048): 0.052,
    (5, 512, 512, 512): 0.008,
    (5, 1024, 1024, 1024): 0.012,
    (5, 2048, 2048, 2048): 0.033,
    (5, 4096, 4096, 4096): 0.272,
    (6, 1024, 1024, 1024): 0.013,
    (6, 2048, 2048, 2048): 0.035,
    (6, 4096, 4096, 4096): 0.239,
    (6, 8192, 8192, 8192): 2.143,
    (7, 1024, 1024, 1024): 0.013,
    (7, 2048, 2048, 2048): 0.031,
    (7, 4096, 4096, 4096): 0.230,
    (7, 8192, 8192, 8192): 2.075,
    (8, 1024, 1024, 1024): 0.010,
    (8, 2048, 2048, 2048): 0.023,
    (8, 4096, 4096, 4096): 0.132,
    (8, 8192, 8192, 8192): 1.109,
    (9, 1024, 1024, 1024): 0.015,
    (9, 2048, 2048, 2048): 0.023,
    (9, 4096, 4096, 4096): 0.116,
    (9, 8192, 8192, 8192): 0.788,
    (10, 1024, 1024, 1024): 0.025,
    (10, 2048, 2048, 2048): 0.035,
    (10, 4096, 4096, 4096): 0.107,
    (10, 8192, 8192, 8192): 0.728,
}

# allow 30% slower than reference
TIMING_TOLERANCE = 1.30


def check_timing(kernel, step, M, N, K, dtype="fp16", warmup=10, repeat=30):
    """Benchmark and assert the kernel is within tolerance of reference time."""
    avg_ms, tflops = benchmark_flops(kernel, M, N, K, dtype, warmup, repeat)
    key = (step, M, N, K)
    ref_ms = REFERENCE_TIMES.get(key)
    if ref_ms is None:
        print(f"Missing reference time for shape {key}")
        return avg_ms, tflops

    ref_tflops = _compute_tflops(M, N, K, ref_ms)
    min_tflops = ref_tflops / TIMING_TOLERANCE
    assert tflops >= min_tflops, (
        f"Submission too slow: {tflops:.2f} TFLOP/S"
        f"(reference {ref_tflops:.2f} TFLOP/S)"
    )
    return avg_ms, tflops
