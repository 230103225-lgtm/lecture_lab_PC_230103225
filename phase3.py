
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import time
import statistics

from harness import run_benchmark, emit_verify_tag


STUDENT_ID = 230103225

N = 384
BUFFER_SIZE = 96 * 1_000_000

TILE_SIZES = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]

# 80 million FLOPs per worker:
# 16 variables * 2 FLOPs * 2,500,000 iterations
COMPUTE_ITERATIONS = 2_500_000


# ============================================================
# DRAM BANDWIDTH
# ============================================================

def copy_worker(args):
    shm_a_name, shm_b_name, start, end = args

    shm_a = shared_memory.SharedMemory(name=shm_a_name)
    shm_b = shared_memory.SharedMemory(name=shm_b_name)

    mv_a = shm_a.buf
    mv_b = shm_b.buf

    mv_a[start:end] = mv_b[start:end]

    del mv_a
    del mv_b

    shm_a.close()
    shm_b.close()


def run_bandwidth_test(p):

    shm_a = shared_memory.SharedMemory(
        create=True,
        size=BUFFER_SIZE
    )

    shm_b = shared_memory.SharedMemory(
        create=True,
        size=BUFFER_SIZE
    )

    try:
        shm_b.buf[:] = b"\x01" * len(shm_b.buf)

        processes = []

        for worker_id in range(p):

            start = worker_id * BUFFER_SIZE // p
            end = (worker_id + 1) * BUFFER_SIZE // p

            process = mp.Process(
                target=copy_worker,
                args=(
                    shm_a.name,
                    shm_b.name,
                    start,
                    end
                )
            )

            processes.append(process)

        start_time = time.perf_counter_ns()

        for process in processes:
            process.start()

        for process in processes:
            process.join()

            if process.exitcode != 0:
                raise RuntimeError("Bandwidth worker failed")

        end_time = time.perf_counter_ns()

        return (end_time - start_time) / 1e9

    finally:
        shm_a.close()
        shm_a.unlink()

        shm_b.close()
        shm_b.unlink()


# ============================================================
# COMPUTE CEILING
# ============================================================

def compute_worker(iterations):

    a = 1.0000001
    b = 0.0000001

    x0 = x1 = x2 = x3 = 1.0
    x4 = x5 = x6 = x7 = 1.0
    x8 = x9 = x10 = x11 = 1.0
    x12 = x13 = x14 = x15 = 1.0

    for _ in range(iterations):

        x0 = x0 * a + b
        x1 = x1 * a + b
        x2 = x2 * a + b
        x3 = x3 * a + b

        x4 = x4 * a + b
        x5 = x5 * a + b
        x6 = x6 * a + b
        x7 = x7 * a + b

        x8 = x8 * a + b
        x9 = x9 * a + b
        x10 = x10 * a + b
        x11 = x11 * a + b

        x12 = x12 * a + b
        x13 = x13 * a + b
        x14 = x14 * a + b
        x15 = x15 * a + b

    return (
        x0 + x1 + x2 + x3
        + x4 + x5 + x6 + x7
        + x8 + x9 + x10 + x11
        + x12 + x13 + x14 + x15
    )


def run_compute_test(p):

    with mp.Pool(processes=p) as pool:

        start = time.perf_counter_ns()

        pool.map(
            compute_worker,
            [COMPUTE_ITERATIONS] * p
        )

        end = time.perf_counter_ns()

    return (end - start) / 1e9


# ============================================================
# MATRIX MULTIPLICATION (GEMM)
# ============================================================

def gemm_worker(args):

    (
        a_name,
        b_name,
        c_name,
        variant,
        row_start,
        row_end,
        bs
    ) = args

    shm_a = shared_memory.SharedMemory(name=a_name)
    shm_b = shared_memory.SharedMemory(name=b_name)
    shm_c = shared_memory.SharedMemory(name=c_name)

    A = (ctypes.c_double * (N * N)).from_buffer(shm_a.buf)
    B = (ctypes.c_double * (N * N)).from_buffer(shm_b.buf)
    C = (ctypes.c_double * (N * N)).from_buffer(shm_c.buf)

    # Variant 1: Naive i-j-k
    if variant == 1:

        for i in range(row_start, row_end):

            for j in range(N):

                total = 0.0

                for k in range(N):

                    total += (
                        A[i * N + k]
                        * B[k * N + j]
                    )

                C[i * N + j] = total

    # Variant 2: Reordered i-k-j
    elif variant == 2:

        for i in range(row_start, row_end):

            for k in range(N):

                a_ik = A[i * N + k]

                for j in range(N):

                    C[i * N + j] += (
                        a_ik * B[k * N + j]
                    )

    # Variant 3: Blocked i-k-j
    elif variant == 3:

        for i0 in range(row_start, row_end, bs):

            i_max = min(i0 + bs, row_end)

            for k0 in range(0, N, bs):

                k_max = min(k0 + bs, N)

                for j0 in range(0, N, bs):

                    j_max = min(j0 + bs, N)

                    for i in range(i0, i_max):

                        for k in range(k0, k_max):

                            a_ik = A[i * N + k]

                            for j in range(j0, j_max):

                                C[i * N + j] += (
                                    a_ik * B[k * N + j]
                                )

    del A
    del B
    del C

    shm_a.close()
    shm_b.close()
    shm_c.close()


def run_gemm(variant, p, bs=32):

    shm_a = shared_memory.SharedMemory(
        create=True,
        size=N * N * 8
    )

    shm_b = shared_memory.SharedMemory(
        create=True,
        size=N * N * 8
    )

    shm_c = shared_memory.SharedMemory(
        create=True,
        size=N * N * 8
    )

    try:

        A = (ctypes.c_double * (N * N)).from_buffer(shm_a.buf)
        B = (ctypes.c_double * (N * N)).from_buffer(shm_b.buf)
        C = (ctypes.c_double * (N * N)).from_buffer(shm_c.buf)

        for i in range(N * N):

            A[i] = (i % 100) * 0.001
            B[i] = (i % 50) * 0.002
            C[i] = 0.0

        del A
        del B
        del C

        processes = []

        for worker_id in range(p):

            row_start = worker_id * N // p
            row_end = (worker_id + 1) * N // p

            process = mp.Process(
                target=gemm_worker,
                args=(
                    shm_a.name,
                    shm_b.name,
                    shm_c.name,
                    variant,
                    row_start,
                    row_end,
                    bs
                )
            )

            processes.append(process)

        start = time.perf_counter_ns()

        for process in processes:
            process.start()

        for process in processes:
            process.join()

            if process.exitcode != 0:
                raise RuntimeError("GEMM worker failed")

        end = time.perf_counter_ns()

        C = (
            ctypes.c_double * (N * N)
        ).from_buffer(shm_c.buf)

        result = list(C)

        del C

        elapsed = (end - start) / 1e6

        return elapsed, result

    finally:

        shm_a.close()
        shm_a.unlink()

        shm_b.close()
        shm_b.unlink()

        shm_c.close()
        shm_c.unlink()


# ============================================================
# MAIN
# ============================================================

def main():

    print("===== PHASE 3: ROOFLINE AND GEMM =====")

    p = 4

    # --------------------------------------------------------
    # DRAM BANDWIDTH
    # --------------------------------------------------------

    print("\n===== DRAM BANDWIDTH =====")

    bandwidth_times = []

    for _ in range(3):

        bandwidth_times.append(
            run_bandwidth_test(p)
        )

    median_bandwidth_time = statistics.median(
        bandwidth_times
    )

    bandwidth = (
        2 * BUFFER_SIZE
        / median_bandwidth_time
        / 1e9
    )

    print(
        "Median time:",
        median_bandwidth_time,
        "seconds"
    )

    print(
        "Sustained bandwidth:",
        bandwidth,
        "GB/s"
    )

    # --------------------------------------------------------
    # COMPUTE CEILING
    # --------------------------------------------------------

    print("\n===== COMPUTE CEILING =====")

    compute_times = []

    for _ in range(3):

        compute_times.append(
            run_compute_test(p)
        )

    median_compute_time = statistics.median(
        compute_times
    )

    total_flops = (
        16 * 2 * COMPUTE_ITERATIONS * p
    )

    compute_gflops = (
        total_flops
        / median_compute_time
        / 1e9
    )

    print(
        "Median time:",
        median_compute_time,
        "seconds"
    )

    print(
        "Compute ceiling:",
        compute_gflops,
        "GFLOP/s"
    )

    ridge_point = compute_gflops / bandwidth

    print(
        "Ridge point:",
        ridge_point,
        "FLOPs/Byte"
    )

    # --------------------------------------------------------
    # GEMM VARIANTS
    # --------------------------------------------------------

    print("\n===== GEMM VARIANTS =====")

    gemm_results = {}

    for variant in [1, 2]:

        elapsed, result = run_gemm(variant, p)

        gflops = (
            2 * N ** 3
            / (elapsed * 1e6)
        )

        gemm_results[variant] = (
            elapsed,
            result,
            gflops
        )

        print(
            f"Variant {variant} | "
            f"Time={elapsed:.3f} ms | "
            f"GFLOP/s={gflops:.5f}"
        )

    # --------------------------------------------------------
    # TILE SIZE SWEEP
    # --------------------------------------------------------

    print("\n===== TILE SIZE SWEEP =====")

    tile_results = {}

    for bs in TILE_SIZES:

        elapsed, result = run_gemm(3, p, bs)

        gflops = (
            2 * N ** 3
            / (elapsed * 1e6)
        )

        tile_results[bs] = (
            elapsed,
            result,
            gflops
        )

        print(
            f"BS={bs:3d} | "
            f"Time={elapsed:.3f} ms | "
            f"GFLOP/s={gflops:.5f}"
        )

    # --------------------------------------------------------
    # NUMERICAL VALIDATION
    # --------------------------------------------------------

    print("\n===== NUMERICAL VALIDATION =====")

    reference = gemm_results[1][1]

    # Validate Variant 2
    result = gemm_results[2][1]

    max_error = max(
        abs(reference[i] - result[i])
        for i in range(N * N)
    )

    print(
        f"Variant 2 | "
        f"Max error={max_error:.12f}"
    )

    # Validate all tiled variants
    for bs, (_, result, _) in tile_results.items():

        max_error = max(
            abs(reference[i] - result[i])
            for i in range(N * N)
        )

        print(
            f"BS={bs:3d} | "
            f"Max error={max_error:.12f}"
        )

    # --------------------------------------------------------
    # BEST TILE
    # --------------------------------------------------------

    best_bs = max(
        tile_results,
        key=lambda bs: tile_results[bs][2]
    )

    variant1_gflops = gemm_results[1][2]

    best_gflops = tile_results[best_bs][2]

    print("\nBest tile size:", best_bs)

    print("Best GFLOP/s:", best_gflops)

    # --------------------------------------------------------
    # TILE WORKING SET
    # --------------------------------------------------------

    print("\n===== TILE WORKING SET =====")

    for bs in TILE_SIZES:

        footprint = 3 * bs * bs * 8

        print(
            f"BS={bs:3d} | "
            f"Working set={footprint:,} bytes"
        )

    # --------------------------------------------------------
    # ROOFLINE ANALYSIS
    # --------------------------------------------------------

    print("\n===== ROOFLINE ANALYSIS =====")

    # Naive operational intensity
    naive_oi = 0.125

    naive_roofline = min(
        compute_gflops,
        bandwidth * naive_oi
    )

    naive_efficiency = (
        variant1_gflops
        / naive_roofline
        * 100
    )

    print(
        "Naive Operational Intensity:",
        naive_oi
    )

    print(
        "Naive Roofline Bound:",
        naive_roofline,
        "GFLOP/s"
    )

    print(
        "Naive Roofline Efficiency:",
        naive_efficiency,
        "%"
    )

    # Best tiled operational intensity
    best_oi = best_bs / 12

    best_roofline = min(
        compute_gflops,
        bandwidth * best_oi
    )

    best_efficiency = (
        best_gflops
        / best_roofline
        * 100
    )

    print(
        "Best Tile Operational Intensity:",
        best_oi
    )

    print(
        "Best Tile Roofline Bound:",
        best_roofline,
        "GFLOP/s"
    )

    print(
        "Best Tile Roofline Efficiency:",
        best_efficiency,
        "%"
    )


    print("\n===== VERIFICATION =====")

    print(
        emit_verify_tag(
            "PHASE3",
            STUDENT_ID,
            variant1_gflops,
            best_gflops
        )
    )


if __name__ == "__main__":

    mp.freeze_support()

    main()
