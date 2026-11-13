import multiprocessing as mp
import math
import time
import statistics

from harness import run_benchmark, emit_verify_tag


STUDENT_ID = 230103225

STRONG_N = 1_600_000
WEAK_N0 = 250_000


def serial_stage(n):

    count = max(1, int(n * 0.05))

    start = time.perf_counter_ns()

    result = 0.0

    for i in range(count):
        x = i * 0.00001 + 1.0

        for _ in range(6):
            result += math.sin(x)
            result += math.sqrt(x)

    end = time.perf_counter_ns()

    elapsed_ms = (end - start) / 1e6

    return elapsed_ms, result


def parallel_worker(args):

    start, end = args

    result = 0.0

    for i in range(start, end):
        x = i * 0.000001 + 1.0

        for _ in range(35):
            x = x * 1.0000001 + 0.0000001
            x += math.sin(x) * 0.00001
            x += math.cos(x) * 0.00001

        result += x

    return result


def make_chunks(n, p):

    chunks = []

    for worker_id in range(p):
        start = worker_id * n // p
        end = (worker_id + 1) * n // p

        chunks.append((start, end))

    return chunks


def run_pipeline(pool, n, p, stage_a_times):

    stage_a_time, _ = serial_stage(n)

    stage_a_times.append(stage_a_time)

    chunks = make_chunks(n, p)

    pool.map(parallel_worker, chunks)


def measure_pipeline(n, p):

    stage_a_times = []

    with mp.Pool(processes=p) as pool:

        stats = run_benchmark(
            lambda: run_pipeline(
                pool,
                n,
                p,
                stage_a_times
            ),
            warmup_runs=2,
            timed_runs=7
        )

    measured_stage_a = stage_a_times[2:]

    median_stage_a = statistics.median(measured_stage_a)

    return stats.median_ms, median_stage_a


def main():
    print("===== PHASE 2: SCALING =====")

    logical_cores = mp.cpu_count()

    process_counts = range(1, logical_cores + 1)


    print("\n===== STRONG SCALING =====")

    strong_results = {}
    strong_stage_a = {}

    for p in process_counts:
        total_time, stage_a_time = measure_pipeline(
            STRONG_N,
            p
        )

        strong_results[p] = total_time
        strong_stage_a[p] = stage_a_time

        print(
            f"P={p:2d} | "
            f"Time={total_time:.3f} ms | "
            f"Stage A={stage_a_time:.3f} ms"
        )

    strong_baseline = strong_results[1]

    print("\nStrong Scaling Metrics:")

    for p, total_time in strong_results.items():

        speedup = strong_baseline / total_time
        efficiency = speedup / p

        print(
            f"P={p:2d} | "
            f"Speedup={speedup:.3f} | "
            f"Efficiency={efficiency:.3f}"
        )


    print("\n===== WEAK SCALING =====")

    weak_results = {}
    weak_stage_a = {}

    for p in process_counts:

        n = p * WEAK_N0

        total_time, stage_a_time = measure_pipeline(n, p)

        weak_results[p] = total_time
        weak_stage_a[p] = stage_a_time

        print(
            f"P={p:2d} | "
            f"N={n:,} | "
            f"Time={total_time:.3f} ms | "
            f"Stage A={stage_a_time:.3f} ms"
        )

    weak_baseline = weak_results[1]

    print("\nWeak Scaling Metrics:")

    gustafson_last = 0.0

    for p, total_time in weak_results.items():

        efficiency = weak_baseline / total_time

        alpha = weak_stage_a[p] / total_time

        gustafson = p - alpha * (p - 1)

        if p == logical_cores:
            gustafson_last = gustafson

        print(
            f"P={p:2d} | "
            f"Weak Efficiency={efficiency:.3f} | "
            f"Alpha={alpha:.5f} | "
            f"Gustafson={gustafson:.3f}"
        )


    print("\n===== AMDAHL'S LAW =====")

    serial_fraction = (
            strong_stage_a[1] / strong_results[1]
    )

    if serial_fraction > 0:
        max_speedup = 1 / serial_fraction

        theoretical_speedup = 1 / (
            serial_fraction
            + (1 - serial_fraction) / logical_cores
        )
    else:
        max_speedup = float("inf")
        theoretical_speedup = float("inf")

    observed_speedup = (
        strong_results[1] / strong_results[logical_cores]
    )

    print("Serial fraction:", serial_fraction)
    print("Amdahl maximum speedup:", max_speedup)
    print("Theoretical speedup:", theoretical_speedup)
    print("Observed speedup:", observed_speedup)


    print("\n===== VERIFICATION =====")

    print(
        emit_verify_tag(
            "PHASE2",
            STUDENT_ID,
            observed_speedup,
            gustafson_last
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()