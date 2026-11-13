import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes

from harness import run_benchmark, emit_verify_tag


M = 24_000_000
STUDENT_ID = 230103225


def fs_worker(shm_name, slot_idx, iters):

    shm = shared_memory.SharedMemory(name=shm_name)

    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

    acc = arr[slot_idx]

    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005

    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc

    del arr
    shm.close()


def run_false_sharing(p, stride):

    shm = shared_memory.SharedMemory(
        create=True,
        size=4096
    )

    processes = []

    try:
        arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

        for i in range(512):
            arr[i] = 1

        del arr

        iterations = M // p

        for t in range(p):
            slot_idx = t * stride

            process = mp.Process(
                target=fs_worker,
                args=(shm.name, slot_idx, iterations)
            )

            processes.append(process)
            process.start()

        for process in processes:
            process.join()

            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker завершился с кодом {process.exitcode}"
                )

    finally:
        shm.close()
        shm.unlink()


def benchmark_case(p, stride):
    return run_benchmark(
        lambda: run_false_sharing(p, stride),
        warmup_runs=2,
        timed_runs=7
    )


def main():
    print("===== PHASE 1: FALSE SHARING =====")

    physical_cores = mp.cpu_count()

    physical_cores = 4
    logical_cores = mp.cpu_count()

    print("Physical cores:", physical_cores)
    print("Logical cores:", logical_cores)


    print("\n===== TABLE 1A: STRIDE SWEEP =====")

    stride_values = [1, 2, 4, 8, 16, 32]

    stride_results = {}

    for stride in stride_values:
        stats = benchmark_case(physical_cores, stride)

        throughput = M / stats.median_ms / 1000

        stride_results[stride] = stats.median_ms

        print(
            f"Stride={stride:2d} | "
            f"Distance={stride * 8:3d} bytes | "
            f"Median={stats.median_ms:.3f} ms | "
            f"IQR={stats.iqr_ms:.3f} ms | "
            f"Throughput={throughput:.3f} Mops/s"
        )


    print("\n===== TABLE 1B: PROCESS SCALING =====")

    process_counts = [
        p for p in [1, 2, 4, 8, 16, 32]
        if p <= logical_cores
    ]

    scaling_results = {}

    for stride in [1, 16]:
        print(f"\n--- STRIDE = {stride} ---")

        times = {}

        for p in process_counts:
            stats = benchmark_case(p, stride)
            times[p] = stats.median_ms

        baseline = times[1]

        for p in process_counts:
            current_time = times[p]

            speedup = baseline / current_time
            efficiency = speedup / p

            if p >= 2:
                karp_flatt = (
                    (1 / speedup) - (1 / p)
                ) / (1 - (1 / p))
            else:
                karp_flatt = 0.0

            print(
                f"Processes={p:2d} | "
                f"Time={current_time:.3f} ms | "
                f"Speedup={speedup:.3f} | "
                f"Efficiency={efficiency:.3f} | "
                f"Karp-Flatt={karp_flatt:.5f}"
            )

        scaling_results[stride] = times


    time_stride_1 = stride_results[1]
    time_stride_16 = stride_results[16]

    print("\n===== VERIFICATION =====")

    print(
        emit_verify_tag(
            "PHASE1",
            STUDENT_ID,
            time_stride_1,
            time_stride_16
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()