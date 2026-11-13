import multiprocessing as mp
import platform
import sys
import time

from harness import emit_verify_tag


def main():
    print("===== PHASE 0: HARDWARE AUDIT =====")

    print("Operating System:", platform.platform())
    print("Python Version:", sys.version)

    logical_cores = mp.cpu_count()

    print("Logical CPU cores:", logical_cores)

    total_reads = 2_000_000

    start = time.perf_counter_ns()

    for _ in range(total_reads):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    total_time = end - start
    average_latency = total_time / total_reads

    print("Total clock reads:", total_reads)
    print("Total time (ns):", total_time)
    print("Average timer latency (ns):", average_latency)

    student_id = 230103225

    print(
        emit_verify_tag(
            "PHASE0",
            student_id,
            average_latency,
            total_time / 1e6
        )
    )


if __name__ == "__main__":
    main()