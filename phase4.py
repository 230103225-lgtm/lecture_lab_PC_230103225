import multiprocessing as mp
import time
import math

from harness import emit_verify_tag


STUDENT_ID = 230103225

N = 12_000

CHUNK_SIZES = [1, 4, 16, 64, 256, 1024]


def calculate_item(i):

    k = 1 + int((i / 60) ** 2)

    result = 0.0

    for j in range(k):
        x = i * 0.0001 + j * 0.00001 + 1.0

        result += math.sin(x)
        result += math.cos(x)
        result += math.sqrt(x)

    return result


def process_items(indices):

    result = 0.0

    for i in indices:
        result += calculate_item(i)

    return result


def block_worker(worker_id, p, times):
    start = worker_id * N // p
    end = (worker_id + 1) * N // p

    start_time = time.perf_counter()

    process_items(range(start, end))

    end_time = time.perf_counter()

    times[worker_id] = end_time - start_time


def run_block(p):
    times = mp.Array("d", p)

    processes = []

    wall_start = time.perf_counter()

    for worker_id in range(p):
        process = mp.Process(
            target=block_worker,
            args=(worker_id, p, times)
        )

        processes.append(process)
        process.start()

    for process in processes:
        process.join()

        if process.exitcode != 0:
            raise RuntimeError("Block worker failed")

    wall_end = time.perf_counter()

    wall_time = wall_end - wall_start

    worker_times = list(times)

    return wall_time, worker_times



def cyclic_worker(worker_id, p, times):
    start_time = time.perf_counter()

    indices = range(worker_id, N, p)

    process_items(indices)

    end_time = time.perf_counter()

    times[worker_id] = end_time - start_time


def run_cyclic(p):
    times = mp.Array("d", p)

    processes = []

    wall_start = time.perf_counter()

    for worker_id in range(p):
        process = mp.Process(
            target=cyclic_worker,
            args=(worker_id, p, times)
        )

        processes.append(process)
        process.start()

    for process in processes:
        process.join()

        if process.exitcode != 0:
            raise RuntimeError("Cyclic worker failed")

    wall_end = time.perf_counter()

    wall_time = wall_end - wall_start

    worker_times = list(times)

    return wall_time, worker_times



def dynamic_worker(worker_id, counter, lock, chunk_size, times):
    start_time = time.perf_counter()

    while True:

        with lock:
            start = counter.value
            counter.value += chunk_size

        if start >= N:
            break

        end = min(start + chunk_size, N)

        process_items(range(start, end))

    end_time = time.perf_counter()

    times[worker_id] = end_time - start_time


def run_dynamic(p, chunk_size):
    times = mp.Array("d", p)

    counter = mp.Value("i", 0)
    lock = mp.Lock()

    processes = []

    wall_start = time.perf_counter()

    for worker_id in range(p):
        process = mp.Process(
            target=dynamic_worker,
            args=(
                worker_id,
                counter,
                lock,
                chunk_size,
                times
            )
        )

        processes.append(process)
        process.start()

    for process in processes:
        process.join()

        if process.exitcode != 0:
            raise RuntimeError("Dynamic worker failed")

    wall_end = time.perf_counter()

    wall_time = wall_end - wall_start

    worker_times = list(times)

    return wall_time, worker_times



def calculate_lambda(worker_times):
    average_time = sum(worker_times) / len(worker_times)

    if average_time == 0:
        return 0.0

    return max(worker_times) / average_time



def main():
    print("===== PHASE 4: SCHEDULING =====")

    p = 4

    print("Workers:", p)
    print("Total items:", N)

    results = {}


    print("\n===== BLOCK STATIC =====")

    block_wall, block_times = run_block(p)

    block_lambda = calculate_lambda(block_times)

    results["block"] = block_wall

    print("Wall time:", block_wall, "seconds")
    print("Worker times:", block_times)
    print("Lambda:", block_lambda)


    print("\n===== CYCLIC STATIC =====")

    cyclic_wall, cyclic_times = run_cyclic(p)

    cyclic_lambda = calculate_lambda(cyclic_times)

    results["cyclic"] = cyclic_wall

    print("Wall time:", cyclic_wall, "seconds")
    print("Worker times:", cyclic_times)
    print("Lambda:", cyclic_lambda)


    print("\n===== DYNAMIC CHUNK QUEUE =====")

    dynamic_results = {}

    for chunk_size in CHUNK_SIZES:

        wall_time, worker_times = run_dynamic(
            p,
            chunk_size
        )

        imbalance = calculate_lambda(worker_times)

        dynamic_results[chunk_size] = wall_time

        print(
            f"Chunk={chunk_size:4d} | "
            f"Wall={wall_time:.3f} sec | "
            f"Lambda={imbalance:.5f} | "
            f"Workers={worker_times}"
        )


    best_chunk = min(
        dynamic_results,
        key=dynamic_results.get
    )

    best_dynamic_time = dynamic_results[best_chunk]

    print("\n===== BEST RESULT =====")

    print("Best chunk size:", best_chunk)
    print("Best dynamic time:", best_dynamic_time)


    print("\n===== VERIFICATION =====")

    print(
        emit_verify_tag(
            "PHASE4",
            STUDENT_ID,
            block_wall * 1000,
            best_dynamic_time * 1000
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()