"""
Triton Inference Server benchmark
Usage:
  python3 benchmark/benchmark_triton.py --url localhost:8100 --label C_triton_gpu
  python3 benchmark/benchmark_triton.py --url localhost:8100 --label C_triton_gpu --concurrency 5
"""
import numpy as np
import tritonclient.http as httpclient
import time
import concurrent.futures
import argparse
import subprocess

def get_vram():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        return r.stdout.strip() + " MB"
    except:
        return "N/A"

def get_cpu_ram():
    try:
        mem = subprocess.run(["free", "-m"], capture_output=True, text=True)
        return mem.stdout.split('\n')[1].split()[2] + " MB RAM used"
    except:
        return "N/A"

def make_inputs():
    input_ids = np.ones((1, 512), dtype=np.int64)
    attention_mask = np.ones((1, 512), dtype=np.int64)
    inputs = [
        httpclient.InferInput('input_ids', [1, 512], 'INT64'),
        httpclient.InferInput('attention_mask', [1, 512], 'INT64'),
    ]
    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(attention_mask)
    outputs = [httpclient.InferRequestedOutput('logits')]
    return inputs, outputs

def run_triton_benchmark(url, n=200, concurrency=1, label=""):
    client = httpclient.InferenceServerClient(url=url)
    inputs, outputs = make_inputs()

    # Warmup
    for _ in range(5):
        client.infer('roberta_segmenter', inputs, outputs=outputs)

    print(f"\n{'='*50}")
    print(f"Option:      {label}")
    print(f"URL:         {url}")
    print(f"Requests:    {n}")
    print(f"Concurrency: {concurrency}")

    if concurrency == 1:
        latencies = []
        errors = 0
        for _ in range(n):
            t0 = time.perf_counter()
            try:
                client.infer('roberta_segmenter', inputs, outputs=outputs)
                latencies.append(time.perf_counter() - t0)
            except:
                errors += 1

        print(f"\n--- Single client latency ---")
        print(f"median: {np.median(latencies)*1000:.1f} ms")
        print(f"p50:  {np.percentile(latencies, 50)*1000:.1f} ms")
        print(f"p95:  {np.percentile(latencies, 95)*1000:.1f} ms")
        print(f"p99:  {np.percentile(latencies, 99)*1000:.1f} ms")
        print(f"tput: {n/sum(latencies):.2f} req/s")
        print(f"error rate: {errors}/{n} = {100*errors/n:.1f}%")
        print(f"VRAM: {get_vram()}")
        print(f"RAM:  {get_cpu_ram()}")
    else:
        # Each thread gets its own client to avoid blocking
        def send_request(_):
            thread_client = httpclient.InferenceServerClient(url=url)
            thread_inputs, thread_outputs = make_inputs()
            t0 = time.perf_counter()
            thread_client.infer('roberta_segmenter', thread_inputs, outputs=thread_outputs)
            return time.perf_counter() - t0

        t_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            latencies = list(executor.map(send_request, range(n)))
        total_time = time.perf_counter() - t_start

        print(f"\n--- Concurrent ({concurrency} workers) latency ---")
        print(f"median: {np.median(latencies)*1000:.1f} ms")
        print(f"p50:  {np.percentile(latencies, 50)*1000:.1f} ms")
        print(f"p95:  {np.percentile(latencies, 95)*1000:.1f} ms")
        print(f"p99:  {np.percentile(latencies, 99)*1000:.1f} ms")
        print(f"tput: {n/total_time:.2f} req/s")
        print(f"Total time: {total_time:.1f}s")
        print(f"VRAM: {get_vram()}")
        print(f"RAM:  {get_cpu_ram()}")

    # Full meeting simulation
    print(f"\n--- Full meeting simulation (800 windows) ---")
    t0 = time.perf_counter()
    for _ in range(800):
        client.infer('roberta_segmenter', inputs, outputs=outputs)
    elapsed = time.perf_counter() - t0
    print(f"Total time:  {elapsed:.1f}s")
    print(f"SLA met:     {elapsed < 300}  (limit: 300s)")
    print(f"VRAM peak:   {get_vram()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="localhost:8100")
    parser.add_argument("--label", default="C_triton_gpu")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    run_triton_benchmark(args.url, n=args.n, concurrency=args.concurrency, label=args.label)