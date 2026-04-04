"""
Usage:
  python3 benchmark/benchmark.py --url http://localhost:8000/segment --label A_pytorch_cpu
"""
import requests, time, numpy as np, json, subprocess, argparse

SAMPLE = {
    "meeting_id": "ES2002a",
    "window": [
        {"position": 0, "speaker": "A", "t_start": 98.3,  "t_end": 109.1, "text": "we need to finalize the interface before the next sprint"},
        {"position": 1, "speaker": "B", "t_start": 110.0, "t_end": 121.4, "text": "agreed the api contract should be locked down first"},
        {"position": 2, "speaker": "C", "t_start": 122.0, "t_end": 134.7, "text": "i can have a draft ready by thursday if that works"},
        {"position": 3, "speaker": "A", "t_start": 135.1, "t_end": 147.9, "text": "thursday works should we also loop in the frontend team"},
        {"position": 4, "speaker": "B", "t_start": 165.2, "t_end": 178.4, "text": "actually before that can we revisit the budget numbers"},
        {"position": 5, "speaker": "C", "t_start": 179.0, "t_end": 191.3, "text": "yes the q3 projections changed significantly last week"},
        {"position": 6, "speaker": "A", "t_start": 192.1, "t_end": 204.8, "text": "right we should update the forecast before the board meeting"}
    ],
    "transition_index": 3,
    "meeting_offset_seconds": 98.3
}

def get_vram():
    try:
        r = subprocess.run(
            ["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        return r.stdout.strip() + " MB"
    except:
        return "N/A"

def run_benchmark(url, n=200, label=""):
    print(f"\n{'='*50}")
    print(f"Option:    {label}")
    print(f"URL:       {url}")
    print(f"Requests:  {n}")

    # warmup
    for _ in range(5):
        requests.post(url, json=SAMPLE)

    latencies = []
    for i in range(n):
        t0 = time.perf_counter()
        r  = requests.post(url, json=SAMPLE)
        latencies.append(time.perf_counter() - t0)
        assert r.status_code == 200, f"Request {i} failed: {r.status_code} {r.text}"

    print(f"\n--- Single window latency ---")
    print(f"p50:  {np.percentile(latencies,50)*1000:.1f} ms")
    print(f"p95:  {np.percentile(latencies,95)*1000:.1f} ms")
    print(f"p99:  {np.percentile(latencies,99)*1000:.1f} ms")
    print(f"tput: {n/sum(latencies):.2f} req/s")
    print(f"VRAM: {get_vram()}")

def run_meeting_benchmark(url, n_windows=800, label=""):
    print(f"\n--- Full meeting simulation ({n_windows} windows) ---")
    print("This validates your <5 min SLA...")

    t0 = time.perf_counter()
    for _ in range(n_windows):
        requests.post(url, json=SAMPLE)
    elapsed = time.perf_counter() - t0

    print(f"Total time:  {elapsed:.1f}s")
    print(f"SLA met:     {elapsed < 300}  (limit: 300s)")
    print(f"VRAM peak:   {get_vram()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",   default="http://localhost:8000/segment")
    parser.add_argument("--label", default="unknown")
    parser.add_argument("--n",     type=int, default=200)
    args = parser.parse_args()

    run_benchmark(args.url, n=args.n, label=args.label)
    run_meeting_benchmark(args.url, label=args.label)