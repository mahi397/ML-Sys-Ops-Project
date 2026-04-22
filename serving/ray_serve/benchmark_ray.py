"""
Benchmark for Ray Serve + Monitoring stack.
Generates traffic to populate Grafana dashboards.

Usage:
  python3 benchmark.py --url http://localhost:8000 --n 200
  python3 benchmark.py --url http://localhost:8000 --n 200 --concurrency 5
  python3 benchmark.py --url http://localhost:8000 --n 50 --summarize
"""

import requests
import time
import numpy as np
import argparse
import subprocess
import concurrent.futures

# Same test payload as your existing benchmark_ray.py
SAMPLE_SEGMENT = {
    "meeting_id": "ES2002a",
    "window": [
        {"position": 0, "speaker": "A", "t_start": 98.3, "t_end": 109.1, "text": "we need to finalize the interface before the next sprint"},
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

SAMPLE_SUMMARIZE = {
    "meeting_id": "ES2002a",
    "segment_id": 1,
    "t_start": 98.3,
    "t_end": 204.8,
    "utterances": [
        {"speaker": "A", "text": "we need to finalize the interface before the next sprint"},
        {"speaker": "B", "text": "agreed the api contract should be locked down first"},
        {"speaker": "C", "text": "i can have a draft ready by thursday if that works"},
        {"speaker": "A", "text": "thursday works should we also loop in the frontend team"},
        {"speaker": "B", "text": "actually before that can we revisit the budget numbers"}
    ],
    "total_utterances": 5,
    "meeting_context": {"total_segments": 3, "segment_index_in_meeting": 1}
}


def get_vram():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        return r.stdout.strip() + " MB"
    except:
        return "N/A"


def run_benchmark(url, endpoint, payload, n, concurrency=1):
    full_url = f"{url}/{endpoint}"
    print(f"\n{'='*60}")
    print(f"Endpoint:    /{endpoint}")
    print(f"URL:         {full_url}")
    print(f"Requests:    {n}")
    print(f"Concurrency: {concurrency}")
    print(f"{'='*60}")

    # Warmup
    print("Warming up (5 requests)...")
    for _ in range(5):
        try:
            requests.post(full_url, json=payload, timeout=60)
        except:
            pass

    def send_one(_):
        t0 = time.perf_counter()
        try:
            r = requests.post(full_url, json=payload, timeout=60)
            return time.perf_counter() - t0, r.status_code
        except Exception as e:
            return time.perf_counter() - t0, 0

    t_start = time.perf_counter()
    if concurrency == 1:
        results = [send_one(i) for i in range(n)]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            results = list(ex.map(send_one, range(n)))
    total_time = time.perf_counter() - t_start

    latencies = [r[0] for r in results if r[1] == 200]
    errors = sum(1 for r in results if r[1] != 200)

    if latencies:
        latencies_arr = np.array(latencies)
        print(f"\n  Results ({len(latencies)}/{n} successful, {errors} errors):")
        print(f"    p50:  {np.percentile(latencies_arr, 50)*1000:.1f} ms")
        print(f"    p95:  {np.percentile(latencies_arr, 95)*1000:.1f} ms")
        print(f"    p99:  {np.percentile(latencies_arr, 99)*1000:.1f} ms")
        print(f"    mean: {np.mean(latencies_arr)*1000:.1f} ms")
        tput = n / total_time if concurrency > 1 else len(latencies) / sum(latencies)
        print(f"    throughput: {tput:.2f} req/s")
        print(f"    VRAM: {get_vram()}")

        if endpoint == "segment":
            p95 = np.percentile(latencies_arr, 95) * 1000
            print(f"    SLA check: p95={p95:.1f}ms {'< 2000ms ' if p95 < 2000 else '> 2000ms '}")
    else:
        print(f"\n  All {n} requests failed!")
        if results:
            print(f"  Status codes: {set(r[1] for r in results)}")


def run_full_meeting(url, num_windows=800):
    print(f"\n{'='*60}")
    print(f"Full Meeting Simulation: {num_windows} windows")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    errors = 0
    for i in range(num_windows):
        try:
            r = requests.post(f"{url}/segment", json=SAMPLE_SEGMENT, timeout=60)
            if r.status_code != 200:
                errors += 1
        except:
            errors += 1
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{num_windows}] elapsed: {time.perf_counter()-t0:.1f}s")

    total = time.perf_counter() - t0
    print(f"\n  Total: {total:.1f}s  (SLA: 300s)")
    print(f"  Errors: {errors}/{num_windows}")
    print(f"  {'PASS' if total < 300 else 'FAIL'}")
    print(f"  VRAM peak: {get_vram()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--summarize", action="store_true", help="Also benchmark /summarize")
    parser.add_argument("--full-meeting", action="store_true", help="Run 800-window meeting sim")
    parser.add_argument("--recap", action="store_true", help="Test /recap endpoint (full pipeline)")
    args = parser.parse_args()

    # Health check
    try:
        h = requests.get(f"{args.url}/health", timeout=5)
        print(f"Health: {h.json()}")
    except Exception as e:
        print(f"Cannot reach {args.url}/health — {e}")
        print("Is the stack running? Try: docker compose logs ray-serve")
        exit(1)

    # Segment benchmark
    run_benchmark(args.url, "segment", SAMPLE_SEGMENT, args.n, args.concurrency)

    # Summarize benchmark
    if args.summarize:
        run_benchmark(args.url, "summarize", SAMPLE_SUMMARIZE, min(args.n, 20), 1)

    # Full meeting
    if args.full_meeting:
        run_full_meeting(args.url)

    # Recap endpoint (full pipeline: segment + summarize)
    if args.recap:
        print(f"\n{'='*60}")
        print("Recap Pipeline Test (full meeting → segments → summaries)")
        print(f"{'='*60}")
        recap_payload = {
            "meeting_id": "ES2002a",
            "utterances": [
                {"speaker": "A", "t_start": 0.0, "t_end": 10.0, "text": "welcome everyone to the quarterly review meeting"},
                {"speaker": "B", "t_start": 10.5, "t_end": 20.0, "text": "thanks for having us lets get started with the numbers"},
                {"speaker": "A", "t_start": 20.5, "t_end": 30.0, "text": "revenue is up twelve percent quarter over quarter"},
                {"speaker": "C", "t_start": 30.5, "t_end": 40.0, "text": "marketing spend was under budget by eight percent"},
                {"speaker": "B", "t_start": 40.5, "t_end": 50.0, "text": "the cost savings came from reducing paid social"},
                {"speaker": "A", "t_start": 50.5, "t_end": 60.0, "text": "great now lets move to the engineering roadmap"},
                {"speaker": "C", "t_start": 60.5, "t_end": 70.0, "text": "we have three features planned for next sprint"},
                {"speaker": "C", "t_start": 70.5, "t_end": 80.0, "text": "first is the authentication refactor"},
                {"speaker": "B", "t_start": 80.5, "t_end": 90.0, "text": "second is the new dashboard for analytics"},
                {"speaker": "A", "t_start": 90.5, "t_end": 100.0, "text": "and third is improving the search functionality"},
                {"speaker": "B", "t_start": 100.5, "t_end": 110.0, "text": "timeline for all three is six weeks"},
                {"speaker": "A", "t_start": 110.5, "t_end": 120.0, "text": "any blockers we should discuss now"}
            ]
        }
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{args.url}/recap", json=recap_payload, timeout=600)
            elapsed = time.perf_counter() - t0
            if r.status_code == 200:
                result = r.json()
                print(f"\n  Recap generated in {elapsed:.1f}s")
                print(f"  Segments found: {result.get('total_segments', '?')}")
                print(f"  Processing time: {result.get('processing_time_seconds', '?')}s")
                print(f"  SLA (300s): {'PASS' if elapsed < 300 else 'FAIL'}")
                for seg in result.get("recap", []):
                    print(f"\n  Segment {seg.get('segment_id', '?')}: {seg.get('topic_label', '(no label)')}")
                    for b in seg.get("summary_bullets", []):
                        print(f"    • {b}")
            else:
                print(f"\n  Recap failed: {r.status_code} — {r.text[:500]}")
        except Exception as e:
            print(f"\n  Recap error: {e}")

    print(f"\n{'='*60}")
    print("Done! Open Grafana: http://<FLOATING_IP>:3000  (admin / admin)")
    print(f"{'='*60}")