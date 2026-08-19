# Test NB3 - FastAPI Benchmark
import sys
import time
import json
import subprocess
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx

DATA = ROOT / "data"
URL = "http://localhost:8000"

def percentile(values: list[float], p: float) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    return sorted(values)[min(int(n * p), n - 1)]

def benchmark_mode(mode: str, reps: int = 2) -> dict[str, float]:
    server_latencies: list[float] = []
    wall_latencies: list[float] = []
    for _ in range(reps):
        for q in golden:
            t0 = time.perf_counter()
            r = httpx.get(f"{URL}/search", params={"q": q["query"], "mode": mode})
            wall_latencies.append((time.perf_counter() - t0) * 1000)
            server_latencies.append(r.json()["latency_ms"])
    return {
        "p50_server": percentile(server_latencies, 0.50),
        "p95_server": percentile(server_latencies, 0.95),
        "p99_server": percentile(server_latencies, 0.99),
        "p99_wall":   percentile(wall_latencies, 0.99),
    }

# Load golden set
golden = [json.loads(l) for l in (DATA / "golden_set.jsonl").open(encoding="utf-8")]
print(f"Loaded {len(golden)} golden queries")

# Test single query first
print("\n--- Single Query Test ---")
r = httpx.get(f"{URL}/search", params={"q": "cloud computing tự động mở rộng", "mode": "hybrid"})
r.raise_for_status()
body = r.json()
print(f"latency_ms: {body['latency_ms']:.1f}")
print(f"top-3 hits:")
for h in body["hits"][:3]:
    print(f"  {h['doc_id']:>14}  score={h['score']:.4f}  {h['title']}")

# Benchmark all modes
print("\n--- Latency Benchmark (100 queries x 3 modes) ---")
print(f"  {'mode':10}  {'P50':>7}  {'P95':>7}  {'P99':>7}  {'P99(wall)':>9}")
results = {}
for mode in ("keyword", "semantic", "hybrid"):
    res = benchmark_mode(mode)
    results[mode] = res
    print(f"  {mode:10}  {res['p50_server']:>5.1f}ms  {res['p95_server']:>5.1f}ms  "
          f"{res['p99_server']:>5.1f}ms  {res['p99_wall']:>7.1f}ms")

# Check rubric
hybrid_p99 = results["hybrid"]["p99_server"]
print(f"\n--- Rubric Check ---")
print(f"Hybrid P99 server-side: {hybrid_p99:.1f}ms")
if hybrid_p99 < 50:
    print(f"PASS - hybrid P99 < 50ms ({hybrid_p99:.1f}ms)")
else:
    print(f"WARN - hybrid P99 >= 50ms ({hybrid_p99:.1f}ms)")
    print("  Possible causes: cold cache, first run, or RRF depth=50 too aggressive")
    print("  Check: re-run benchmark after warm-up; or reduce RRF depth")

print("\n=== NB3 COMPLETE ===")
