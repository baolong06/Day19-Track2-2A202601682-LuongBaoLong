import sys, time, json
sys.path.insert(0, "e:/AI_thucchien/lab/Day19-Track2-2A202601682-LuongBaoLong")

from pathlib import Path
from app.search import Searcher

CORPUS = Path("e:/AI_thucchien/lab/Day19-Track2-2A202601682-LuongBaoLong/data/corpus_vn.jsonl")
GOLDEN = Path("e:/AI_thucchien/lab/Day19-Track2-2A202601682-LuongBaoLong/data/golden_set.jsonl")

print("Loading Searcher...")
t0 = time.perf_counter()
s = Searcher.from_corpus(CORPUS)
print(f"Loaded in {time.perf_counter()-t0:.1f}s, {s.size} docs")

golden = [json.loads(l) for l in GOLDEN.open(encoding="utf-8")]

# Warm up
print("Warming up (20 queries)...")
for q in golden[:20]:
    s.search(q["query"], mode="hybrid")

print("Benchmarking (direct, depth=20, warm)...")
def percentile(values, p):
    n = len(values)
    if n == 0: return 0.0
    return sorted(values)[min(int(n * p), n - 1)]

for mode in ("keyword", "semantic", "hybrid"):
    times = []
    for _ in range(3):
        for q in golden:
            t0 = time.perf_counter()
            s.search(q["query"], mode=mode)
            times.append((time.perf_counter() - t0) * 1000)
    p99 = percentile(times, 0.99)
    status = "PASS" if p99 < 50 else "FAIL"
    print(f"  {mode:10} P50={percentile(times,0.50):5.1f}ms P95={percentile(times,0.95):5.1f}ms P99={p99:5.1f}ms [{status}] n={len(times)}")
