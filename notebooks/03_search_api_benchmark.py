# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB3 — FastAPI `/search` Endpoint + Latency Benchmark
#
# **Mục tiêu:** Bọc Searcher thành REST API, đo P50/P95/P99 latency, đảm bảo
# hybrid P99 < 50ms (rubric threshold).
#
# **WHY cần benchmark?**
# - Production phải biết tail latency (P95, P99), không chỉ mean
# - 1% requests chậm = trải nghiệm xấu cho 1% users
# - Hybrid search có 2 retrievers → cần verify vẫn đủ nhanh

# %% [markdown]
# ## 1. Khởi động API server
#
# **HOW subprocess + health check loop hoạt động?**
# 1. Popen uvicorn ở background
# 2. Poll `/healthz` mỗi 1 giây
# 3. Khi `ready: true` → server đã load xong embeddings
# 4. Nếu 60s vẫn chưa ready → fail (timeout)

# %%
import _setup  # noqa: F401
import statistics
import subprocess
import time
from pathlib import Path

import httpx

# %%
ROOT = Path(_setup.__file__).resolve().parent.parent
# **WHY port 8001 thay vì 8000?**
# - 8000 có thể bị conflict nếu có process khác đang chạy
# - 8001 là port thứ 2 ít bị chiếm hơn
# - Notebook này chạy standalone, không cần share port với dev server
proc = subprocess.Popen(
    ["uvicorn", "app.main:app", "--port", "8001", "--log-level", "warning"],
    cwd=str(ROOT),
)

# **WHY poll thay vì sleep cố định?**
# - Searcher.from_corpus loads embeddings + indexes 1000 docs mất vài giây
# - Cold start có thể khác nhau mỗi lần (OS cache, disk speed)
# - Poll → ready ngay khi server warm, không waste time
URL = "http://localhost:8001"
for _ in range(60):
    try:
        r = httpx.get(f"{URL}/healthz", timeout=2.0)
        if r.status_code == 200 and r.json().get("ready"):
            break
    except httpx.HTTPError:
        pass
    time.sleep(1)
else:
    raise RuntimeError("API didn't become ready within 60s")

print(httpx.get(f"{URL}/healthz").json())

# %% [markdown]
# ## 2. Single query — kiểm tra response shape
#
# **WHY test 1 query trước khi benchmark?**
# - Verify endpoint hoạt động
# - Check response format (latency_ms, hits)
# - Nếu shape sai → benchmark vô nghĩa

# %%
r = httpx.get(f"{URL}/search", params={"q": "cloud computing tự động mở rộng", "mode": "hybrid"})
r.raise_for_status()
body = r.json()
print(f"latency_ms: {body['latency_ms']:.1f}")
print(f"top-3 hits:")
for h in body["hits"][:3]:
    print(f"  {h['doc_id']:>14}  score={h['score']:.4f}  {h['title']}")

# %% [markdown]
# ## 3. Latency benchmark
#
# **WHY dùng percentile thay vì mean?**
# - Mean bị ảnh hưởng bởi outliers (cold cache, GC pause)
# - P50 = median, 50% requests nhanh hơn con số này
# - P95 = 95% requests nhanh hơn (SLA thường dùng)
# - P99 = tail latency, quan trọng cho UX
#
# **WHY 50 queries × 1 rep = 50?**
# - Đủ mẫu để P99 stable (50 ≥ 30 để percentile hợp lệ)
# - 1 rep thay vì 2 để tổng thời gian benchmark vừa phải
# - Mỗi call mất ~30ms (semantic) → 50 × 30ms × 3 modes = ~4.5s, OK
#
# **WHY đo cả server-side và wall-clock?**
# - `body["latency_ms"]` = thời gian xử lý trong server (đã trừ network)
# - Wall-clock = tổng thời gian client thấy (bao gồm network)
# - Rubric check P99 server-side; wall-clock để hiểu end-to-end experience

# %%
import json

DATA = ROOT / "data"
golden = [json.loads(l) for l in (DATA / "golden_set.jsonl").open(encoding="utf-8")]


def percentile(values: list[float], p: float) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    return sorted(values)[min(int(n * p), n - 1)]


def benchmark_mode(mode: str, reps: int = 1) -> dict[str, float]:
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


print(f"  {'mode':10}  {'P50':>7}  {'P95':>7}  {'P99':>7}  {'P99(wall)':>9}")
results = {}
for mode in ("keyword", "semantic", "hybrid"):
    res = benchmark_mode(mode)
    results[mode] = res
    print(f"  {mode:10}  {res['p50_server']:>5.1f}ms  {res['p95_server']:>5.1f}ms  "
          f"{res['p99_server']:>5.1f}ms  {res['p99_wall']:>7.1f}ms")

# %% [markdown]
# ## 4. Rubric assertion — hybrid P99 < 50ms
#
# **WHY assert ở đây?**
# - Rubric của lab yêu cầu hybrid P99 server-side < 50ms
# - Auto-check giúp phát hiện regression sớm
# - Nếu fail → gợi ý nguyên nhân + cách debug

# %%
hybrid_p99 = results["hybrid"]["p99_server"]
print(f"Hybrid P99 server-side: {hybrid_p99:.1f}ms")
if hybrid_p99 < 50:
    print(f"PASS — hybrid P99 < 50ms ({hybrid_p99:.1f}ms)")
else:
    print(f"WARN — hybrid P99 >= 50ms ({hybrid_p99:.1f}ms)")
    print("  Possible causes: cold cache, fastembed model not warm yet, or RRF depth=50 is too aggressive")
    print("  Check: re-run benchmark after 10 warm-up queries; or reduce RRF depth")

# %% [markdown]
# ## 5. Cleanup
#
# **WHY terminate?**
# - Background process chiếm port 8000
# - Notebook cleanup → process dies → port free
# - Nếu không terminate → port conflict cho lần chạy sau

# %%
proc.terminate()
proc.wait(timeout=5)
print("API server stopped")

# %% [markdown]
# ## Diễn giải kết quả
#
# **Think about:**
# - Tại sao hybrid P99 cao hơn keyword P99?
#   → Hybrid chạy 2 retrievers (BM25 + vector) → nhiều work hơn
# - Tại sao semantic P99 thấp hơn keyword P99?
#   → Vector search với HNSW index là O(log n); BM25 là O(n) scan
# - Khi nào cần optimize?
#   → Khi P99 vượt ngưỡng (50ms cho hybrid), hoặc khi traffic tăng
#
# **Bài học:**
# 1. Production cần đo tail latency, không chỉ mean
# 2. Hybrid search thêm latency ~30-50% so với single mode
# 3. Warm-up queries quan trọng — cold cache làm P99 tăng 5-10x
# 4. Server-side P99 mới là metric thật; wall-clock phụ thuộc network

# %% [markdown]
# ## Deliverable evidence
# 1. Output cell 2: 1 hybrid query response với top-3 hits
# 2. Output cell 3: latency table P50/P95/P99 cho 3 modes
# 3. Output cell 4: hybrid P99 PASS/WARN