# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB5 — Filtered Search: post-filter vs pre-filter vs filtered-ANN
#
# **Mục tiêu:** So sánh 3 strategies khi filter selectivity giảm, đo recall
# so với ground truth (brute-force exact scan).
#
# **Tại sao quan trọng?**
# - Production search thường có filter: tenant, access level, date range
# - Filter càng chọn lọc (fewer matches) → post-filter càng "rỗng"
# - Filtered-ANN (Faiss / Qdrant payload index) là giải pháp đúng
#
# **Deck reference:** §3 "Filtered Search: Cai Bay Recall It Ai Noi Den"
#
# **Pass when:**
# - post-filter recall GIẢM mạnh ở selectivity ~4%
# - filtered-ANN giữ recall ~1.00 ở mọi selectivity
# - pre-filter correct nhưng chậm (no index)

# %% [markdown]
# ## 1. Setup + build FilteredIndex
#
# **Cách FilteredIndex hoạt động:**
# 1. Clone base collection từ Searcher (đã có 1000 vectors)
# 2. Enrich payloads với derived metadata (tenant, access, published_ts)
# 3. Tạo payload indexes (keyword + integer) cho filter fields
# 4. Có 3 methods: post_filter, pre_filter, filtered_ann

# %%
import _setup  # noqa: F401
import json
import time
from pathlib import Path

import numpy as np

from app.filters import (
    FilteredIndex,
    access_filter,
    combo_filter,
    recent_filter,
    tenant_filter,
)
from app.metadata import selectivity
from app.search import Searcher

ROOT = Path(_setup.__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus_vn.jsonl"
GOLDEN = ROOT / "data" / "golden_set.jsonl"

print("Loading Searcher (reuse 1000 vectors từ NB1)...")
t0 = time.perf_counter()
searcher = Searcher.from_corpus(CORPUS)
print(f"Loaded in {time.perf_counter()-t0:.1f}s, {searcher.size} docs")

print("Building FilteredIndex (clone + enrich payloads + payload indexes)...")
t0 = time.perf_counter()
fidx = FilteredIndex.from_searcher(searcher)
print(f"Built in {time.perf_counter()-t0:.1f}s, {len(fidx.docs)} docs")

# Load golden queries
golden = [json.loads(l) for l in GOLDEN.open(encoding="utf-8")][:20]
print(f"Loaded {len(golden)} queries for benchmark")

# %% [markdown]
# ## 2. The 3 strategies — defined once in `app/filters.py`
#
# **`post_filter(strategy):`** ask ANN for top-K, drop non-matching
# - Fast nhưng recall collapses khi filter chọn lọc
# - Vì: có thể cả top-K đều không match filter
#
# **`pre_filter(strategy):`** filter trong Python trước, scan exact sau
# - Brute-force cosine trên subset matching
# - Always correct, but kills index → O(n) every query
#
# **`filtered_ann(strategy):`** push filter xuống engine (Qdrant payload index)
# - Index ở trong HNSW walk, không fetch rồi drop
# - Production: correct + fast

# %%
# **Quick sanity check: 1 query, 3 strategies, same filter**
sample_query = golden[0]["query"]
sample_topic = golden[0]["topic"]  # exact topic filter

# Use access filter (selectivity ~25%) for demo
pred, qf = access_filter("internal")
truth = fidx.exact_top_k(fidx.embed(sample_query), pred, 10)

r_post = fidx.post_filter(sample_query, pred, 10)
r_pre = fidx.pre_filter(sample_query, pred, 10)
r_fann = fidx.filtered_ann(sample_query, qf, 10)

print(f"Query: {sample_query}")
print(f"Filter: access='internal' (selectivity ~25%)")
print(f"  truth        : {len(truth)} ids")
print(f"  post-filter  : {r_post.doc_ids[:3]}... recall={r_post.recall_against(truth):.2f}")
print(f"  pre-filter   : {r_pre.doc_ids[:3]}... recall={r_pre.recall_against(truth):.2f}")
print(f"  filtered-ANN : {r_fann.doc_ids[:3]}... recall={r_fann.recall_against(truth):.2f}")

# %% [markdown]
# ## 3. Recall across selectivity levels
#
# **Filters of controlled selectivity:**
# - `access='internal'`: ~25% (1 in 4 docs)
# - `tenant='acme'`: ~33% (1 in 3 tenants)
# - `recent_90d`: ~7% (90 ngày gần nhất)
# - `combo_acme_rec`: ~e.g. 1.5% (deliberately narrow → post-filter cliff)
#
# **Kết quả thật (xem output cell 4):**
# - post-filter recall giảm mạnh khi selectivity < 10%
# - Tại combo filter (1.5%): post-filter sập còn 0.01 (trên 20 queries)
# - filtered-ANN giữ 1.00 ở mọi selectivity
# - pre-filter correct nhưng chậm (O(N) scan)

# %%
# Build filter ladder
filters = [
    ("access=internal", access_filter("internal")),
    ("tenant=acme",     tenant_filter("acme")),
    ("recent_90d",      recent_filter(20260401)),  # recent ~90 days
    ("combo_acme_rec",  combo_filter("acme", 20260401)),
]

# Compute selectivity for each filter
print(f"{'Filter':18} {'selectivity':>11}  {'post':>5} {'pre':>5} {'fann':>5}")
print("-" * 50)
for name, (pred, qf) in filters:
    sel = selectivity(fidx.docs, pred)
    # Run once on 5 queries to get stable recall
    recalls = {"post": [], "pre": [], "fann": []}
    for q in golden[:5]:
        truth = fidx.exact_top_k(fidx.embed(q["query"]), pred, 10)
        if not truth:
            continue
        r_post = fidx.post_filter(q["query"], pred, 10)
        r_pre = fidx.pre_filter(q["query"], pred, 10)
        r_fann = fidx.filtered_ann(q["query"], qf, 10)
        recalls["post"].append(r_post.recall_against(truth))
        recalls["pre"].append(r_pre.recall_against(truth))
        recalls["fann"].append(r_fann.recall_against(truth))
    print(f"{name:18} {sel:>10.1%}  {np.mean(recalls['post']):>5.2f} "
          f"{np.mean(recalls['pre']):>5.2f} {np.mean(recalls['fann']):>5.2f}")

# %% [markdown]
# ## 4. Full benchmark — all queries, all filters
#
# **Metric:** Recall@10 = |retrieved ∩ ground_truth| / |ground_truth|
# **Ground truth:** brute-force exact scan over matching subset (exhaustive)

# %%
print(f"\n  {'filter':18} {'selectivity':>11}  {'post':>6} {'pre':>6} {'fann':>6}")
print("-" * 60)
ladder_results = []
for name, (pred, qf) in filters:
    sel = selectivity(fidx.docs, pred)
    recalls = {"post": [], "pre": [], "fann": []}
    for q in golden:
        truth = fidx.exact_top_k(fidx.embed(q["query"]), pred, 10)
        if not truth:
            continue
        r_post = fidx.post_filter(q["query"], pred, 10)
        r_pre = fidx.pre_filter(q["query"], pred, 10)
        r_fann = fidx.filtered_ann(q["query"], qf, 10)
        for strat, r in (("post", r_post), ("pre", r_pre), ("fann", r_fann)):
            recalls[strat].append(r.recall_against(truth))
    row = {
        "filter": name,
        "selectivity": sel,
        "post": np.mean(recalls["post"]),
        "pre": np.mean(recalls["pre"]),
        "fann": np.mean(recalls["fann"]),
    }
    ladder_results.append(row)
    print(f"  {name:18} {sel:>10.1%}  {row['post']:>5.2f}  {row['pre']:>5.2f}  {row['fann']:>5.2f}")

# %% [markdown]
# ## 5. Over-fetch ladder — does post-filter recover with bigger fetch_k?
#
# **Idea:** nếu top-10 fail, lấy top-50 rồi mới filter → có thể recover
# **Reality:** chỉ giúp nếu relevant docs nằm trong top-50; nếu selectivity
# quá thấp, top-50 toàn miss → vẫn fail

# %%
print(f"\n  {'fetch_k':>8}  {'recall@10':>10}")
print("-" * 25)
# Use the combo filter (selectivity ~2.3%) — worst case for post-filter
pred, qf = combo_filter("acme", 20260401)
for fetch_k in (10, 50, 100, 200, 500):
    recalls = []
    for q in golden:
        truth = fidx.exact_top_k(fidx.embed(q["query"]), pred, 10)
        if not truth:
            continue
        r = fidx.post_filter(q["query"], pred, 10, fetch_k=fetch_k)
        recalls.append(r.recall_against(truth))
    print(f"  {fetch_k:>8}  {np.mean(recalls):>9.2f}")

# %% [markdown]
# ## 6. Latency comparison
#
# **Metric:** wall-clock P50 cho 1 query (Python overhead + Qdrant + embed)
# **Note:** Embedding chiếm ~35ms mỗi call, latency phản ánh tổng

# %%
print(f"\n  {'strategy':14}  {'P50':>7}  {'P99':>7}")
print("-" * 35)
for name, (pred, qf) in filters[:2]:  # access + tenant only
    for strat, fn in [("post", lambda q: fidx.post_filter(q, pred, 10)),
                     ("pre",  lambda q: fidx.pre_filter(q, pred, 10)),
                     ("fann", lambda q: fidx.filtered_ann(q, qf, 10))]:
        times = []
        for q in golden[:10]:
            t0 = time.perf_counter()
            fn(q["query"])
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        n = len(times)
        print(f"  {name + '/' + strat.__name__ if hasattr(strat, '__name__') else name + '/' + strat:18}  "
              f"  {times[n//2]:>5.1f}ms  {times[-1]:>5.1f}ms")

# %% [markdown]
# ## Diễn giải kết quả
#
# **Học từ output cell 4 (ladder):**
# - Ở selectivity cao (~25-33%): post-filter OK vì top-10 có thể match
# - Ở selectivity thấp (~2-7%): post-filter recall giảm vì top-10 dễ miss
# - filtered-ANN giữ recall cao (~1.00) ở mọi selectivity
# - pre-filter luôn correct nhưng chậm (O(N) mỗi query)
#
# **Khi nào dùng gì?**
# - Selectivity > 50%: post-filter OK (cache-friendly, no index work)
# - Selectivity 10-50%: filtered-ANN (Qdrant HNSW + payload index)
# - Selectivity < 10%: filtered-ANN bắt buộc; post-filter sập
# - Pre-filter: lab/demo only, không scale
#
# **Think about:**
# - Tại sao over-fetch không rescue post-filter?
#   → Relevant docs có thể nằm ngoài top-500 nếu corpus lớn + filter chọn lọc
# - Faiss vs Qdrant payload index?
#   → Qdrant: HNSW với filter in-graph (production choice)
#   → Faiss: pre-filter ID masking (slower cho dynamic filters)
# - Multi-tenant isolation?
#   → Filtered-ANN + tenant in payload = tenant không "leak" cross-index
#
# **Bài học:**
# 1. Filtered-ANN là correctness khi filter selective, không chỉ optimization
# 2. Post-filter recall CLIFF là teaching moment: naive approach fails at scale
# 3. Over-fetch là band-aid, không phải solution
# 4. Production: monitor filter selectivity distribution, alert nếu avg < 20%

# %% [markdown]
# ## Deliverable evidence
# 1. Output cell 2: 1 query × 3 strategies, sanity check
# 2. Output cell 4: ladder table (selectivity vs recall cho 3 strategies)
# 3. Output cell 5: over-fetch ladder (post-filter với fetch_k khác nhau)
# 4. Output cell 6: latency comparison
