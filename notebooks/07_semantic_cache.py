# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB7 — Semantic Cache: threshold / TTL / namespace
#
# **Mục tiêu:** Implement semantic cache over Qdrant, sweep threshold/TTL,
# chứng minh cross-tenant leak + fix bằng namespace.
#
# **3 knobs, 3 failure modes:**
# - **threshold** (cosine sim): quá thấp → FALSE HIT (trả nhầm câu hỏi)
# - **ttl**: missing → STALE HIT (March answer phục vụ August)
# - **namespace**: missing → CROSS-TENANT LEAK (user A receives user B's answer)
#
# **Deck reference:** §6 "Semantic Cache: Hien Thuc Trong 12 Dong" + OWASP LLM08
#
# **Pass when:**
# - Bảng sweep có cả cột "tiết kiệm" (hit rate) **và** "trả lời sai" (false hit rate)
# - Demo leak rồi fix được bằng namespace

# %% [markdown]
# ## 1. Setup — load corpus, build cache
#
# **Cache structure:**
# - `CACHE_COLLECTION = "lab19_semantic_cache"` (Qdrant in-memory)
# - payload: `{tenant, question, answer, ts}`
# - vector: query embedding (384-dim)
# - Query: `query_points(query=embed(question), query_filter=tenant=X, limit=1)`

# %%
import _setup  # noqa: F401
import json
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient

from app.cache import SemanticCache
from app.embeddings import Embedder

ROOT = Path(_setup.__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus_vn.jsonl"

print("Loading embedder directly (cache uses raw embed, not Searcher)...")
embedder = Embedder()
print(f"dim={embedder.dim}")

# %% [markdown]
# ## 2. Seed cache — store 5 question-answer pairs across 2 tenants
#
# Why craft by hand?
# - Need to know **which answer keys exist** to classify hit/hit-but-wrong
# - Need 2+ tenants to demo namespace leak

# %%
seed = [
    # (tenant, question, answer)
    ("acme",    "Cách mở rộng Kubernetes linh hoạt theo lưu lượng?", "K8s HPA + cluster autoscaler"),
    ("acme",    "Làm sao mã hoá dữ liệu nhạy cảm khi lưu trữ?",     "AES-256 at rest + TLS in transit"),
    ("globex",  "Cách mở rộng cloud linh hoạt?",                     "spot instance + auto-scaling group"),
    ("globex",  "Cách bảo vệ dữ liệu an toàn?",                     "encrypt at rest + WAF"),
    ("initech", "Triển khai blue-green deployment?",                 "router switch + health check"),
]

cache = SemanticCache(
    client=QdrantClient(":memory:"),
    embedder=embedder,
    dim=embedder.dim,
    threshold=0.75,
    ttl_s=3600.0,
    namespaced=True,
)
print(f"Created semantic cache (threshold=0.75, ttl=3600s, namespaced=True)")

# Seed: put each (tenant, question, answer) into cache
for tenant, q, a in seed:
    if cache.get(tenant, q) is None:
        cache.put(tenant, q, a)
print(f"Seeded {len(seed)} entries")

# %% [markdown]
# ## 3. Sweep threshold — hit rate vs false-hit rate
#
# **Phân biệt hit types:**
# - **lexical hit**: cached question có cùng exact string (gold answer)
# - **semantic hit**: paraphrase cousin (gold answer vẫn OK)
# - **false hit**: lookup-match nhưng answer SAI (khác intent)
# - **miss**: dưới threshold

# %%
# Build probe set: 3 categories × 2 queries
probes = {
    "lexical": [
        ("acme", "Cách mở rộng Kubernetes linh hoạt theo lưu lượng?",
         "K8s HPA + cluster autoscaler"),  # exact same
    ],
    "semantic": [
        ("acme", "Làm thế nào co giãn K8s theo tải?",
         "K8s HPA + cluster autoscaler"),  # paraphrase → answer vẫn đúng
    ],
    "false_hit": [
        ("acme", "Cách giảm chi phí spot instance?",
         "spot instance + auto-scaling group"),  # similar vector, sai answer
    ],
    "miss": [
        ("acme", "Cách cấu hình WAF?",
         None),  # no cache entry, dưới threshold
    ],
}

# Cache peek — get nearest score without applying threshold/TTL
# Sweep threshold 0.0..1.0
thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
print(f"\n  {'threshold':>10}  {'hit_rate':>8}  {'false_hit_rate':>14}  {'lexical':>8} {'semantic':>9} {'miss':>5}")
print("-" * 65)
for thr in thresholds:
    cache.threshold = thr
    cache.reset_stats()
    correctness = {"lexical": [], "semantic": [], "false_hit": [], "miss": []}
    for category, cases in probes.items():
        for tenant, q, expected in cases:
            if expected is None:
                # Should miss at any threshold
                hit = cache.get(tenant, q)
                correctness[category].append(hit is None)
            else:
                hit = cache.get(tenant, q)
                correctness[category].append(hit is not None and hit.answer == expected)
    n = cache.stats.total
    hit_rate = cache.stats.hits / n if n else 0.0
    # Among HITS, how many were wrong answer?
    fp = 0; tp = 0
    for cat in ("lexical", "semantic", "false_hit"):
        for ok in correctness[cat]:
            if not ok:
                fp += 1
    print(f"  {thr:>10.2f}  {hit_rate:>7.2f}    "
          f"{fp:>3d}/{len(probes['false_hit'])}     "
          f"{sum(correctness['lexical']):>8d} {sum(correctness['semantic']):>9d} {sum(correctness['miss']):>5d}")

# %% [markdown]
# ## 4. TTL — show stale eviction
#
# Cache has virtual clock so TTL testable without sleeping.
# - Query at t=0: hit
# - Advance 2h, set TTL=1h: miss + stale_eviction

# %%
cache.threshold = 0.75
cache.ttl_s = 3600.0  # 1h
cache.reset_stats()

# Seed entry at t=0
cache.put("acme", "Caching test?", "answer_1")
hit = cache.get("acme", "Caching test?")
print(f"At t=0: hit={hit is not None}, age={hit.age_s:.0f}s")

# Advance 30 min — still hit
cache.advance(1800)
hit = cache.get("acme", "Caching test?")
print(f"At t=1800s: hit={hit is not None}, age={hit.age_s:.0f}s")

# Advance 2h past TTL — stale, eviction
cache.advance(7200)
hit = cache.get("acme", "Caching test?")
print(f"At t=9000s: hit={hit is not None}, stale_evictions={cache.stats.stale_evictions}")

# %% [markdown]
# ## 5. Cross-tenant leak — observe then fix
#
# **Setup:** acme + globex have similar questions about "mở rộng cloud"
# - `namespaced=True` (production): acme only sees acme entries
# - `namespaced=False` (vulnerable): acme sees globex's answer

# %%
# Reset cache with fresh entries
cache_ns = SemanticCache(
    client=QdrantClient(":memory:"),
    embedder=embedder,
    dim=embedder.dim,
    threshold=0.75,
    ttl_s=3600.0,
    namespaced=True,
)
cache_vuln = SemanticCache(
    client=QdrantClient(":memory:"),
    embedder=embedder,
    dim=embedder.dim,
    threshold=0.75,
    ttl_s=3600.0,
    namespaced=False,  # vulnerable for demo
)

# Both caches get same seed
for c in (cache_ns, cache_vuln):
    for tenant, q, a in seed:
        c.put(tenant, q, a)

# Acme asks a similar question to globally-cached one
probe_q = "Cách co giãn cloud tự động?"
acme_expected = "K8s HPA + cluster autoscaler"
globex_actual = "spot instance + auto-scaling group"

hit_ns = cache_ns.get("acme", probe_q)
hit_vuln = cache_vuln.get("acme", probe_q)

print(f"\nProbe: acme asks '{probe_q}'")
print(f"  namespaced=True:  answer={hit_ns.answer if hit_ns else 'MISS'}")
print(f"  namespaced=False: answer={hit_vuln.answer if hit_vuln else 'MISS'}")
print(f"  Expected acme: {acme_expected}")
print(f"  Globex leaked: {globex_actual}")
print(f"  LEAK detected: {hit_vuln is not None and hit_vuln.answer == globex_actual}")

# %% [markdown]
# ## Diễn giải kết quả
#
# **Học từ output cell 3 (threshold sweep):**
# - Threshold thấp (0.50): hit rate cao, **NHƯNG** false_hit rate cao
#   → cache rẻ (less compute) nhưng sai (wrong answer)
# - Threshold cao (0.95): hit rate thấp, false_hit = 0
#   → cache đắt (re-compute) nhưng correct
# - Sweet spot (0.75-0.85): balance giữa savings và safety
# - **Lexical vs semantic**: lexical ổn định ở mọi threshold; semantic cần threshold vừa
#
# **Học từ output cell 4 (TTL):**
# - TTL=3600s: cache hit tại t=0, t=1800s; miss + eviction tại t=9000s
# - Production: TTL phải thấp hơn "drift rate" của knowledge base
# - Wikipedia: 30 ngày. Documentation: 7-30 ngày. AI answers: 1-7 ngày
#
# **Học từ output cell 5 (cross-tenant leak):**
# - namespaced=False: acme nhận được answer của globex → SECURITY INCIDENT
# - namespaced=True: filter tenant trong query → correct isolation
# - **Đây không phải caching bug — đây là data breach**
#
# **Khi nào dùng gì?**
# - Stable FAQ, low drift: threshold 0.75, TTL 7-30d, namespaced=True
# - Real-time Q&A (news, prices): threshold 0.85, TTL 1h, namespaced=True
# - Multi-tenant SaaS: **bắt buộc** namespaced=True + tenant claim from JWT
# - Single-tenant internal: có thể relax namespace nhưng vẫn TTL
#
# **Think about:**
# - Tại sao threshold sweep cần measure CẢ hit rate VÀ false hit rate?
#   → Optimization 1 chiều (max hit rate) thiển cận → cache sai = mất user trust
# - TTL vs invalidation?
#   → TTL: simple, time-based; dùng được khi drift đều
#   → Invalidation: explicit (set/update/delete entry); tốn code, chính xác hơn
# - Cache poisoning?
#   → Attacker submits question → put answer → other users get attacker answer
#   → Defense: signed answers + admin-only writes
#
# **Bài học:**
# 1. Semantic cache là correctness-security trade-off, không chỉ performance
# 2. Threshold sweep cần 2D metric (hit + false)
# 3. Namespace leak = data breach, không phải bug
# 4. TTL virtual clock giúp test mà không cần sleep

# %% [markdown]
# ## Deliverable evidence
# 1. Output cell 3: threshold sweep table (hit_rate + false_hit_rate)
# 2. Output cell 4: TTL demo (hit at t=0,1800; miss+eviction at t=9000)
# 3. Output cell 5: cross-tenant leak demo (vulnerable vs namespaced)
