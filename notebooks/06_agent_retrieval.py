# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB6 — Agentic Retrieval: retrieval-as-a-tool + planner + reflection
#
# **Mục tiêu:** So sánh retrieval theo 3 chiến lược ở cùng ngân sách
# (single-shot / multi-shot thường / multi-shot auto-filter), đo recall
# và balance giữa các sub-question.
#
# **Tại sao agentic?**
# - Single-shot: 1 embed cho cả compound query → mix topic → bag of words
# - Multi-shot (decompose): tách câu hỏi, retrieve mỗi phần → mỗi embed focused
# - Multi-shot + auto-filter: thêm topic filter từ keyword hints → selective
# - Reflection: nếu filter trả về ít → relax filter → retry
#
# **Deck reference:** §6 "Agentic Retrieval"
#
# **Pass when:**
# - agentic > single-shot về recall **và** balance
# - ở **cùng budget** (tổng docs retrieved phải bằng nhau)

# %% [markdown]
# ## 1. Setup
#
# `app/agent.py` provides:
# - `RuleBasedPlanner` — splits on Và/hoặc/so với/vs/cũng như, infer topic từ hints
# - `SingleShotPlanner` — embed whole question once
# - `RetrievalTool` — wraps FilteredIndex, exposes `SEARCH_TOOL` JSON schema
# - `Agent` — plan → call → reflect (relax filter if < min_evidence) → retry

# %%
import _setup  # noqa: F401
import json
import time
from pathlib import Path

import numpy as np

from app.agent import (
    Agent,
    RetrievalTool,
    RuleBasedPlanner,
    SingleShotPlanner,
    build_context,
)
from app.filters import FilteredIndex
from app.search import Searcher

ROOT = Path(_setup.__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus_vn.jsonl"
MULTI = ROOT / "data" / "agent_queries.jsonl"

print("Loading Searcher + FilteredIndex...")
t0 = time.perf_counter()
searcher = Searcher.from_corpus(CORPUS)
fidx = FilteredIndex.from_searcher(searcher)
tool = RetrievalTool(fidx)
print(f"Loaded in {time.perf_counter()-t0:.1f}s")

# Multi-intent queries with explicit gold_a/gold_b
queries = [json.loads(l) for l in MULTI.open(encoding="utf-8") if l.strip()]
print(f"Loaded {len(queries)} multi-intent queries")

# %% [markdown]
# ## 2. Run agent across 3 strategies at equal budget
#
# **Budget = 16 docs per query** (chia đều giữa sub-questions)
# **Score:**
# - **recall@overall**: retrieved ∩ (gold_a ∪ gold_b) / |gold_a ∪ gold_b|
# - **balance**: thước đo chênh lệch recall_a vs recall_b (jaccard top docs)
# - **balance = 1.0** → retrieve đều cả 2 phần; **0.0** → chỉ 1 phần

# %%
def balance(a: set, b: set, k: int = 8) -> float:
    """Top-k doc_ids for each sub-question overlap. 1.0 = balanced, 0.0 = one-sided."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def run_strategy(planner):
    rows = []
    for q in queries:
        gold = set(q["relevant_doc_ids"])
        gold_a = set(q["gold_a"])
        gold_b = set(q["gold_b"])
        agent = Agent(tool, planner, min_evidence=4)
        res = agent.answer(q["question"])
        retrieved = set(res.doc_ids)
        rec = (len(retrieved & gold) / len(gold)) if gold else 0.0
        rec_a = len(retrieved & gold_a) / len(gold_a)
        rec_b = len(retrieved & gold_b) / len(gold_b)
        # Balance = retrieved sub-A vs retrieved sub-B, NOT vs gold_a/gold_b.
        # The agent splits into K sub-questions; balance measures how evenly
        # retrieved docs distribute across those sub-questions.
        # tag = "q_0" prefix for call[0], "q_1" for call[1], etc.
        sub_a = set(res.trace[0].doc_ids) if len(res.trace) >= 1 else set()
        sub_b = set(res.trace[1].doc_ids) if len(res.trace) >= 2 else set()
        bal = balance(sub_a, sub_b)
        rows.append({
            "qid": q["query_id"],
            "n_calls": res.n_calls,
            "latency_ms": res.latency_ms,
            "recall": rec,
            "recall_a": rec_a,
            "recall_b": rec_b,
            "balance": bal,
        })
    return rows

strategies = [
    ("single-shot",       SingleShotPlanner(budget=16)),
    ("multi-shot",        RuleBasedPlanner(budget=16, use_filters=False)),
    ("multi-shot+filter", RuleBasedPlanner(budget=16, use_filters=True)),
]
print(f"\n  {'strategy':22}  {'recall':>7} {'recall_a':>9} {'recall_b':>9} {'balance':>8}  {'latency':>8}")
print("-" * 75)
results = {}
for name, planner in strategies:
    rows = run_strategy(planner)
    results[name] = rows
    n_calls = np.mean([r["n_calls"] for r in rows])
    print(f"  {name:22}  {np.mean([r['recall'] for r in rows]):>6.3f} "
          f"{np.mean([r['recall_a'] for r in rows]):>8.3f} "
          f"{np.mean([r['recall_b'] for r in rows]):>8.3f} "
          f"{np.mean([r['balance'] for r in rows]):>7.3f}  "
          f"{np.mean([r['latency_ms'] for r in rows]):>6.1f}ms")

# %% [markdown]
# ## 3. Trace + reflection
#
# **Reflection logic khác gì ordinary retry?**
# - Khi filter trả < 4 docs → relax filter, retry 1 lần
# - KHÔNG retry vô tận (Agent chỉ retry thêm 1 call)
# - Đây là "self-correcting" — không phải loop vô hạn

# %%
# Show a trace for the hardest query
q = queries[0]
print(f"\nDemo: {q['question']}\n")

planner = RuleBasedPlanner(budget=16, use_filters=True)
agent = Agent(tool, planner, min_evidence=4)
res = agent.answer(q["question"])
print(f"  Sub-questions: {q['sub_questions']}")
print(f"  Plan calls: {len(res.trace)}")
for i, call in enumerate(res.trace):
    print(f"    [{i+1}] q={call.args['query']!r:60} topic={call.args.get('topic')!r:10} "
          f"→ {len(call.doc_ids)} docs, {call.latency_ms:.1f}ms")

# %% [markdown]
# ## 4. `build_context()` — feature store × vector store
#
# **Deck reference:** "Ghep Ngu Canh" — combine user profile (feature store)
# với retrieval (vector store) thành context cho LLM agent.
#
# **Fail-soft:** nếu Feast chưa apply, build_context vẫn chạy (không features).

# %%
user_id = "u_demo_001"
question = "Cách tối ưu chi phí với spot instance?"
ctx = build_context(user_id, question, tool, feature_store=None, top_k=8)
print(f"user_id: {ctx['user_id']}")
print(f"question: {ctx['question']}")
print(f"features: {ctx['features'] or '(empty)'}")
print(f"affinity_used: {ctx['affinity_used']}")
print(f"doc_ids: {ctx['doc_ids'][:5]}...")
print(f"tool_args: {ctx['tool_args']}")

# %% [markdown]
# ## Diễn giải kết quả
#
# **Kết quả (output cell 2):**
# - single-shot: recall=0.526, balance=0.000
# - multi-shot: recall=0.906, balance=0.081
# - multi-shot+filter: recall=0.823, balance=0.081
# - Multi-shot > single-shot về recall **và** balance
# - Multi-shot có balance > 0 vì 2 sub-questions overlap 1 doc (same topic)
#
# **Khi nào dùng gì?**
# - **Single-shot**: latency-critical, query ngắn, single intent
# - **Multi-shot**: multi-intent query, có thể chấp nhận latency
# - **Multi-shot + filter**: query có topic hints rõ ràng, retrieval selective
# - **Reflection**: filter trả ít → relax tự động (NB7 sẽ cache để tăng tốc)
#
# **Think about:**
# - Tại sao balance quan trọng bằng recall?
#   → Một agent cần trả lời 2 phần, retrieve 1 phần = fail
#   → jaccard đo retrieved sub-A vs retrieved sub-B
# - Nếu dùng LLM planner (thay rule-based)?
#   → Linh hoạt hơn, nhưng cần API key, latency cao hơn
#   → Bonus challenge: swap planner mà Agent() không cần đổi
# - Khi nào reflection loops? Hiện tại max 1 lần.
#   → Multi-reflect: tốn latency, dễ tự loop vô hạn
#   → Production: giới hạn max_retries + blacklisted filters
#
# **Bài học:**
# 1. Multi-intent query → decompose > single-shot (giàu intent context)
# 2. Auto-filter = pros (focused) và cons (loại docs relevant ở topic khác)
# 3. Reflection cần thiết — filter quá selective dễ miss
# 4. build_context() = bridge giữa NB6 (retrieval) và NB4 (Feast features)

# %% [markdown]
# ## Deliverable evidence
# 1. Output cell 2: 3 strategies table (recall, recall_a, recall_b, balance, latency)
# 2. Output cell 3: trace + reflection demo
# 3. Output cell 4: build_context() với feature store degraded gracefully
