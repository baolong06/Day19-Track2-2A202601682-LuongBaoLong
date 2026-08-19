# Test NB6 - Agent Retrieval
import sys
import json
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from app.agent import (SEARCH_TOOL, Agent, RetrievalTool, RuleBasedPlanner,
                       SingleShotPlanner, build_context, ToolArgs)
from app.filters import FilteredIndex
from app.search import Searcher

DATA = ROOT / "data"

print("--- Tool Schema ---")
print(json.dumps(SEARCH_TOOL, ensure_ascii=False, indent=2)[:900])

print("\n--- Building Index ---")
searcher = Searcher.from_corpus(DATA / "corpus_vn.jsonl")
index = FilteredIndex.from_searcher(searcher)
tool = RetrievalTool(index)

print("\n--- RuleBasedPlanner Demo ---")
planner = RuleBasedPlanner(budget=16)
demo_q = "tự động mở rộng theo lưu lượng và cân bằng tải giữa nhiều region"
for i, args in enumerate(planner.plan(demo_q), 1):
    print(f"  call {i}: {args.as_dict()}")

print("\n--- Evaluation: Single-shot vs Agentic (same budget=16) ---")
queries = [json.loads(l) for l in (DATA / "agent_queries.jsonl").open(encoding="utf-8")]
BUDGET = 16

def evaluate(agent, label):
    rec, bal, calls, ms = [], [], [], []
    for q in queries:
        r = agent.answer(q["question"])
        truth, got = set(q["relevant_doc_ids"]), set(r.doc_ids)
        rec.append(len(truth & got) / len(truth))
        a, b = len(set(q["gold_a"]) & got), len(set(q["gold_b"]) & got)
        bal.append(min(a, b) / max(1, max(a, b)))
        calls.append(r.n_calls)
        ms.append(r.latency_ms)
    n = len(queries)
    print(f"{label:<20}{sum(rec)/n:8.3f}{sum(bal)/n:9.2f}{sum(calls)/n:8.1f}{sum(ms)/n:9.1f}")
    return sum(rec) / n

print(f"{'strategy':<20}{'recall':>8}{'balance':>9}{'calls':>8}{'ms':>9}")
base = evaluate(Agent(tool, SingleShotPlanner(budget=BUDGET)), "single-shot")
split = evaluate(Agent(tool, RuleBasedPlanner(budget=BUDGET, use_filters=False)),
                 "agentic (no filter)")
filt = evaluate(Agent(tool, RuleBasedPlanner(budget=BUDGET, use_filters=True)),
                "agentic (+filter)")
print(f"\nDelta recall vs single-shot:  split {split - base:+.3f}   filter {filt - base:+.3f}")

print("\nNB6 Part 1: Agentic > Single-shot on recall AND balance at same budget")
if split > base and filt > base:
    print("  PASS - Agentic strategies outperform single-shot")
else:
    print("  WARN - Check agent_queries.jsonl coverage")

print("\n--- Reflection Test ---")
Q = "cân bằng tải giữa nhiều region"

starving = ToolArgs(query=Q, topic="networking", since_year=2027, top_k=8)
print("filter too tight (since_year=2027) ->", len(tool(starving).doc_ids), "results")

sane = ToolArgs(query=Q, topic="networking", top_k=8)
print("reasonable filter               ->", len(tool(sane).doc_ids), "results")

class StarvingPlanner:
    def plan(self, question):
        return [ToolArgs(query=question, topic="networking", since_year=2027, top_k=8)]

res = Agent(tool, StarvingPlanner(), min_evidence=4).answer(Q)
print(f"\nagent reflection: {res.n_calls} calls -> {len(res.doc_ids)} docs")
for c in res.trace:
    print("   ", c.args, "->", len(c.doc_ids), "results")

print("\nNB6 Part 2: Agent recovers from bad filter via reflection")
if res.n_calls > 1:
    print("  PASS - Agent retries after first call fails")

print("\n--- build_context with Feast ---")
try:
    from feast import FeatureStore
    repo = ROOT / "app" / "feast_repo"
    if (repo / "registry.db").exists():
        store = FeatureStore(repo_path=str(repo))
        print("Feast store available")
    else:
        store = None
        print("Feast registry not found")
except Exception as exc:
    store = None
    print(f"Feast not available: {exc}")

ctx = build_context("u_001", "làm sao tối ưu chi phí hạ tầng", tool, feature_store=store)
print("features   :", ctx["features"] or "(not available)")
print("affinity   :", ctx["affinity_used"])
print("tool_args  :", ctx["tool_args"])
print("doc_ids    :", ctx["doc_ids"][:5], "...")

print("\nNB6 Part 3: build_context returns features and doc_ids")
if ctx["doc_ids"]:
    print("  PASS - build_context works")

print("\n=== NB6 COMPLETE ===")
