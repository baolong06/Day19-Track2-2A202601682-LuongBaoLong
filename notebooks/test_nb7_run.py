# Test NB7 - Semantic Cache
import sys
import json
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from app.cache import SemanticCache

DATA = ROOT / "data"
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
client = QdrantClient(":memory:")

print("--- Basic Hit/Miss Test ---")
cache = SemanticCache(client=client, embedder=embedder, threshold=0.75, ttl_s=3600)
cache.put("acme", "làm sao tối ưu chi phí cloud", "Dùng spot instance và autoscaling.")

for probe in ["làm sao tối ưu chi phí cloud",        # y hệt
              "cách giảm chi phí hạ tầng đám mây",   # diễn đạt khác
              "cách kiểm thử unit test"]:            # chủ đề khác hẳn
    hit = cache.get("acme", probe)
    print(f"{probe:<38} -> {'HIT  ' + f'{hit.score:.3f}' if hit else 'MISS'}")

print("\n--- Threshold Sweep ---")
golden = [json.loads(l) for l in (DATA / "golden_set.jsonl").open(encoding="utf-8")]
warm, cold = golden[::2], golden[1::2]

sweep = SemanticCache(client=client, embedder=embedder, threshold=0.0, ttl_s=None)
for g in warm:
    sweep.put("acme", g["query"], f"ANSWER::{g['query_id']}")

def variants(q: str) -> list[str]:
    w = q.split()
    return [f"cho tôi hỏi {q}",
            f"{q} thì làm thế nào",
            " ".join(w[:-1]) if len(w) > 2 else q]

positives = []
for g in warm:
    for v in variants(g["query"]):
        p = sweep.peek("acme", v)
        if p:
            positives.append((p[0], p[1]["question"] == g["query"]))

negatives = []
for g in cold:
    for v in variants(g["query"]):
        p = sweep.peek("acme", v)
        if p:
            negatives.append(p[0])

print(f"cache: {len(warm)} queries   probes: {len(positives)} positive / {len(negatives)} negative\n")
print(f"{'threshold':>8}{'saved':>12}{'wrong':>14}   {'note':<10}")
for th in (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
    saved = sum(1 for sc, ok in positives if sc >= th and ok) / len(positives)
    wrong = sum(1 for sc in negatives if sc >= th) / len(negatives)
    flag = "DANGEROUS" if wrong > 0.20 else ("too tight" if saved < 0.80 else "balanced")
    print(f"{th:>8.2f}{saved:>12.0%}{wrong:>14.0%}   {flag:<10}")

print("\nNB7 Part 1: Threshold sweep shows savings vs wrong answers tradeoff")
print("  Expected: 0.75 is a starting point, not a universal constant")

print("\n--- TTL Test ---")
ttl_cache = SemanticCache(client=client, embedder=embedder, threshold=0.75, ttl_s=1800)
ttl_cache.put("acme", "giá GPU hiện tại là bao nhiêu", "Khoảng $2/giờ cho A100.")

for jump in (0, 600, 3600):
    ttl_cache.advance(jump)
    hit = ttl_cache.get("acme", "giá GPU hiện tại là bao nhiêu")
    print(f"t = {ttl_cache._clock:>6.0f}s  -> {'HIT' if hit else 'MISS (expired)'}")

print(f"\nstale evictions: {ttl_cache.stats.stale_evictions}")
print("\nNB7 Part 2: TTL expires entries after configured time")

print("\n--- Tenant Leak Test ---")
leaky = SemanticCache(client=client, embedder=embedder, threshold=0.70,
                      ttl_s=None, namespaced=False)
leaky.put("acme", "doanh thu quý 3 của chúng tôi",
          "Doanh thu ACME quý 3: 4,2 tỷ VND.")

stolen = leaky.get("globex", "doanh thu quý 3 của chúng tôi")
print("namespaced=False -> GLOBEX receives:")
print("   ", stolen.answer if stolen else "(miss)")
print("    real owner:", stolen.tenant if stolen else "-")

safe = SemanticCache(client=client, embedder=embedder, threshold=0.70,
                     ttl_s=None, namespaced=True)
safe.put("acme", "doanh thu quý 3 của chúng tôi", "Doanh thu ACME quý 3: 4,2 tỷ VND.")
blocked = safe.get("globex", "doanh thu quý 3 của chúng tôi")
print("\nnamespaced=True  -> GLOBEX receives:", blocked.answer if blocked else "MISS (correct)")

print("\nNB7 Part 3: Tenant isolation")
if stolen:
    print("  LEAK DETECTED - tenant isolation required for multi-tenant systems")
else:
    print("  OK - no leak in this configuration")

print("\n=== NB7 COMPLETE ===")
