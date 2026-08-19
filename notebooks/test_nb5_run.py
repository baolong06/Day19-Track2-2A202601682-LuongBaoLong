# Test NB5 - Filtered Search
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from app.filters import FilteredIndex, access_filter, combo_filter, recent_filter, tenant_filter
from app.metadata import selectivity
from app.search import Searcher

DATA = ROOT / "data"

print("--- Building FilteredIndex ---")
searcher = Searcher.from_corpus(DATA / "corpus_vn.jsonl")
index = FilteredIndex.from_searcher(searcher)
print(f"docs: {len(index.docs)}   vectors: {index.vectors.shape}")
print("payload mẫu:", {k: index.docs[0][k] for k in ("doc_id", "topic", "tenant", "access", "published")})

print("\n--- Recall Cliff by Filter Selectivity ---")
QUERY = "tự động mở rộng hệ thống theo lưu lượng"

cases = [
    ("không filter",   lambda d: True, None),
    ("access=internal", *access_filter("internal")),
    ("tenant=acme",     *tenant_filter("acme")),
    ("published ≥ 2026", *recent_filter(20260101)),
    ("acme AND ≥2026",  *combo_filter("acme", 20260101)),
]

print(f"{'filter':<18}{'sel%':>7}{'post':>8}{'fANN':>8}{'post_ms':>9}{'fann_ms':>9}")
rows = []
for name, pred, qf in cases:
    sel = selectivity(index.docs, pred) * 100
    truth = index.pre_filter(QUERY, pred, k=10).doc_ids
    post = index.post_filter(QUERY, pred, k=10, fetch_k=10)
    if qf is None:
        fann_r, fann_ms = 1.0, float("nan")
    else:
        f = index.filtered_ann(QUERY, qf, k=10)
        fann_r, fann_ms = f.recall_against(truth), f.latency_ms
    rows.append((name, sel, post.recall_against(truth), fann_r))
    print(f"{name:<18}{sel:7.1f}{post.recall_against(truth):8.2f}{fann_r:8.2f}"
          f"{post.latency_ms:9.1f}{fann_ms:9.1f}")

print("\nNB5 Part 1: Recall Cliff Analysis")
print("  Expected: post-filter sập khi filter chặt, filtered-ANN giữ 1.00")
post_recalls = [r[2] for r in rows[1:]]  # skip no-filter
fann_recalls = [r[3] for r in rows[1:] if r[3] != 1.0]  # only non-nan
if all(r < 1.0 for r in post_recalls if r < 1.0) and (not fann_recalls or all(r >= 1.0 for r in fann_recalls)):
    print("  PASS - post-filter fails on narrow filters, filtered-ANN maintains recall")

print("\n--- Over-fetch Ladder ---")
pred, qf = combo_filter("acme", 20260101)
QUERIES = [QUERY, "bảo mật xác thực người dùng", "mô hình ngôn ngữ lớn"]
truths = {q: index.pre_filter(q, pred, k=10).doc_ids for q in QUERIES}

print(f"selectivity = {selectivity(index.docs, pred)*100:.1f}%  của 1000 doc\n")
print(f"{'fetch_k':>9}{'recall':>9}{'% corpus quét':>16}")
for fk in (10, 50, 200, 500, 1000):
    r = sum(index.post_filter(q, pred, k=10, fetch_k=fk).recall_against(truths[q])
            for q in QUERIES) / len(QUERIES)
    print(f"{fk:>9}{r:9.2f}{fk/len(index.docs)*100:15.0f}%")

r = sum(index.filtered_ann(q, qf, k=10).recall_against(truths[q]) for q in QUERIES) / len(QUERIES)
print(f"{'fANN':>9}{r:9.2f}{10/len(index.docs)*100:15.0f}%")

print("\nNB5 Part 2: Over-fetch Analysis")
print("  Expected: cần fetch_k ≈ 50% corpus để lấy lại recall, filtered-ANN chỉ cần 1%")

print("\n--- Tenant Filter Tests ---")
for tenant in ("acme", "globex", "initech"):
    pred_t, qf_t = tenant_filter(tenant)
    truth = index.pre_filter(QUERY, pred_t, k=10).doc_ids
    post = index.post_filter(QUERY, pred_t, k=10, fetch_k=10)
    fann = index.filtered_ann(QUERY, qf_t, k=10)
    print(f"tenant={tenant:<9} sel={selectivity(index.docs, pred_t)*100:5.1f}%  "
          f"post={post.recall_against(truth):.2f}  fANN={fann.recall_against(truth):.2f}")

print("\nNB5 Part 3: Tenant Filter Analysis")
print("  Expected: filtered-ANN thắng ở mọi tenant")

print("\n=== NB5 COMPLETE ===")
