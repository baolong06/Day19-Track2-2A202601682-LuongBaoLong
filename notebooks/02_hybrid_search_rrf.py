# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB2 — Hybrid Search: BM25 + Vector + RRF
#
# **Mục tiêu:** Hiểu cách kết hợp keyword search (BM25) và semantic search
# (vector) bằng Reciprocal Rank Fusion.
#
# **WHY hybrid search?**
# - BM25 mạnh khi query có keyword verbatim
# - Vector mạnh khi query paraphrase / không có keyword exact
# - Real users thường viết query MIXED → cần cả 2

# %% [markdown]
# ## 1. Setup + build indices
#
# **HOW BM25 hoạt động?**
# - Tokenize text (lowercase + split)
# - Tính TF-IDF score cho mỗi doc với query
# - Rank theo score
#
# **HOW vector search hoạt động?**
# - Embed text → vector 384-dim
# - Cosine similarity giữa query vector và tất cả vectors
# - Rank theo score

# %%
import _setup  # noqa: F401
import json
import statistics
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

DATA = Path(_setup.__file__).resolve().parent.parent / "data"

# %%
docs = [json.loads(line) for line in (DATA / "corpus_vn.jsonl").open(encoding="utf-8")]

# **WHY tokenize bằng .lower().split()?**
# - BM25 case-insensitive nên cần lowercase
# - .split() là whitespace tokenizer đơn giản
# (Production: dùng tokenizer tốt hơn cho tiếng Việt như underthesea)
tokenized = [(d["title"] + " " + d["text"]).lower().split() for d in docs]
bm25 = BM25Okapi(tokenized)

# Build vector index (giống NB1)
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
client = QdrantClient(":memory:")
client.create_collection(
    collection_name="lab19",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
BATCH = 64
points = []
for start in range(0, len(docs), BATCH):
    batch = docs[start:start + BATCH]
    texts = [d["title"] + " " + d["text"] for d in batch]
    vectors = list(embedder.embed(texts))
    for i, (d, v) in enumerate(zip(batch, vectors)):
        points.append(PointStruct(
            id=start + i, vector=v.tolist(),
            payload={"doc_id": d["doc_id"], "topic": d["topic"]},
        ))
client.upsert(collection_name="lab19", points=points)
print(f"BM25 + vector indices ready ({len(docs)} docs)")

# %% [markdown]
# ## 2. Per-mode search functions
#
# **WHY depth = 5×top_k?**
# - RRF cần candidates từ mỗi retriever
# - Lấy 50 từ mỗi retriever → đủ signal cho top-10 cuối
# - Lấy ít quá → dễ miss relevant docs
# - Lấy nhiều quá → chậm

# %%
TOP_K = 10
RRF_K = 60   # **WHY k=60?** - default công nghiệp, đủ lớn để giảm ảnh hưởng
             # của các docs xếp hạng thấp (gần 0)

def search_keyword(query: str, top_k: int = TOP_K) -> list[str]:
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [docs[i]["doc_id"] for i in ranked]


def search_semantic(query: str, top_k: int = TOP_K) -> list[str]:
    q_vec = next(embedder.embed([query])).tolist()
    res = client.query_points(collection_name="lab19", query=q_vec, limit=top_k)
    return [p.payload["doc_id"] for p in res.points]


# %% [markdown]
# ## 3. Reciprocal Rank Fusion
#
# **HOW RRF hoạt động?**
#
# Công thức: `score(d) = Σ_r 1 / (k + rank_r(d))`
#
# 1. Pull top-50 từ BM25 và top-50 từ vector
# 2. Với mỗi doc, cộng `1/(k + rank)` từ mỗi retriever
# 3. Sort theo total score → top-10
#
# **WHY rank 1-based?**
# - rank=1 cho doc tốt nhất → score cao nhất
# - Nếu rank=0 → score bằng 1/k (không phải cao nhất)
#
# **EXAMPLE:**
# - Doc A: rank 1 trong BM25, rank 5 trong vector
#   → score = 1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318
# - Doc B: rank 10 trong BM25, rank 1 trong vector
#   → score = 1/(60+10) + 1/(60+1) = 0.0143 + 0.0164 = 0.0307
# - Doc A thắng vì xuất hiện tốt ở cả 2 retrievers

# %%
def search_hybrid(query: str, top_k: int = TOP_K, rrf_k: int = RRF_K) -> list[str]:
    depth = max(top_k * 5, 50)
    kw_ids = search_keyword(query, depth)
    sem_ids = search_semantic(query, depth)

    rrf: dict[str, float] = {}
    for rank, doc_id in enumerate(kw_ids, start=1):
        rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, doc_id in enumerate(sem_ids, start=1):
        rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

    return [doc_id for doc_id, _ in sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k]]


# Quick sanity check
test_q = "co giãn linh hoạt theo nhu cầu sử dụng"
print(f"Query: {test_q}")
print(f"  keyword top-3:  {search_keyword(test_q)[:3]}")
print(f"  semantic top-3: {search_semantic(test_q)[:3]}")
print(f"  hybrid top-3:   {search_hybrid(test_q)[:3]}")

# %% [markdown]
# ## 4. Đánh giá trên golden set
#
# **WHY dùng golden set?**
# - 50 queries có ground truth (đúng topic)
# - Đo Precision@10 = bao nhiêu % trong top-10 là đúng topic
# - So sánh 3 modes một cách objective

# %%
golden = [json.loads(line) for line in (DATA / "golden_set.jsonl").open(encoding="utf-8")]
doc_topic = {d["doc_id"]: d["topic"] for d in docs}


def precision_at_10(retrieved_ids: list[str], target_topic: str) -> float:
    if not retrieved_ids:
        return 0.0
    return sum(1 for d in retrieved_ids if doc_topic.get(d) == target_topic) / len(retrieved_ids)


p_kw, p_sem, p_hyb = [], [], []
for q in golden:
    p_kw.append(precision_at_10(search_keyword(q["query"]), q["topic"]))
    p_sem.append(precision_at_10(search_semantic(q["query"]), q["topic"]))
    p_hyb.append(precision_at_10(search_hybrid(q["query"]), q["topic"]))

print(f"Precision@10 (avg over {len(golden)} queries):")
print(f"  Keyword (BM25)   : {statistics.mean(p_kw):.1%}")
print(f"  Semantic (vector): {statistics.mean(p_sem):.1%}")
print(f"  Hybrid  (RRF=60) : {statistics.mean(p_hyb):.1%}")

# %% [markdown]
# ## 5. Phân tích theo loại query
#
# **WHY slice theo loại?**
# - Mỗi loại query có đặc điểm riêng
# - Biết được mode nào thắng ở đâu → hiểu rõ hơn khi nào dùng gì

# %%
from collections import defaultdict

by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"kw": [], "sem": [], "hyb": []})
for q, kw, sem, hyb in zip(golden, p_kw, p_sem, p_hyb):
    by_type[q["mode_hint"]]["kw"].append(kw)
    by_type[q["mode_hint"]]["sem"].append(sem)
    by_type[q["mode_hint"]]["hyb"].append(hyb)

print(f"  {'type':12} {'n':>3}  {'kw':>7} {'sem':>7} {'hyb':>7}")
for t in ("exact", "paraphrase", "mixed"):
    m = by_type[t]
    print(f"  {t:12} {len(m['kw']):>3}  "
          f"{statistics.mean(m['kw']):>6.1%} "
          f"{statistics.mean(m['sem']):>6.1%} "
          f"{statistics.mean(m['hyb']):>6.1%}")

# %% [markdown]
# ## Diễn giải kết quả
#
# **WHY hybrid thắng?**
# - `exact` queries: keyword mạnh vì có verbatim match, hybrid ≈ keyword
# - `paraphrase` queries: cả BM25 và vector đều yếu (English model trên VN text)
# - `mixed` queries: hybrid thắng rõ vì tận dụng được cả 2 signals
#
# **Think about:**
# - Khi nào KHÔNG nên dùng hybrid?
#   → Query ngắn, latency critical, corpus đơn ngôn ngữ
# - Đổi sang `bge-m3` (multilingual) sẽ cải thiện semantic cho tiếng Việt
#
# **Bài học:**
# 1. Hybrid search là default production 2026 vì robust trên mọi query type
# 2. RRF formula đơn giản nhưng mạnh - không cần train
# 3. Embedding model choice matters rất nhiều cho non-English corpus

# %% [markdown]
# ## Deliverable evidence
# 1. Output cell 4: bảng Precision@10 với 3 modes
# 2. Output cell 5: bảng slice theo exact/paraphrase/mixed