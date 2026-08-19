# ---
# title: NB1 — Embeddings & Vector Indexing
# description: |
#   **Stack:** `fastembed` (ONNX, CPU) + Qdrant in-memory.
#   Maps to slide §1 (Embeddings) + §2 (Vector DB Landscape).
# ---

# ## Mục tiêu bài này
#
# Hiểu cách text được chuyển thành vector (embeddings) và cách vector DB
# index/query vectors đó. Không cần GPU, không cần Docker.
#
# **Think about:** Tại sao text → vector? Vì máy tính không hiểu text trực tiếp,
# nhưng nó tính được khoảng cách giữa các vector.

# ## Setup
import _setup  # noqa: F401
import json
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

DATA = Path(_setup.__file__).resolve().parent.parent / "data"

# ## 1. Load corpus
# Corpus đã được sinh sẵn: 1000 docs tiếng Việt, 10 chủ đề × 100 docs/chủ đề.
#
# **WHY:** Cần corpus để test embedding. Nếu không có corpus, không có gì để embed.
docs = []
with (DATA / "corpus_vn.jsonl").open(encoding="utf-8") as f:
    for line in f:
        docs.append(json.loads(line))

print(f"Corpus size: {len(docs)} docs")
print(f"First doc:")
print(json.dumps(docs[0], ensure_ascii=False, indent=2))

# ## 2. Embedding model
#
# Model: `BAAI/bge-small-en-v1.5` (384-dim vectors)
#
# **WHY dùng model này?**
# - fastembed chạy ONNX → CPU friendly, không cần GPU
# - 384 dimensions là "vừa đủ": đủ thông tin semantic, không quá nặng
# - bge-small-en = baseline model, lab này không cần model mạnh
#
# **Think about:** Nếu là tiếng Việt thuần, model nào tốt hơn?
# (Hint: xem deck §1 bảng *Embedding Models 2026*)

embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
sample = list(embedder.embed(["cloud computing tiếng Việt"]))[0]
print(f"Vector dim: {len(sample)}")
print(f"First 8 values: {sample[:8].tolist()}")

# **HOW: Embedding hoạt động thế nào?**
# 1. Text đi vào transformer model (BERT-based)
# 2. Model output 384 số thực cho mỗi token/đoạn
# 3. Các số này represent "vị trí" của text trong không gian 384 chiều
# 4. Text có nghĩa tương tự → vector gần nhau trong không gian này

# ## 3. Tạo Qdrant collection
#
# **WHY Qdrant?**
# - Qdrant là vector DB, chuyên truy vấn "tìm vector gần nhất"
# - In-memory mode: chạy trong process, không cần server
# - Production: chỉ cần đổi `:memory:` → `url="http://..."`
#
# **HOW cosine similarity?**
# - Distance.COSINE = đo góc giữa 2 vector
# - 0° = identical = score 1.0
# - 90° = unrelated = score 0.0
# - 180° = opposite = score -1.0

client = QdrantClient(":memory:")
client.create_collection(
    collection_name="lab19",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
print("Collection 'lab19' created with 384-dim cosine similarity")

# ## 4. Embed + upsert toàn bộ corpus
#
# **HOW batch processing?**
# - fastembed CPU-bound, batch=64 là sweet spot (thử nghiệm)
# - Batch lớn → nhanh hơn nhưng tốn RAM
# - Batch nhỏ → chậm hơn nhưng ít RAM
#
# **WHY batch?**
# - Embedding model xử lý nhiều texts cùng lúc hiệu quả hơn
# - GPU inference thường batch được, nhưng CPU cũng受益

BATCH = 64
points: list[PointStruct] = []

for start in range(0, len(docs), BATCH):
    batch = docs[start:start + BATCH]
    texts = [d["title"] + " " + d["text"] for d in batch]
    vectors = list(embedder.embed(texts))
    for i, (d, v) in enumerate(zip(batch, vectors)):
        points.append(PointStruct(
            id=start + i,
            vector=v.tolist(),
            payload={"doc_id": d["doc_id"], "topic": d["topic"], "title": d["title"]},
        ))

client.upsert(collection_name="lab19", points=points)
n_indexed = client.count(collection_name="lab19").count
print(f"Indexed: {n_indexed} vectors")
assert n_indexed == 1000, f"expected 1000 indexed, got {n_indexed}"

# ## 5. Similarity search
#
# **HOW similarity search hoạt động?**
# 1. Query text → embed → vector 384 chiều
# 2. Qdrant tính cosine similarity giữa query vector và tất cả 1000 vectors trong DB
# 3. Sort theo score descending
# 4. Return top-k results

query = "cloud computing và tự động mở rộng"
q_vec = next(embedder.embed([query])).tolist()
hits = client.query_points(collection_name="lab19", query=q_vec, limit=5).points

print(f"Query: {query!r}")
print(f"Top-5:")
for i, h in enumerate(hits, 1):
    print(f"  {i}. [{h.payload['topic']:>9}] score={h.score:.3f}  {h.payload['title']}")

# **WHY top-5 toàn là topic "cloud"?**
# Vì query có từ "cloud" và "mở rộng" → vector gần với docs cloud nhất
# Đây là semantic search, không phải keyword search!

# ## 6. Paraphrase query
#
# **WHY query này KHÔNG có từ "cloud" nhưng vẫn tìm được cloud docs?**
# Vì embedding model hiểu "tự động mở rộng hạ tầng" ≈ "auto scaling infrastructure"
# ≈ "cloud" concepts. Đây là sức mạnh của semantic search.

query2 = "phương pháp tự động mở rộng hạ tầng theo lưu lượng người dùng"
q_vec2 = next(embedder.embed([query2])).tolist()
hits2 = client.query_points(collection_name="lab19", query=q_vec2, limit=5).points

print(f"\nQuery (paraphrase, no 'cloud' keyword): {query2!r}")
for h in hits2:
    print(f"  [{h.payload['topic']:>9}] score={h.score:.3f}  {h.payload['title']}")

# **WHY score thấp hơn query gốc?**
# Query gốc có từ "cloud" → direct match → score ~0.80
# Query paraphrase không có "cloud" → phải dựa vào semantic similarity → score thấp hơn
# Nhưng vẫn đủ cao để rank #1 vì nội dung đúng topic

# ## Bài học rút ra
#
# 1. **Embedding biến text thành vector** - máy tính tính được khoảng cách
# 2. **Vector DB index vectors** để query nhanh, không duyệt O(n)
# 3. **Cosine similarity** đo độ tương tự qua góc vector
# 4. **Semantic search > keyword search** vì hiểu ý nghĩa, không chỉ từ khóa
# 5. **Batch processing** quan trọng để tối ưu performance
#
# **Think further:** Nếu corpus 10M docs thì sao? Cần cải thiện gì?
