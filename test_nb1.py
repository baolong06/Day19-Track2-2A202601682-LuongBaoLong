# Test NB1 - Run directly with venv Python
import _setup
import json
from pathlib import Path
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

DATA = Path(_setup.__file__).resolve().parent.parent / "data"

# 1. Load corpus
docs = []
with (DATA / "corpus_vn.jsonl").open(encoding="utf-8") as f:
    for line in f:
        docs.append(json.loads(line))

print(f"Corpus size: {len(docs)} docs")
print(f"First doc:")
print(json.dumps(docs[0], ensure_ascii=False, indent=2))

# 2. Embedding model
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
sample = list(embedder.embed(["cloud computing tiếng Việt"]))[0]
print(f"\nVector dim: {len(sample)}")
print(f"First 8 values: {sample[:8].tolist()}")

# 3. Index to Qdrant
client = QdrantClient(":memory:")
client.create_collection(
    collection_name="lab19",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 4. Embed + upsert corpus
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
print(f"\nIndexed: {n_indexed} vectors")
assert n_indexed == 1000, f"expected 1000 indexed, got {n_indexed}"
print("NB1 Part 1: PASS - 1000 vectors indexed")

# 5. First similarity search
query = "cloud computing và tự động mở rộng"
q_vec = next(embedder.embed([query])).tolist()
hits = client.query_points(collection_name="lab19", query=q_vec, limit=5).points

print(f"\nQuery: {query!r}")
print(f"Top-5:")
for i, h in enumerate(hits, 1):
    print(f"  {i}. [{h.payload['topic']:>9}] score={h.score:.3f}  {h.payload['title']}")

# 6. Paraphrase query
query2 = "phương pháp tự động mở rộng hạ tầng theo lưu lượng người dùng"
q_vec2 = next(embedder.embed([query2])).tolist()
hits2 = client.query_points(collection_name="lab19", query=q_vec2, limit=5).points

print(f"\nQuery (paraphrase): {query2!r}")
cloud_count = 0
for h in hits2:
    print(f"  [{h.payload['topic']:>9}] score={h.score:.3f}  {h.payload['title']}")
    if h.payload['topic'] == 'cloud':
        cloud_count += 1

print(f"\nNB1 Part 2: {cloud_count}/5 top results are 'cloud' topic")
if cloud_count >= 4:
    print("NB1 Part 2: PASS - Paraphrase query returns correct cluster")
else:
    print("NB1 Part 2: WARN - Low cloud topic match (may vary with embedding model)")

print("\n=== NB1 COMPLETE ===")
