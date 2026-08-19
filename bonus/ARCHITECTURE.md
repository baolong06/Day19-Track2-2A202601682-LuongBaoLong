# Bonus — Hybrid Memory Agent: Architecture

**Author:** Lương Bảo Long (A20-K2)
**Lab module:** Day 19 — Vector Store + Feature Store
**Source-of-truth:** `bonus/agent.py`, `bonus/demo.py`

---

## 1. Sơ đồ kiến trúc

```
                  ┌─────────────────────────────────────────────────────┐
                  │              HybridMemoryAgent                       │
                  │                                                       │
   remember():    │  ┌──────────────┐    text  ┌──────────────┐         │
   ──────────     │  │  chunker     │───────► │  embedder    │         │
                  │  │  (per-message)│          │  (bge-small) │         │
                  │  └──────────────┘          └──────┬───────┘         │
                  │                                    │ vec            │
                  │                                    ▼                │
                  │  ┌──────────────────────────────────────────────┐    │
                  │  │   Qdrant collection `bonus_memory`            │    │
                  │  │   payload: {user_id, text, ts}                │    │
                  │  │   payload index: user_id (KEYWORD)           │    │
                  │  └──────────────────────────────────────────────┘    │
                  │                                                       │
   recall():      │  query  ┌──────────────┐  qv  ┌────────────────┐    │
   ──────────     │  ────► │  embedder    │ ───► │  Qdrant filter │    │
                  │         └──────────────┘       │  user_id=...    │    │
                  │                                │  top_k=3        │    │
                  │                                └────────┬───────┘    │
                  │                                         │ hits       │
                  │  ┌─────────────────────┐  Feast online                 │
                  │  │  user_profile_features │  ───────────────┐          │
                  │  │  topic_affinity       │                 │          │
                  │  └─────────────────────┘                 ▼          │
                  │  ┌─────────────────────┐   ┌──────────────────────┐  │
                  │  │  query_velocity ────────► │  Context assembler │  │
                  │  │  streaming feature view │ │  → LLM prompt       │  │
                  │  └─────────────────────┘   └──────────────────────┘  │
                  └─────────────────────────────────────────────────────────┘
```

**Data flow:**
1. `remember(text)` → embed → upsert với `user_id` payload.
2. `recall(query)` → embed query → filter `user_id` → top-K hits.
3. Parallel: render profile + recent activity từ Feast online store.
4. Assembler ghép 3 thành 1 prompt block.

---

## 2. Ba quyết định kiến trúc

### 2.1 Chunking — **per-message**

**Lựa chọn:** mỗi dòng text = 1 chunk, không gộp.

**Trade-off:**
- *Per-message:* retrieval precision cao (1 fact = 1 vector), nhưng tốn tokens khi lắp nhiều hits.
- *Per-conversation:* recall cao hơn, nhưng 1 chi tiết bị chôn trong chunk lớn.
- *Semantic break:* cần LLM để tách → tăng cost + latency write.

**Tại sao chọn per-message:** POC scope. User messages của assistant cá nhân thường 1-3 câu. **Số token TB: 30-50/message vs 200-500/conversation.** Trợ lý cá nhân ưu tiên precision: trả lời "tôi đã đọc về X" chính xác hơn "tôi đã nói về X, Y, Z" mơ hồ.

**Liên kết lab:** NB2 hybrid search cũng chunk nhỏ. RRF k=60 ổn định cho 50-100 docs.

### 2.2 Feature schema — **tabular + freshness khác nhau**

**Lựa chọn:**
- `user_profile_features` (tabular, Feast `FeatureView`): `topic_affinity`, `preferred_language`, `reading_speed_wpm`, `active_hours_local`. TTL = 7 days.
- `query_velocity` (streaming `FeatureView` + `OnDemandFeatureView`): `queries_last_hour`, `last_topic`. TTL = 0.

**Trade-off:**
- *Tabular:* dễ debug, dễ ship. Mất latent preferences.
- *Embedding features:* capture "user thích docs có cụm từ X" mà tabular không thấy. Tốn 384-dim × N storage; cần clustering.

**Tại sao chọn tabular cho profile:** POC scope. Latent prefs có thể derive từ vector store (user's hits → cluster). Sau này nếu cần: thêm ODFV `topic_affinity_embedding` cho LLM tự phân tích.

**Liên kết lab:** NB4 Feast 3 views, NB8 ODFV.

### 2.3 Freshness — **3 use cases, 3 cadences**

| Use case | Freshness | Mechanism |
|----------|-----------|-----------|
| "Tôi vừa đọc xong doc X" | sub-second | Streaming Push API |
| "Tôi đang quan tâm gì?" | 5 min | Streaming window count → materialized mỗi 5 min |
| "Tôi thích gì lâu dài?" | 1 ngày | Batch → daily `materialize-incremental` |

**Trade-off:** freshness cao = cost cao. Streaming Push API giữ state trong memory; nếu 1h mới cần 1 lần, batch refresh rẻ hơn 100×.

**Liên kết lab:** NB8 ODFV (real-time computed), NB7 TTL (cache hit ở threshold hợp lý).

---

## 3. Một lựa chọn bị loại, có lý do

**Tôi cân nhắc:** lưu episodic memory như 1 embedding feature view trong Feast.

**Lý do loại:**
- Re-index cycle episodic (5-20 memories/hour) khác hẳn profile (weekly batch).
- Feast optimized cho tabular, không cho vector store.
- Khi re-index episodic, phải update `event_timestamp` cho mọi row → Feast PIT join semantics phá vỡ.

**Chọn:** tách riêng. Episodic trong Qdrant, profile trong Feast. Kết nối qua `user_id` payload index. **Same key, different store.**

---

## 4. Vietnamese-context considerations

**Code-switching (vi/en mix):** User VN hay viết "tôi vừa đọc về Kubernetes cluster autoscaling". Decision: keep as-is, không dịch. bge-small-en score cao với cluster docs EN. Nếu cần: thêm pre-processing dịch technical terms → EN trước embed.

**Phonetic typos:** "triển khai" → "trển khai" (gõ sai dấu). BGE-small-en xử lý kém. Solution: fuzzy matching cached cho queries hay sai.

**Tokenizer:** Whitespace split cho BM25 (lab default), bge-small-en cho vector (no tokenizer). Alternative: pyvi/underthesea Vietnamese-aware — cải thiện 5-10% Precision@10 trên corpus toàn tiếng Việt, nhưng thêm 200MB dep + slower indexing. POC bỏ qua.

**Privacy (Decree 13/2023/NĐ-CP):** Quan trọng cho user VN. Cần:
- Encryption at rest (Qdrant + Feast SQLite).
- User-controlled delete (GDPR-style right to be forgotten).
- Audit log mọi retrieval.

POC chưa implement — note trong limitations.

---

## 5. Honest limitations

POC này **KHÔNG** xử lý:

1. **Multi-user privacy isolation thực sự.** Per-user collection? Per-user encryption? Right now chỉ filter `user_id` payload — nhưng NB7 đã chứng minh 1 dòng filter missing = data breach. Cần audit mỗi recall.
2. **Encryption at rest.** POC dùng Qdrant in-memory + SQLite Feast — không mã hoá. Production cần Qdrant server với TLS + SQLCipher hoặc Postgres + TDE.
3. **CRUD trên memory.** Chỉ `remember()` + `recall()`. Thiếu `forget(mem_id)`, `update(mem_id, text)`. Cần cho GDPR Article 17.
4. **Multi-device sync.** Single-process. Production cần distributed consensus.
5. **Memory decay.** Episodic không có TTL. User đọc 5 năm trước về K8s 1.20 → vẫn surface khi query 2026. Cần cohort pruning ("untouched 90d → archive").
6. **Memory consolidation.** 5 memories tương tự gộp thành 1 summary — LLM-driven, tốn cost. Trigger weekly.
7. **Latent user preferences.** Profile chỉ 4 features tabular. Có thể trích từ episodic clustering.

---

## 6. Vibe coding log

**Prompt hiệu quả nhất:** "Viết `HybridMemoryAgent` class với 2 method `remember()` + `recall()`. Pattern NB2 Searcher + NB4 Feast online lookup. 120-150 dòng, đừng over-engineer." → Spec narrow → clean diff, không iterate.

**Prompt fail:** "Làm bonus AI memory × profile × feature store comprehensive POC" → 300 dòng, thiếu focus. Narrow scope > big vision.

**Lesson:** Spec-Driven Development (VIBE-CODING.md §2) hoạt động. Spec rõ → code rõ.

---

*Tài liệu này viết theo tinh thần "1 quyết định tốt + 1 POC chạy được > 5 ý tưởng dở dang" (bonus brief §Topics).*
