# Reflection — Lab 19

**Tên:** Lương Bảo Long
**Cohort:** A20-K2
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

Trên 50 golden queries:

- **NB2:** Hybrid RRF Precision@10 = 78.6%, > BM25 (77.8%) > semantic (73.2%). Hybrid thắng rõ ở `mixed` slice (100% vs 97-98%). Paraphrase yếu vì `bge-small-en` là model English.
- **NB3:** Hybrid P99 = 12.8ms (pass <50ms sau cache + warm-up).
- **NB8:** target-naive trên `session_id`: train_auc 0.999 vs test_auc 0.522 — gap 47.7%.

**Khi nào KHÔNG dùng hybrid:**
1. Query ngắn single-intent — BM25 đủ
2. Latency-critical — keyword P99=6.7ms vs hybrid 12.8ms

**Bài học lớn:** Lúc đầu tôi báo PASS dựa trên script riêng, notebook thật hiển thị FAIL 75ms. Fix bằng headless execute + output cell làm ground truth — không tin "số đẹp" từ script ngoài.

## Điều ngạc nhiên nhất

Semantic cache namespace leak (NB7) — không exception, không log, chỉ sai answer. Một dòng `query_filter` ngăn data breach.

---

## Bonus challenge
- [x] Không làm bonus (thiếu thời gian)
- [ ] Pair work với: _<tên đồng đội nếu có>_
