# Reflection — Lab 19

**Tên:** Lương Bảo Long
**Cohort:** A20-K2
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

Trên golden set 50 queries của lab:

- **Exact queries:** BM25 thắng (96.7%) vì có từ kỹ thuật verbatim khớp với corpus
- **Paraphrase queries:** Semantic vector thắng về mặt concept, nhưng BGE-small-en trên tiếng Việt yếu (chỉ ~24-32%). BM25 cũng kém vì không có keyword match
- **Mixed queries:** Hybrid RRF thắng rõ rệt (100% vs 97-98%) - kết hợp cả keyword lẫn semantic

**Khi nào KHÔNG dùng hybrid:**
1. Query rất ngắn hoặc đơn nghĩa - BM25 đủ, hybrid thêm overhead
2. Corpus chủ yếu tiếng Anh đơn thuần - vector model English-trained là đủ
3. Latency critical với tail budget cực thấp - hybrid P99 cao hơn (đo được: hybrid 99.6ms vs keyword 1.8ms)
4. Khi hybrid không cải thiện đáng kể so với single mode

---

## Điều ngạc nhiên nhất khi làm lab này

Semantic cache có thể trả lời sai cho tenant khác nếu quên namespace - đây là lỗ hổng bảo mật OWASP LLM08 trong 15 dòng code, không có exception hay log đỏ.

---

## Bonus challenge

- [ ] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _<tên đồng đội nếu có>_
