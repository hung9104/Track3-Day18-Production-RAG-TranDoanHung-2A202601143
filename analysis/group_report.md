# Group Report - Lab 18

**Nhóm:** K34-Day18-Production-RAG  
**Ngày:** 2026-08-18

## Thành viên & Module

| Tên | Module | Hoàn thành | Tests pass |
|---|---|---:|---:|
| - | M1: Chunking | Hoàn thành | 13/13 |
| - | M2: Search | Hoàn thành | 5/5 |
| - | M3: Rerank | Hoàn thành | 5/5 |
| - | M4: Eval | Hoàn thành | 4/4 |
| - | M5: Enrichment | Hoàn thành | 10/10 |

## Kết quả

| Metric | Naive | Production | Delta |
|---|---:|---:|---:|
| Faithfulness | Chưa có số đo RAGAS ổn định | Chưa có số đo RAGAS ổn định | - |
| Answer Relevancy | Chưa có số đo RAGAS ổn định | Chưa có số đo RAGAS ổn định | - |
| Context Precision | Chưa có số đo RAGAS ổn định | Chưa có số đo RAGAS ổn định | - |
| Context Recall | Chưa có số đo RAGAS ổn định | Chưa có số đo RAGAS ổn định | - |

## Key Findings

1. **Biggest improvement:** pipeline production đã hoàn thiện đầy đủ 5 module và toàn bộ auto-test nội bộ đều pass 37/37.
2. **Biggest challenge:** môi trường đánh giá RAGAS cần API key, model cache phù hợp và runtime đủ dài; đây là phần dễ phát sinh timeout nhất.
3. **Surprise finding:** reranker BAAI có thể trả score quá bão hòa nếu dùng trực tiếp, nên cần chuẩn hóa score và thêm tie-break lexical.

## Presentation Notes

1. RAGAS scores (naive vs production): chưa dùng số đo cuối cùng vì các lần chạy đầy đủ trước đó bị timeout hoặc không ổn định do môi trường.
2. Biggest win - module nào, tại sao: M1 đến M5 đều có test riêng và pipeline end-to-end đã chạy được với dữ liệu nội bộ.
3. Case study - 1 failure, Error Tree: case khó nhất là truy xuất văn bản scan/PDF không có text layer, cần OCR hoặc loại khỏi nguồn text retrieval.
4. Next optimization nếu có thêm 1 giờ: bổ sung OCR cho PDF scan và chuẩn hóa một bộ đánh giá RAGAS chạy ổn định hơn.
