# Failure Analysis - Lab 18: Production RAG

**Nhóm:** K34-Day18-Production-RAG  
**Thành viên:** - · - · - · -

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Delta |
|---|---:|---:|---:|
| Faithfulness | Chưa chốt số đo ổn định | Chưa chốt số đo ổn định | - |
| Answer Relevancy | Chưa chốt số đo ổn định | Chưa chốt số đo ổn định | - |
| Context Precision | Chưa chốt số đo ổn định | Chưa chốt số đo ổn định | - |
| Context Recall | Chưa chốt số đo ổn định | Chưa chốt số đo ổn định | - |

## Bottom-5 Failures

### #1 - BAAI model cache không tương thích
- **Question:** Các model `BAAI/bge-m3` và `BAAI/bge-reranker-v2-m3` không load ổn định trong môi trường hiện tại.
- **Expected:** Search/rerank dùng embedding và cross-encoder thật.
- **Got:** Load lỗi hoặc hành vi không ổn định khi chạy offline.
- **Worst metric:** Context Precision / Answer Relevancy.
- **Error Tree:** Output sai -> Context không đáng tin -> Query vẫn đúng -> Root cause: cache/model runtime không tương thích.
- **Suggested fix:** Cho phép fallback model nhẹ hơn và chuẩn hóa dimension/runtime cấu hình qua biến môi trường.

### #2 - Naive baseline timeout
- **Question:** Chạy `naive_baseline.py` để tạo baseline report đầy đủ.
- **Expected:** Có report baseline với số lượng câu hỏi hợp lệ.
- **Got:** Report có thể ra `num_questions = 0` khi eval/API timeout.
- **Worst metric:** Tất cả metrics vì không có dữ liệu thật để so sánh.
- **Error Tree:** Output sai -> Không có kết quả eval -> Query/API timeout -> Root cause: thời gian chạy và API phụ thuộc.
- **Suggested fix:** Tăng timeout hợp lý, cache dữ liệu, hoặc chạy baseline trên một tập mẫu nhỏ hơn.

### #3 - RAGAS evaluation timeout
- **Question:** Chạy đánh giá end-to-end cho production pipeline.
- **Expected:** Có `reports/ragas_report.json` ổn định.
- **Got:** Nhiều job RAGAS chạy rất lâu và dễ chạm timeout.
- **Worst metric:** Faithfulness / Answer Relevancy / Context Recall.
- **Error Tree:** Output sai -> Eval không xong -> Model/API chậm -> Root cause: benchmark quá nặng cho session ngắn.
- **Suggested fix:** Chia batch nhỏ hơn, giảm số câu hỏi, và tăng timeout cho bước eval.

### #4 - Reranker score saturation
- **Question:** Rerank tài liệu đúng theo mức độ liên quan.
- **Expected:** Tài liệu liên quan nhất đứng đầu.
- **Got:** Score sigmoid gần như bão hòa, tài liệu có thể bị xếp sai nếu so trực tiếp.
- **Worst metric:** Context Precision.
- **Error Tree:** Output sai -> Context chưa tối ưu -> Query đúng -> Root cause: score model không được chuẩn hóa.
- **Suggested fix:** Chuẩn hóa score theo batch và thêm tie-break lexical để tránh đảo thứ hạng vô lý.

### #5 - PDF scan không có text layer
- **Question:** Truy hồi nội dung từ `BCTC.pdf` và các PDF scan khác.
- **Expected:** Nội dung được chunk và search như markdown text.
- **Got:** Một số PDF scan không có text layer nên bị bỏ qua.
- **Worst metric:** Context Recall.
- **Error Tree:** Output sai -> Context thiếu -> Query đúng -> Root cause: nguồn dữ liệu scan cần OCR.
- **Suggested fix:** Thêm OCR pipeline hoặc đánh dấu riêng nhóm tài liệu scan để xử lý khác.

## Case Study (presentation)

**Question chọn phân tích:** truy xuất văn bản từ PDF scan không có text layer.

**Error Tree walkthrough:**
1. Output đúng? -> Không.
2. Context đúng? -> Không đầy đủ vì PDF scan bị skip.
3. Query rewrite OK? -> Có, query vẫn đúng.
4. Fix ở bước: ingestion/OCR, không phải ở search.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Bổ sung OCR cho PDF scan.
- Thêm report ngắn cho các tài liệu bị bỏ qua khi ingest.
- Chạy lại một batch RAGAS nhỏ để có số đo production ổn định hơn.
