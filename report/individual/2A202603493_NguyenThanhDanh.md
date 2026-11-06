# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603493
- **Họ và tên:** Nguyễn Thành Danh
- **SEED** (in ở ô đầu notebook): 8472

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | [x] |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | [x] |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | [x] |

## 2. Kết quả chính
                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.830      0.907        0.947   
2. BoW + MLP (deep learning)         1.000   0.911      0.953        1.000   
3a. Encoder (MiniLM) + kNN           0.893   0.973      0.932        1.000   
3b. Encoder (MiniLM) + LogReg        0.972   0.938      0.955        0.974   
3c. Encoder (MiniLM) + MLP head      0.964   0.955      0.960        0.974   
4a. LLM nhỏ zero-shot                0.000   0.000      0.000        0.000   
4b. LLM lớn zero-shot                0.000   0.000      0.000        0.000   
4c. LLM nhỏ few-shot                 0.000   0.000      0.000        0.000   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            3.555      20084.623       0.065   
2. BoW + MLP (deep learning)         6.579      47290.273       0.180   
3a. Encoder (MiniLM) + kNN         109.629         23.465      53.209   
3b. Encoder (MiniLM) + LogReg      110.197         28.720      31.754   
3c. Encoder (MiniLM) + MLP head    111.074         29.459      34.956   
4a. LLM nhỏ zero-shot                0.000          1.131     754.554   
4b. LLM lớn zero-shot                0.000          1.009     914.738   
4c. LLM nhỏ few-shot                 0.000          1.117     689.363   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.095         0.000  
2. BoW + MLP (deep learning)          0.272         0.000  
3a. Encoder (MiniLM) + kNN           64.120         0.000  
3b. Encoder (MiniLM) + LogReg        43.136         0.000  
3c. Encoder (MiniLM) + MLP head      59.244         0.000  
4a. LLM nhỏ zero-shot              1394.242         0.011  
4b. LLM lớn zero-shot              1246.037         0.110  
4c. LLM nhỏ few-shot               1397.595         0.024  
*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*

## 3. Dự đoán và thực tế
Thí nghiệm 1b, dự đoán là cong nhưng kết quả là của cả 3 trọng số, lệch là do chưa hiểu đề bài
Thí nghiệm 2a, dự đoán là 5 nhưng kết quả là 8, lý do lệch là do không hiểu về cách phân bố trọng số, dẫn đến phán đoán sai
Thí nghiệm ở 3b, em đoán là Encoder + LogReg nhưng kết quả là cả 2 đều bằng nhau, lệch do bị thiếu kiến thức
*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

## 4. Phân tích 50 case khó
Mô hình 2 BoW + MLP (Deep Learning) là mô hình bị giảm F1 thảm hại nhất, từ 1.000 (trên tập SUB thường) xuống còn chỉ 0.426 (trên 50 case khó)
Nhóm case gây khó khăn lớn nhất cho BoW (LogReg & MLP):
*Né từ khóa*: Đạt tỉ lệ chính xác cực thấp (gần như 0% đối với các từ bị viết cố ý chèn dấu chấm hoặc ký tự lạ).
*Spam giọng tự nhiên*: Do không chứa các từ khóa kích hoạt mạnh (như "free", "win", "call"), mô hình dựa trên Bag-of-Words hoàn toàn bị qua mặt.
**2. Hai case cụ thể và cơ chế gây sai lệch**
**Case 1: Né từ khóa (Obfuscation)**
Nội dung SMS: "F R E E e n t r y into our weekly draw, t x t WIN to 80086"
Cơ chế lỗi: Phương pháp Bag-of-Words dựa vào việc tách từ bằng khoảng trắng mặc định. Khi kẻ tấn công cố tình viết tách rời từng chữ cái (F R E E, t x t), tokenizer của BoW sẽ nhận diện đây là các từ đơn lẻ có độ dài 1 ký tự (f, r, e, e). Các từ đơn lẻ này không nằm trong từ điển đặc trưng phổ biến (vocab) để đại diện cho Spam, làm mất hoàn toàn từ gốc mang tính phân loại cao như free hay txt.
**Case 2: Spam giọng tự nhiên / Đóng vai (Natural Spam / Impersonation)**
Nội dung SMS: "Hi Mum, I dropped my phone in the toilet, this is my new number. Can you send £200 to my friend's account? I'll explain later x"
Cơ chế lỗi: Mô hình BoW chỉ quan sát sự xuất hiện đơn lẻ của các từ trong từ điển mà bỏ qua hoàn toàn ngữ cảnh, trật tự từ và ý đồ ngữ nghĩa. SMS này được viết dưới giọng điệu cực kỳ tự nhiên, mô phỏng cuộc trò chuyện thân mật của người thân gia đình (Mum, phone, friend, explain). Không có bất kỳ từ khóa quảng cáo hay trúng thưởng truyền thống nào. Do đó, các trọng số kéo về phía ham (như my, I, to) lấn át hoàn toàn, khiến mô hình MLP/LogReg tin tưởng đây là tin nhắn bình thường và bỏ lọt cú lừa chuyển tiền này.
*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân
Gọi hàm predict_proba mục đích là tính xác suất dự đoán trên tập test
Tiếp đó khai báo ngưỡng dự đoán là 0.2; nó nghi ngờ nếu quá 20% sẽ lập tức đánh spam, và ngưỡng 0.8 được set cao lên, và nếu vượt qua nó mới bị đánh spam. Các dòng metrics và print đi kèm chỉ đơn giản là dùng để đo lường và in ra các chỉ số hiệu năng tương ứng
*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật
Em sẽ đề xuất thiết kế hệ thống với khả năng lọc với 2 tầng. Tầng một dùng BoW kết hợp MLP để quét và chốt kết quả ngay lập tức cho 90 phần trăm lượng tin nhắn có nội dung rõ ràng là rác hoặc cực kỳ an toàn. Tầng hai chỉ kích hoạt mô hình ngôn ngữ lớn để phân tích 10 phần trăm số tin nhắn lọt vào vùng xác suất lưỡng lự từ 0.3 đến 0.7. Cách làm này giúp chặn đứng các thủ đoạn lừa đảo tinh vi mà không tốn kém tài nguyên hay làm chậm toàn bộ hệ thống

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học
| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |

| 2.2 | Hỗ trợ viết code |Trả lời code | Chỉnh lại công thức |

| 4 | Giúp chỉnh sửa cấu hình để kết nối API | Hướng dẫn cách setup từng bước cũng như lỗi xảy ra | Tạo API key, điều chỉnh tên của API key|

| 4 | Hỗ trợ viết phần My test case | Trả ra 5 câu cho phần My test case | Điều chỉnh lại về câu từ |
*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
