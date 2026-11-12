# Chuẩn bị chạy lại thí nghiệm

Notebook trong kho giữ nguyên các ô code và output từ bản đã được chỉnh Markdown. Đây là hồ sơ thực nghiệm có hướng dẫn, chưa phải pipeline đã được xác minh bằng một lượt “Restart & Run All” trong môi trường mới.

## 1. Đọc đúng vị trí dữ liệu

KaggleHub trả về vị trí dataset qua biến `path`. Khi chạy ngoài Kaggle, thay dòng đọc đường dẫn cố định bằng cách đọc từ biến này:

```python
from pathlib import Path

df = pd.read_csv(Path(path) / "IMDB Dataset.csv")
df = df.iloc[:1000, :].copy()
```

Kiểm tra file thực sự tồn tại trước khi đọc. Tham khảo [KaggleHub](https://github.com/Kaggle/kagglehub) về thư mục cache và cách truy cập dataset.

## 2. Rà soát tiền xử lý

Từ điển `chat_word` hiện có mục `TIME`. Do phép thay thế không phân biệt hoa/thường, một từ thông dụng trong review có thể bị thay sai nghĩa. Cần loại bỏ hoặc thiết kế lại quy tắc nhập nhằng này, rồi kiểm tra một số câu trước khi áp dụng lên toàn bộ dữ liệu.

Các mô tả emoji liền nhau có thể bị nối từ. Notebook chưa có sửa lỗi chính tả hoặc thống kê review rỗng/trùng sau làm sạch. Mọi thay đổi ở phần này tạo ra pipeline khác với lần chạy đã lưu và cần được đánh giá lại.

## 3. Gọi hàm demo đúng tham số

Chữ ký hiện tại:

```python
def predict_sentiment(text, model, tokenizer, max_len=128):
    ...
```

Lời gọi phù hợp:

```python
label, probs = predict_sentiment(cleaned, model, tokenizer, max_len=128)
```

Không truyền `device` vào vị trí thứ tư: vị trí đó là chiều dài chuỗi. Hàm hiện lấy thiết bị từ biến `device` trong môi trường notebook.

## 4. Checkpoint và Google Drive

File `best_model.pt` chưa được cung cấp trong kho. Chạy huấn luyện sẽ tạo checkpoint khi validation accuracy cải thiện. Để suy luận mà không huấn luyện lại, cần có checkpoint tương thích do người dùng cung cấp.

Hai ô cuối sử dụng `google.colab.drive`; bỏ qua khi chạy trên máy cá nhân hoặc Kaggle và chọn thư mục lưu phù hợp. State dict cần được nạp vào đúng kiến trúc BERT hai lớp, với ánh xạ `negative=0`, `positive=1`. Nên lưu thêm tokenizer, cấu hình và phiên bản thư viện khi tạo bản mô hình để sử dụng lại.

## 5. Thực hiện và báo cáo một lần chạy mới

Sau khi xử lý các điểm trên, khởi động lại kernel và chạy các ô theo thứ tự. Ghi lại môi trường, thiết bị, seed, phiên bản dataset, phân hoạch và cấu hình huấn luyện. Không dùng tập test để chọn checkpoint.

Output hiện có: validation accuracy tốt nhất 84,00% ở epoch 4; test accuracy 86,00% trên 150 mẫu. Nếu chạy lại hoặc sửa code, cần thay các bảng trong notebook và README bằng kết quả của lần chạy mới, ghi rõ cấu hình đã thay đổi.

Để tăng độ tin cậy, kiểm tra review trùng trước khi chia, duy trì tỷ lệ nhãn, báo cáo precision/recall/F1 và confusion matrix, đồng thời so sánh với mô hình đối chứng. Các mục này chưa được thực hiện trong bản lịch sử.
