# Bai_TH_KTDL

Bài tập thực hành Khai thác dữ liệu.

Cài thư viện: `pip install -r requirements.txt`

Mỗi thư mục chứa code, dữ liệu cần dùng và biểu đồ đã xuất. Vào đúng thư mục bài rồi chạy
`python ten_file.py`. Các file `phieu_*`, `bai_*` là bài trong phiếu, `thuc_hanh_*` là bài thực hành.

| Thư mục | Nội dung | File chính |
|---|---|---|
| `bai_1` | Luật kết hợp | `bai_1` đến `bai_4` (Apriori, FP-Growth tự cài), `thuc_hanh_6_1` đến `6_3` |
| `bai_2` | Cây quyết định | `phieu_cau_1_7`, `phieu_cau_8`, `thuc_hanh_7_1` (ID3 tự cài), `7_2`, `7_3` |
| `bai_3` | Mạng nơ-ron, xác suất | `phieu_mlp`, `thuc_hanh_8_1`, `thuc_hanh_8_2` |
| `bai_4` | Gom cụm | `phieu_9_bai_1` đến `6`, `phieu_phan_cap_*`, `thuc_hanh_9_1`, `9_2` |
| `bai_5` | Mạng CNN | `cnn_mnist.py`, `cnn_nhan_dang_mat_nguoi.py` |

## Các giả định khi đề bài không nêu rõ

- `bai_1`, bài 3 và 4 của phiếu: với minsup = 80% chỉ có {a}, {c} phổ biến nên không sinh được luật nào. Đây là kết quả đúng của dữ liệu.
- `bai_2`, `thuc_hanh_7_3`: thư mục có dữ liệu `xtrainV3` nhưng không có đề, nên làm theo các bước của thực hành 7-2.
- `bai_3`: `income_test.csv` không có nhãn, nên train, validation, test đều chia từ `Income_train.csv`.
- `bai_4`, phiếu 9 bài 3: đề không cho K, chọn K = 2 và C1 = {x1, x2, x4}, C2 = {x3, x5}.
- `bai_4`, phiếu 9 bài 6: đề không cho ngưỡng, dùng ngưỡng 3 như bài 5.
- `bai_4`, `thuc_hanh_9_1_bai_tap`: dùng `load_digits` của scikit-learn thay cho MNIST đầy đủ.
- `bai_5`, nhận dạng mặt người: dùng bộ Olivetti Faces. Giữ đúng số lớp theo hình, thêm BatchNormalization và Dropout vì nếu không có chúng mạng không học được. Nên chạy trên GPU.
