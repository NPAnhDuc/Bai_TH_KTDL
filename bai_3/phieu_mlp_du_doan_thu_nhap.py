# -*- coding: utf-8 -*-
"""
PHIẾU BÀI TẬP MLP: Áp dụng mạng MLP để dự đoán thu nhập (file Income_train.csv).

Chương trình làm lần lượt 10 yêu cầu của đề, mỗi phần có ghi số thứ tự yêu cầu.

File income_test.csv của đề không có cột target_income nên không dùng để đánh giá
được. Vì vậy tập train, validation, test đều được chia từ Income_train.csv.

Cài thư viện: pip install pandas numpy matplotlib scikit-learn
Chạy: python phieu_mlp_du_doan_thu_nhap.py   (mất khoảng vài phút)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

THU_MUC = Path(__file__).parent

# Hiển thị đủ các cột khi in bảng
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

# ----- 1. Đọc dữ liệu -----
# Trong file, giá trị bị thiếu được ghi bằng dấu ?. na_values chuyển chúng thành NaN
data = pd.read_csv(THU_MUC / "Income_train.csv", na_values="?")
print("1. Kích thước dữ liệu:", data.shape)
print(data.head())

# ----- 2. Chuyển target_income về dạng 0, 1 -----
# Cột này trong file đã là 0 và 1, ép kiểu để chắc chắn dữ liệu là số nguyên
data["target_income"] = data["target_income"].astype(int)
print("\n2. Số mẫu mỗi nhãn của target_income:")
print(data["target_income"].value_counts())

# ----- 3. Mô tả dữ liệu -----
# describe() cho biết số mẫu, trung bình, độ lệch, min, max và các phân vị 25%, 50%, 75%
cac_cot_so = ["age", "final_weight", "total_education_yrs",
              "capital_gain", "capital_loss", "hrs_per_week"]
print("\n3. Mô tả các thuộc tính số:")
print(data[cac_cot_so].describe().round(2))

# ----- 4. Xử lý dữ liệu bị thiếu -----
print("\n4. Số giá trị bị thiếu trước khi xử lý:")
print(data.isnull().sum()[data.isnull().sum() > 0])

# Các cột bị thiếu đều là cột chữ, nên điền bằng giá trị xuất hiện nhiều nhất
for cot in data.columns[data.isnull().any()]:
    data[cot] = data[cot].fillna(data[cot].mode()[0])
print("Số giá trị bị thiếu sau khi xử lý:", data.isnull().sum().sum())

# ----- 5. Ma trận tương quan giữa thuộc tính điều kiện và thuộc tính mục tiêu -----
cac_cot_tuong_quan = cac_cot_so + ["target_income"]
tuong_quan = data[cac_cot_tuong_quan].corr()
print("\n5. Ma trận tương quan:")
print(tuong_quan.round(2))

plt.figure(figsize=(8, 7))
plt.imshow(tuong_quan, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(label="Hệ số tương quan")
plt.xticks(range(len(cac_cot_tuong_quan)), cac_cot_tuong_quan, rotation=45, ha="right")
plt.yticks(range(len(cac_cot_tuong_quan)), cac_cot_tuong_quan)

# Ghi giá trị vào từng ô
for i in range(len(cac_cot_tuong_quan)):
    for j in range(len(cac_cot_tuong_quan)):
        gia_tri = tuong_quan.iloc[i, j]
        mau_chu = "white" if abs(gia_tri) > 0.6 else "black"
        plt.text(j, i, f"{gia_tri:.2f}", ha="center", va="center", color=mau_chu)

plt.title("Ma trận tương quan")
plt.tight_layout()
plt.savefig(THU_MUC / "ma_tran_tuong_quan.png", dpi=150)
plt.close()
print("Đã lưu biểu đồ: ma_tran_tuong_quan.png")

# ----- 6. Ghi dữ liệu vừa được xử lý -----
data.to_csv(THU_MUC / "Income_train_da_xu_ly.csv", index=False)
print("\n6. Đã ghi file: Income_train_da_xu_ly.csv")

# ----- 7. Chia dữ liệu thành tập train, validation và test -----
# Cột ID chỉ là số thứ tự nên bỏ đi. Các cột chữ được mã hóa one-hot thành cột 0, 1
X = pd.get_dummies(data.drop(columns=["ID", "target_income"]), dtype=int)
y = data["target_income"]

# Chia 70% train, phần còn lại chia đôi cho validation và test (mỗi tập 15%)
X_train, X_tam, y_train, y_tam = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tam, y_tam, test_size=0.5, random_state=1, stratify=y_tam
)
print("\n7. Số mẫu train, validation, test:", len(X_train), len(X_val), len(X_test))

# Chuẩn hóa dữ liệu giúp mạng nơ-ron học tốt hơn.
# Chỉ fit trên tập train để không làm lộ thông tin của validation và test
chuan_hoa = StandardScaler()
X_train = chuan_hoa.fit_transform(X_train)
X_val = chuan_hoa.transform(X_val)
X_test = chuan_hoa.transform(X_test)


def huan_luyen(cac_lop_an):
    """Tạo và huấn luyện một mạng MLP với các lớp ẩn được truyền vào."""
    # Với 2 nhãn, MLPClassifier tự tạo lớp ra 1 nơ-ron dùng hàm sigmoid
    mo_hinh = MLPClassifier(
        hidden_layer_sizes=cac_lop_an,
        activation="relu",
        solver="adam",
        batch_size=128,
        max_iter=20,
        random_state=1,
    )
    mo_hinh.fit(X_train, y_train)

    acc_val = accuracy_score(y_val, mo_hinh.predict(X_val))
    print(f"   Lớp ẩn {cac_lop_an}: accuracy validation = {acc_val:.4f}")
    return mo_hinh


# ----- 8. Mạng MLP theo kiến trúc 64 - 32 - 16 - 1 -----
print("\n8. Huấn luyện kiến trúc 64 - 32 - 16 - 1")
mo_hinh_1 = huan_luyen((64, 32, 16))
print("   Số nơ-ron lớp ra:", mo_hinh_1.n_outputs_, ", hàm kích hoạt:", mo_hinh_1.out_activation_)

# ----- 9. Tạo một kiến trúc khác -----
print("\n9. Huấn luyện kiến trúc 128 - 64 - 1")
mo_hinh_2 = huan_luyen((128, 64))

# ----- 10. Đánh giá các mô hình trên tập kiểm tra, so sánh Accuracy và F1-score -----
ket_qua = []
for ten, mo_hinh in [("64-32-16-1", mo_hinh_1), ("128-64-1", mo_hinh_2)]:
    du_doan = mo_hinh.predict(X_test)
    ket_qua.append({
        "Kiến trúc": ten,
        "Accuracy": round(accuracy_score(y_test, du_doan), 4),
        "F1-score": round(f1_score(y_test, du_doan), 4),
    })

print("\n10. So sánh trên tập test:")
print(pd.DataFrame(ket_qua).to_string(index=False))
