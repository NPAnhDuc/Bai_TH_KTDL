# -*- coding: utf-8 -*-
"""
THỰC HÀNH 8-2: Phân lớp bằng Naive Bayes trên dữ liệu PlayTennis.csv.

Naive Bayes dựa trên định lý Bayes với giả định các thuộc tính độc lập với nhau:
    P(lớp | dữ liệu) = P(dữ liệu | lớp) * P(lớp) / P(dữ liệu)
Mô hình chọn lớp có xác suất hậu nghiệm P(lớp | dữ liệu) lớn nhất.

Cài thư viện: pip install pandas scikit-learn
Chạy: python thuc_hanh_8_2_naive_bayes.py
"""

from pathlib import Path

import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Đọc dữ liệu: 14 mẫu, 4 thuộc tính điều kiện và cột nhãn play
data = pd.read_csv(Path(__file__).parent / "PlayTennis.csv")
print(data.head(14))

# GaussianNB làm việc với số, nên mã hóa các cột chữ thành các cột 0, 1
X_train = pd.get_dummies(data[["outlook", "temp", "humidity", "windy"]], dtype=int)
X_train["windy"] = X_train["windy"].astype(int)
y_train = data["play"]

print("\nDữ liệu sau khi mã hóa:")
print(X_train.head())

# Tạo và huấn luyện mô hình
model = GaussianNB()
model.fit(X_train, y_train)

# Dự đoán mẫu mới: outlook = overcast, temp = hot, humidity = high, windy = false.
# Tạo một dòng toàn 0 có đúng tên cột, rồi bật các cột tương ứng với mẫu lên 1
mau_moi = pd.DataFrame(0, index=[0], columns=X_train.columns)
mau_moi[["outlook_overcast", "temp_hot", "humidity_high"]] = 1

print("\nDự đoán cho mẫu mới:", model.predict(mau_moi)[0])

# Xác suất của từng lớp, theo thứ tự trong model.classes_
for lop, xac_suat in zip(model.classes_, model.predict_proba(mau_moi)[0]):
    print(f"   P({lop}) = {xac_suat:.4f}")

print("\nAccuracy trên chính dữ liệu huấn luyện:", round(model.score(X_train, y_train), 4))
