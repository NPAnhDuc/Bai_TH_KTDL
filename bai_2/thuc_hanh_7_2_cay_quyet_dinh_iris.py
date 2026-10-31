# -*- coding: utf-8 -*-
"""
THỰC HÀNH 7-2: Ứng dụng cây quyết định cho bộ dữ liệu iris.

Biến mục tiêu là 1 nếu là loài Versicolor, 0 nếu là các loài còn lại.
Biến đầu vào là độ dài, độ rộng của đài hoa và cánh hoa.

Các bước:
    1. Nạp dữ liệu từ thư viện scikit-learn
    2. Phân tích dữ liệu
    3. Chia dữ liệu thành tập train và test
    4. Tạo và huấn luyện mô hình
    5. Đánh giá mô hình trên tập test

Cài thư viện: pip install pandas matplotlib scikit-learn
Chạy: python thuc_hanh_7_2_cay_quyet_dinh_iris.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, metrics
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

THU_MUC = Path(__file__).parent

# ----- 1. Nạp dữ liệu -----
# Thứ tự cột của iris là: đài hoa dài, đài hoa rộng, cánh hoa dài, cánh hoa rộng
iris = datasets.load_iris()
data = pd.DataFrame(
    iris["data"],
    columns=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
)
data["Species"] = [iris["target_names"][nhan] for nhan in iris["target"]]
print(data.head())

# ----- 2. Phân tích dữ liệu -----
print("\nCác loài hoa:", data["Species"].unique())
print(data.groupby("Species").mean().round(2))

# Chuyển về bài toán 2 lớp: versicolor là 1, còn lại là 0
data["Species"] = data["Species"].map({"versicolor": 1, "setosa": 0, "virginica": 0})
print("\nSố mẫu mỗi nhãn:")
print(data["Species"].value_counts())

# ----- 3. Chia dữ liệu thành tập train và test -----
X = data[["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]].values
y = data["Species"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=1, stratify=y
)

# ----- 4. Tạo và huấn luyện mô hình -----
clf = DecisionTreeClassifier(random_state=12)
clf.fit(X_train, y_train)

# ----- 5. Đánh giá mô hình -----
prediction = clf.predict(X_test)
print("\nNhãn dự đoán:", prediction)
print("Accuracy:", metrics.accuracy_score(y_test, prediction))
print(classification_report(y_test, prediction))

# Ma trận nhầm lẫn: hàng là nhãn thật, cột là nhãn dự đoán
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, cmap="Blues")
plt.title("Ma trận nhầm lẫn của cây quyết định")
plt.tight_layout()
plt.savefig(THU_MUC / "ma_tran_nham_lan_iris.png", dpi=150)
plt.close()
print("Đã lưu biểu đồ: ma_tran_nham_lan_iris.png")
