# -*- coding: utf-8 -*-
"""
PHIẾU BÀI TẬP CÂY QUYẾT ĐỊNH - CÂU 8.

Dùng bộ dữ liệu iris, dự báo biến mục tiêu:
    1 nếu là loài Versicolor, 0 nếu là các loài còn lại.
Dùng grid search để tìm bộ tham số của cây quyết định có độ chính xác tốt nhất.

Grid search thử lần lượt mọi tổ hợp tham số. Mỗi tổ hợp được đánh giá bằng
kiểm định chéo (cross validation) trên tập train, tập test chỉ dùng ở bước cuối.

Cài thư viện: pip install scikit-learn
Chạy: python phieu_cau_8_grid_search.py
"""

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

# Đọc dữ liệu. Trong iris, nhãn 1 là loài Versicolor
iris = load_iris()
X = iris.data
y = (iris.target == 1).astype(int)

print("Số mẫu nhãn 1 (Versicolor):", y.sum())
print("Số mẫu nhãn 0 (loài khác):", (y == 0).sum())

# Chia dữ liệu 70% train, 30% test, giữ nguyên tỉ lệ nhãn nhờ stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=1, stratify=y
)

# Các giá trị tham số cần thử
tham_so = {
    "max_depth": [2, 3, 4, 5, 6, None],   # độ sâu tối đa
    "min_samples_leaf": [1, 2, 5, 10],    # số mẫu tối thiểu tại nút lá
    "max_leaf_nodes": [4, 8, 16, None],   # số nút lá tối đa
}

# Chia train thành 5 phần để kiểm định chéo
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=12),
    tham_so,
    scoring="accuracy",
    cv=5,
)
grid.fit(X_train, y_train)

print("\nBộ tham số tốt nhất:", grid.best_params_)
print("Accuracy kiểm định chéo:", round(grid.best_score_, 4))

# Đánh giá mô hình tốt nhất trên tập test
du_doan = grid.best_estimator_.predict(X_test)
print("Accuracy trên tập test:", round(accuracy_score(y_test, du_doan), 4))
print(classification_report(y_test, du_doan))
