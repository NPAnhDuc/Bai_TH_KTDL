# -*- coding: utf-8 -*-
"""
THỰC HÀNH THÊM: Cây quyết định dự đoán thu nhập với các file xtrainV3, ytrainV3,
xtestV3, ytestV3.

Thư mục bài tập có sẵn 4 file này nhưng tài liệu không ghi đề bài cụ thể,
nên bài này làm theo các bước giống thực hành 7-2:
huấn luyện cây, đánh giá trên tập test và thử nhiều độ sâu.

Dữ liệu đã được mã hóa thành số và đã chia sẵn train, test.
income = 1 là thu nhập cao, income = 0 là thu nhập thấp.

Cài thư viện: pip install pandas scikit-learn
Chạy: python thuc_hanh_7_3_du_doan_thu_nhap.py
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

THU_MUC = Path(__file__).parent

# Đọc dữ liệu, cột income trong file y là nhãn
X_train = pd.read_csv(THU_MUC / "xtrainV3.csv")
y_train = pd.read_csv(THU_MUC / "ytrainV3.csv")["income"]
X_test = pd.read_csv(THU_MUC / "xtestV3.csv")
y_test = pd.read_csv(THU_MUC / "ytestV3.csv")["income"]

print("Kích thước tập train:", X_train.shape)
print("Kích thước tập test:", X_test.shape)

# Thử nhiều độ sâu để thấy hiện tượng quá khớp:
# cây càng sâu thì accuracy trên train càng cao, nhưng trên test có thể giảm
print("\nĐộ sâu | Accuracy train | Accuracy test")
ket_qua = {}
for do_sau in [2, 4, 6, 8, 10, 15, 20]:
    clf = DecisionTreeClassifier(max_depth=do_sau, random_state=12)
    clf.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, clf.predict(X_train))
    acc_test = accuracy_score(y_test, clf.predict(X_test))
    ket_qua[do_sau] = acc_test
    print(f"{do_sau:6d} | {acc_train:14.4f} | {acc_test:13.4f}")

# Đánh giá chi tiết cây có accuracy test cao nhất.
# Chọn độ sâu theo tập test ở đây chỉ để minh họa cho đơn giản,
# làm đúng thì nên chọn bằng tập validation như grid search ở câu 8
do_sau_tot_nhat = max(ket_qua, key=ket_qua.get)
print("\nĐộ sâu tốt nhất:", do_sau_tot_nhat)

clf = DecisionTreeClassifier(max_depth=do_sau_tot_nhat, random_state=12)
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))
