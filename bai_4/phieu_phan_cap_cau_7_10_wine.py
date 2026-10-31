# -*- coding: utf-8 -*-
"""
PHIẾU BÀI TẬP GOM CỤM BẰNG PHÂN CẤP - CÂU 7 ĐẾN CÂU 10: dữ liệu rượu vang (wine).

Câu 7: Chia train, test và khảo sát dữ liệu
Câu 8: Chuẩn hóa và giảm chiều dữ liệu
Câu 9: Xây dựng mô hình phân cụm, không dùng biến mục tiêu khi huấn luyện
Câu 10: Dùng biến mục tiêu thật để đánh giá, vẽ các cụm trong không gian 2 chiều

Cài thư viện: pip install pandas numpy matplotlib scipy scikit-learn
Chạy: python phieu_phan_cap_cau_7_10_wine.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

THU_MUC = Path(__file__).parent


def khop_nhan(nhan_cum, nhan_that):
    """
    Thuật toán phân cụm chỉ đặt tên cụm 0, 1, 2 mà không biết loại rượu thật.
    Hàm này ghép mỗi cụm với loại rượu xuất hiện nhiều nhất trong cụm đó.
    Trả về dict {tên cụm: nhãn thật}.
    """
    bang_ghep = {}
    for cum in np.unique(nhan_cum):
        cac_nhan, so_luong = np.unique(nhan_that[nhan_cum == cum], return_counts=True)
        bang_ghep[cum] = cac_nhan[np.argmax(so_luong)]
    return bang_ghep


# ----- Câu 7: chia dữ liệu và khảo sát -----
wine = load_wine(as_frame=True)
data = wine.frame
print("Kích thước dữ liệu:", data.shape)
print(data.describe().round(2))
print("\nCó dữ liệu bị thiếu không:", data.isnull().values.any())
print("\nSố mẫu mỗi loại rượu:")
print(data["target"].value_counts())

X = data.drop(columns=["target"])
y = data["target"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("\nKích thước train, test:", X_train.shape, X_test.shape)

# ----- Câu 8: chuẩn hóa và giảm chiều -----
# Chuẩn hóa về khoảng [0, 1] để các thuộc tính có đơn vị lớn không lấn át thuộc tính khác
chuan_hoa = MinMaxScaler()
X_train_cn = chuan_hoa.fit_transform(X_train)
X_test_cn = chuan_hoa.transform(X_test)

# Giảm về 2 chiều để vẽ biểu đồ, PCA và t-SNE là 2 cách phổ biến
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_cn)
X_tsne = TSNE(n_components=2, random_state=123).fit_transform(X_train_cn)
print("\nKích thước sau khi giảm chiều:", X_pca.shape)
print("Tỉ lệ thông tin PCA giữ lại:", round(pca.explained_variance_ratio_.sum(), 4))

# ----- Câu 9: phân cụm, không dùng nhãn y -----
# Vẽ dendrogram để chọn số cụm
plt.figure(figsize=(14, 6))
plt.title("Dendrogram của dữ liệu rượu vang")
shc.dendrogram(shc.linkage(X_train_cn, method="ward"), no_labels=True, color_threshold=5)
plt.axhline(5, linestyle="--", color="gray")
plt.ylabel("Khoảng cách giữa các cụm")
plt.tight_layout()
plt.savefig(THU_MUC / "wine_dendrogram.png", dpi=150)
plt.close()
print("\nĐã lưu biểu đồ: wine_dendrogram.png")

# Cắt dendrogram thành 3 cụm
cluster = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels = cluster.fit_predict(X_train_cn)

# ----- Câu 10: đánh giá bằng nhãn thật và vẽ biểu đồ -----
bang_ghep = khop_nhan(labels, y_train)
du_doan_train = np.array([bang_ghep[cum] for cum in labels])
print("Accuracy trên tập train:", round(accuracy_score(y_train, du_doan_train) * 100, 2), "%")

# AgglomerativeClustering không có hàm predict, nên mỗi mẫu test
# được gán vào cụm có tâm (tính từ tập train) gần nó nhất
cac_tam = np.array([X_train_cn[labels == cum].mean(axis=0) for cum in range(3)])
cum_test = np.argmin(cdist(X_test_cn, cac_tam), axis=1)
du_doan_test = np.array([bang_ghep[cum] for cum in cum_test])
print("Accuracy trên tập test:", round(accuracy_score(y_test, du_doan_test) * 100, 2), "%")

# Vẽ các cụm: mỗi cụm một màu và một hình dạng điểm
mau_sac = ["#2a78d6", "#eb6834", "#1baf7a"]
hinh_dang = ["o", "s", "^"]

fig, cac_truc = plt.subplots(1, 2, figsize=(13, 5))
for truc, toa_do, ten in zip(cac_truc, [X_pca, X_tsne], ["PCA", "t-SNE"]):
    for cum in range(3):
        truc.scatter(toa_do[labels == cum, 0], toa_do[labels == cum, 1],
                     c=mau_sac[cum], marker=hinh_dang[cum], label=f"Cụm {cum}")
    truc.set_title(f"Các cụm trong không gian 2 chiều ({ten})")
    truc.legend()

plt.tight_layout()
plt.savefig(THU_MUC / "wine_cac_cum.png", dpi=150)
plt.close()
print("Đã lưu biểu đồ: wine_cac_cum.png")
