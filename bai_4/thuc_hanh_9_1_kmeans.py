# -*- coding: utf-8 -*-
"""
THỰC HÀNH 9-1: Thuật toán K-means.

Gồm 3 phần:
    1. Tự cài K-means trên dữ liệu tạo ngẫu nhiên, so sánh với scikit-learn
    2. Dùng phương pháp Elbow chọn số cụm cho dữ liệu Mall_Customers.csv
    3. Giảm chiều bằng t-SNE để vẽ dữ liệu nhiều chiều

Cài thư viện: pip install pandas numpy matplotlib scipy scikit-learn
Chạy: python thuc_hanh_9_1_kmeans.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE

THU_MUC = Path(__file__).parent

# Màu và hình dạng điểm cho từng cụm
MAU_SAC = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100"]
HINH_DANG = ["o", "s", "^", "D", "v"]

np.random.seed(11)


def kmeans_init_centers(X, k):
    """Chọn ngẫu nhiên k điểm dữ liệu làm tâm ban đầu."""
    return X[np.random.choice(X.shape[0], k, replace=False)]


def kmeans_assign_labels(X, centers):
    """Gán mỗi điểm vào cụm có tâm gần nhất."""
    D = cdist(X, centers)
    return np.argmin(D, axis=1)


def kmeans_update_centers(X, labels, K):
    """Tâm mới của mỗi cụm là trung bình các điểm trong cụm."""
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        centers[k, :] = np.mean(X[labels == k, :], axis=0)
    return centers


def kmeans(X, K):
    """Lặp cho đến khi các tâm không đổi. Trả về tâm, nhãn và số vòng lặp."""
    centers = kmeans_init_centers(X, K)
    so_vong_lap = 0

    while True:
        so_vong_lap += 1
        labels = kmeans_assign_labels(X, centers)
        new_centers = kmeans_update_centers(X, labels, K)

        if np.allclose(centers, new_centers):
            return centers, labels, so_vong_lap
        centers = new_centers


def ve_cac_cum(X, labels, ten_file, tieu_de, centers=None):
    """Vẽ các điểm theo cụm và lưu ra file ảnh."""
    plt.figure(figsize=(8, 6))
    for k in np.unique(labels):
        plt.scatter(X[labels == k, 0], X[labels == k, 1], s=20, alpha=0.8,
                    c=MAU_SAC[k], marker=HINH_DANG[k], label=f"Cụm {k}")

    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], s=120, c="black", marker="X", label="Tâm cụm")

    plt.title(tieu_de)
    plt.legend()
    plt.tight_layout()
    plt.savefig(THU_MUC / ten_file, dpi=150)
    plt.close()
    print("Đã lưu biểu đồ:", ten_file)


# ----- 1. Tự cài K-means trên dữ liệu tạo ngẫu nhiên -----
# Tạo 3 nhóm điểm quanh các tâm (2, 2), (8, 3) và (3, 6), mỗi nhóm 500 điểm
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X = np.concatenate([np.random.multivariate_normal(m, cov, N) for m in means], axis=0)
K = 3

centers, labels, so_vong_lap = kmeans(X, K)
print("Tâm cụm do thuật toán tự cài tìm được, sau", so_vong_lap, "vòng lặp:")
print(centers)
ve_cac_cum(X, labels, "kmeans_tu_cai.png", "K-means tự cài đặt", centers)

# So sánh với thư viện scikit-learn
model = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X)
print("\nTâm cụm do scikit-learn tìm được:")
print(model.cluster_centers_)

# ----- 2. Phương pháp Elbow trên dữ liệu khách hàng -----
# Lấy 2 cột: thu nhập hằng năm và điểm chi tiêu
dataset = pd.read_csv(THU_MUC / "Mall_Customers.csv")
X_mall = dataset.iloc[:, [3, 4]].values

# WCSS là tổng bình phương khoảng cách từ các điểm đến tâm cụm của chúng
wcss = []
for i in range(1, 11):
    model = KMeans(n_clusters=i, init="k-means++", random_state=42, n_init=10)
    model.fit(X_mall)
    wcss.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o", linewidth=2, color="#2a78d6")
plt.xlabel("Số cụm")
plt.ylabel("WCSS")
plt.title("Phương pháp Elbow")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(THU_MUC / "elbow_mall.png", dpi=150)
plt.close()
print("\nĐã lưu biểu đồ: elbow_mall.png")

# Đồ thị gấp khúc rõ tại K = 5 nên chọn 5 cụm
model = KMeans(n_clusters=5, init="k-means++", random_state=42, n_init=10)
y_kmeans = model.fit_predict(X_mall)
ve_cac_cum(X_mall, y_kmeans, "cac_cum_mall.png", "Phân cụm khách hàng với K = 5",
           model.cluster_centers_)

# ----- 3. Biểu diễn dữ liệu nhiều chiều bằng t-SNE -----
# Tạo 250 điểm có 5 chiều thuộc 2 nhóm
X_5_chieu, _ = make_blobs(n_samples=250, centers=2, n_features=5,
                          cluster_std=3.0, random_state=123)
nhan_5_chieu = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(X_5_chieu)

# Biến đổi dữ liệu về 2 chiều để có thể vẽ được
X_tsne = TSNE(n_components=2, random_state=123).fit_transform(X_5_chieu)
ve_cac_cum(X_tsne, nhan_5_chieu, "tsne_5_chieu.png", "Dữ liệu 5 chiều biểu diễn bằng t-SNE")
