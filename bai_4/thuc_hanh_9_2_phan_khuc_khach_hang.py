# -*- coding: utf-8 -*-
"""
THỰC HÀNH 9-2: Phân khúc khách hàng bằng K-means trên Mall_Customers.csv.

Các bước:
    1. Khám phá dữ liệu
    2. Trực quan hóa dữ liệu
    3. Phân cụm theo Tuổi và Điểm chi tiêu
    4. Phân cụm theo Thu nhập và Điểm chi tiêu
    5. Phân cụm theo cả ba thuộc tính

Mỗi lần phân cụm đều dùng Elbow để chọn số cụm, rồi vẽ các cụm cùng đường biên.

Cài thư viện: pip install pandas numpy matplotlib scikit-learn
Chạy: python thuc_hanh_9_2_phan_khuc_khach_hang.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

THU_MUC = Path(__file__).parent

TUOI = "Age"
THU_NHAP = "Annual Income (k$)"
CHI_TIEU = "Spending Score (1-100)"


def tao_mo_hinh(so_cum):
    """Tạo mô hình K-means với các tham số dùng chung."""
    return KMeans(n_clusters=so_cum, init="k-means++", n_init=10, max_iter=300,
                  tol=0.0001, random_state=111)


def ve_elbow(X, ten_file):
    """Tính inertia với K từ 1 đến 10 và vẽ đồ thị Elbow."""
    inertia = [tao_mo_hinh(n).fit(X).inertia_ for n in range(1, 11)]

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, 11), inertia, marker="o", linewidth=2, color="#2a78d6")
    plt.xlabel("Số cụm")
    plt.ylabel("Inertia")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(THU_MUC / ten_file, dpi=150)
    plt.close()
    print("Đã lưu biểu đồ:", ten_file)


def ve_duong_bien(X, so_cum, ten_x, ten_y, ten_file):
    """Phân cụm dữ liệu 2 chiều, tô màu nền từng vùng và vẽ các điểm lên trên."""
    mo_hinh = tao_mo_hinh(so_cum).fit(X)

    # Tạo lưới điểm phủ kín biểu đồ, dự đoán cụm của từng điểm để tô màu nền
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))
    Z = mo_hinh.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.imshow(Z, extent=(xx.min(), xx.max(), yy.min(), yy.max()), origin="lower",
               cmap="Pastel2", aspect="auto", interpolation="nearest")
    plt.scatter(X[:, 0], X[:, 1], c=mo_hinh.labels_, s=40, cmap="Set1", edgecolor="white")
    plt.scatter(mo_hinh.cluster_centers_[:, 0], mo_hinh.cluster_centers_[:, 1],
                s=160, c="black", marker="X", label="Tâm cụm")
    plt.xlabel(ten_x)
    plt.ylabel(ten_y)
    plt.legend()
    plt.tight_layout()
    plt.savefig(THU_MUC / ten_file, dpi=150)
    plt.close()
    print("Đã lưu biểu đồ:", ten_file)


# ----- 1. Khám phá dữ liệu -----
df = pd.read_csv(THU_MUC / "Mall_Customers.csv")
print(df.head())
print("\nKích thước dữ liệu:", df.shape)
print(df.describe().round(2))
print("\nSố giá trị bị thiếu:", df.isnull().sum().sum())
print("\nSố khách hàng theo giới tính:")
print(df["Gender"].value_counts())

# ----- 2. Trực quan hóa dữ liệu -----
# Biểu đồ phân bố của 3 thuộc tính
plt.figure(figsize=(13, 4))
for vi_tri, cot in enumerate([TUOI, THU_NHAP, CHI_TIEU], start=1):
    plt.subplot(1, 3, vi_tri)
    plt.hist(df[cot], bins=20, color="#2a78d6", edgecolor="white")
    plt.title(f"Phân bố {cot}")
plt.tight_layout()
plt.savefig(THU_MUC / "phan_bo_thuoc_tinh.png", dpi=150)
plt.close()
print("\nĐã lưu biểu đồ: phan_bo_thuoc_tinh.png")

# Quan hệ giữa thu nhập và điểm chi tiêu theo giới tính
plt.figure(figsize=(9, 5))
for gioi_tinh, mau, hinh in [("Male", "#2a78d6", "o"), ("Female", "#eb6834", "s")]:
    nhom = df[df["Gender"] == gioi_tinh]
    plt.scatter(nhom[THU_NHAP], nhom[CHI_TIEU], c=mau, marker=hinh, s=50, alpha=0.7, label=gioi_tinh)
plt.xlabel(THU_NHAP)
plt.ylabel(CHI_TIEU)
plt.title("Thu nhập và điểm chi tiêu theo giới tính")
plt.legend()
plt.tight_layout()
plt.savefig(THU_MUC / "thu_nhap_chi_tieu.png", dpi=150)
plt.close()
print("Đã lưu biểu đồ: thu_nhap_chi_tieu.png")

# ----- 3. Phân cụm theo Tuổi và Điểm chi tiêu -----
# Đồ thị Elbow gấp tại K = 4
X1 = df[[TUOI, CHI_TIEU]].values
ve_elbow(X1, "elbow_tuoi_chi_tieu.png")
ve_duong_bien(X1, 4, TUOI, CHI_TIEU, "cum_tuoi_chi_tieu.png")

# ----- 4. Phân cụm theo Thu nhập và Điểm chi tiêu -----
# Đồ thị Elbow gấp tại K = 5
X2 = df[[THU_NHAP, CHI_TIEU]].values
ve_elbow(X2, "elbow_thu_nhap_chi_tieu.png")
ve_duong_bien(X2, 5, THU_NHAP, CHI_TIEU, "cum_thu_nhap_chi_tieu.png")

# ----- 5. Phân cụm theo cả ba thuộc tính -----
# Đồ thị Elbow gấp tại K = 6
X3 = df[[TUOI, THU_NHAP, CHI_TIEU]].values
ve_elbow(X3, "elbow_ba_thuoc_tinh.png")

mo_hinh = tao_mo_hinh(6).fit(X3)
df["Cum"] = mo_hinh.labels_

print("\nGiá trị trung bình của mỗi cụm khi dùng 3 thuộc tính:")
print(df.groupby("Cum")[[TUOI, THU_NHAP, CHI_TIEU]].mean().round(1))

# Biểu đồ 3 chiều
fig = plt.figure(figsize=(9, 7))
truc = fig.add_subplot(projection="3d")
truc.scatter(df[TUOI], df[CHI_TIEU], df[THU_NHAP], c=df["Cum"], cmap="Set1", s=40)
truc.set_xlabel(TUOI)
truc.set_ylabel(CHI_TIEU)
truc.set_zlabel(THU_NHAP)
truc.set_title("Các cụm theo 3 thuộc tính")
plt.tight_layout()
plt.savefig(THU_MUC / "cum_ba_thuoc_tinh.png", dpi=150)
plt.close()
print("\nĐã lưu biểu đồ: cum_ba_thuoc_tinh.png")
