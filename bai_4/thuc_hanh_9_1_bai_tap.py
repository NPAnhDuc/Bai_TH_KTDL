# -*- coding: utf-8 -*-
"""
THỰC HÀNH 9-1 - PHẦN BÀI TẬP CUỐI BÀI (10 câu).

Câu 1 - 6, 9, 10 là lý thuyết, câu 7 - 8 là thực hành.

Câu 7 và 8 dùng bộ chữ số viết tay load_digits có sẵn trong scikit-learn:
1797 ảnh 8x8 giống MNIST nhưng nhỏ hơn, không cần tải về và chạy nhanh.

Cài thư viện: pip install numpy matplotlib scikit-learn
Chạy: python thuc_hanh_9_1_bai_tap.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

THU_MUC = Path(__file__).parent

print("Câu 1: K-means là học không giám sát, vì dữ liệu huấn luyện không có nhãn.")

print("\nCâu 2: Các bước của K-means:")
print("   B1: Chọn K điểm làm tâm ban đầu.")
print("   B2: Gán mỗi điểm vào cụm có tâm gần nhất.")
print("   B3: Tính lại tâm mỗi cụm bằng trung bình các điểm trong cụm.")
print("   B4: Lặp lại B2, B3 đến khi các tâm không đổi.")

print("\nCâu 3: Cả bước gán cụm và bước tính lại tâm đều không làm tăng tổng bình phương")
print("   khoảng cách. Số cách chia cụm là hữu hạn nên thuật toán luôn dừng sau hữu hạn bước.")
print("   Kết quả có thể chỉ là tối ưu cục bộ.")

print("\nCâu 4: Hạn chế của K-means:")
print("   - Phải chọn trước số cụm.")
print("   - Kết quả phụ thuộc tâm khởi tạo.")
print("   - Không tốt với cụm có hình dạng phức tạp hoặc mất cân bằng.")
print("   - Nhạy cảm với điểm ngoại lai và với đơn vị của các biến.")
print("   - Tốn bộ nhớ với dữ liệu rất lớn.")

print("\nCâu 5: Dùng phương pháp Elbow để xác định số cụm.")

print("\nCâu 6: Dùng phương pháp giảm chiều như t-SNE hoặc PCA, đưa dữ liệu về")
print("   2 hoặc 3 chiều rồi vẽ biểu đồ.")

# ----- Câu 7: phân cụm ảnh và chọn số cụm bằng Elbow -----
digits = load_digits()
X = digits.data
print("\nCâu 7: Kích thước dữ liệu ảnh:", X.shape)

cac_so_cum = range(2, 21, 2)
wcss = []
for k in cac_so_cum:
    wcss.append(KMeans(n_clusters=k, random_state=0, n_init=10).fit(X).inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(cac_so_cum), wcss, marker="o", linewidth=2, color="#2a78d6")
plt.xlabel("Số cụm")
plt.ylabel("WCSS")
plt.title("Phương pháp Elbow trên dữ liệu chữ số")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(THU_MUC / "elbow_chu_so.png", dpi=150)
plt.close()
print("   Đã lưu biểu đồ: elbow_chu_so.png")
print("   WCSS giảm đều, điểm gấp không rõ. Dữ liệu có 10 chữ số nên chọn K = 10.")

labels = KMeans(n_clusters=10, random_state=0, n_init=10).fit_predict(X)

# ----- Câu 8: giảm chiều bằng t-SNE và vẽ các cụm -----
X_tsne = TSNE(n_components=2, random_state=123).fit_transform(X)

plt.figure(figsize=(9, 7))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=12, cmap="tab10")

# Ghi số thứ tự cụm tại trung vị của mỗi cụm để dễ phân biệt
for k in range(10):
    x_giua, y_giua = np.median(X_tsne[labels == k], axis=0)
    plt.text(x_giua, y_giua, str(k), fontsize=16, fontweight="bold",
             bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})

plt.title("Các cụm chữ số biểu diễn bằng t-SNE")
plt.tight_layout()
plt.savefig(THU_MUC / "tsne_chu_so.png", dpi=150)
plt.close()
print("\nCâu 8: Đã lưu biểu đồ: tsne_chu_so.png")

print("\nCâu 9: Ứng dụng trong kinh doanh: phân khúc khách hàng, phân nhóm sản phẩm,")
print("   chọn vị trí cửa hàng, phát hiện giao dịch bất thường.")

print("\nCâu 10: Trong xử lý ảnh, K-means thường dùng để phân đoạn ảnh và nén ảnh")
print("   bằng cách gom các điểm ảnh có màu gần nhau.")
