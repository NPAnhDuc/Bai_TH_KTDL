# -*- coding: utf-8 -*-
"""
PHIẾU BÀI 9 - BÀI 3: K-means với điều kiện hội tụ dùng lỗi bình phương.

Thuật toán dừng khi lỗi bình phương E không vượt quá ngưỡng delta = 6,2.

Đề bài không cho số cụm và cách khởi tạo, bài này chọn:
    K = 2, cụm ban đầu C1 = {x1, x2, x4}, C2 = {x3, x5}.

Chạy: python phieu_9_bai_3_kmeans_loi_binh_phuong.py
"""

from phieu_9_bai_1_kmeans import gan_vao_cum, in_cac_cum, tinh_loi, tinh_tam

# Dữ liệu của đề bài
DIEM = {
    "x1": (0, 2),
    "x2": (0, 0),
    "x3": (1.5, 0),
    "x4": (5, 0),
    "x5": (5, 2),
}

CUM_BAN_DAU = [["x1", "x2", "x4"], ["x3", "x5"]]
DELTA = 6.2


def k_means_theo_loi(diem, cum_ban_dau, delta):
    """Lặp cho đến khi lỗi bình phương E <= delta hoặc các cụm không đổi."""
    cac_cum = cum_ban_dau
    lan_lap = 0

    while True:
        lan_lap += 1
        loi = tinh_loi(cac_cum, diem)

        print(f"Lần lặp {lan_lap}:")
        in_cac_cum(cac_cum, diem)
        print(f"   Lỗi bình phương E = {loi:.2f}")

        # Điều kiện hội tụ của đề bài
        if loi <= delta:
            print(f"E <= {delta}, thuật toán hội tụ.")
            return cac_cum

        # Tính lại tâm rồi gán lại các điểm vào cụm gần nhất
        cac_tam = [tinh_tam(cum, diem) for cum in cac_cum]
        cum_moi = gan_vao_cum(cac_tam, diem)

        if cum_moi == cac_cum:
            print("Các cụm không đổi, thuật toán dừng.")
            return cac_cum
        cac_cum = cum_moi


if __name__ == "__main__":
    k_means_theo_loi(DIEM, CUM_BAN_DAU, DELTA)
