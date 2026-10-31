# -*- coding: utf-8 -*-
"""
PHIẾU BÀI 9 - BÀI 2: Gom cụm phân cấp với liên kết đơn và liên kết đầy đủ.

Thuật toán (chiến lược hợp nhất):
    1. Ban đầu mỗi điểm là một cụm.
    2. Tìm hai cụm gần nhau nhất và hợp chúng lại.
    3. Lặp lại đến khi chỉ còn 1 cụm, hoặc khoảng cách vượt ngưỡng cho trước.

Khoảng cách giữa hai cụm:
    - Liên kết đơn: khoảng cách giữa hai điểm GẦN nhất của hai cụm.
    - Liên kết đầy đủ: khoảng cách giữa hai điểm XA nhất của hai cụm.

Chạy: python phieu_9_bai_2_phan_cap.py
"""

import math

from phieu_9_bai_1_kmeans import DIEM


def khoang_cach_cum(cum_a, cum_b, diem, kieu):
    """Tính khoảng cách giữa hai cụm theo kiểu 'don' hoặc 'day_du'."""
    cac_khoang_cach = [math.dist(diem[a], diem[b]) for a in cum_a for b in cum_b]
    if kieu == "don":
        return min(cac_khoang_cach)
    return max(cac_khoang_cach)


def gom_cum_phan_cap(diem, kieu, nguong=None):
    """
    Gom cụm phân cấp, trả về danh sách các cụm.
    Nếu có ngưỡng thì dừng khi hai cụm gần nhất cách nhau xa hơn ngưỡng.
    """
    # Bước 1: mỗi điểm là một cụm
    cac_cum = [[ten] for ten in diem]

    while len(cac_cum) > 1:
        # Bước 2: tìm cặp cụm gần nhau nhất
        gan_nhat = None
        for i in range(len(cac_cum)):
            for j in range(i + 1, len(cac_cum)):
                d = khoang_cach_cum(cac_cum[i], cac_cum[j], diem, kieu)
                if gan_nhat is None or d < gan_nhat[0]:
                    gan_nhat = (d, i, j)

        d, i, j = gan_nhat

        # Hai cụm gần nhất đã vượt ngưỡng thì không hợp được nữa
        if nguong is not None and d > nguong:
            print(f"Khoảng cách nhỏ nhất {d:.2f} > ngưỡng {nguong}, dừng.")
            break

        # Hợp cụm j vào cụm i
        print(f"Hợp {cac_cum[i]} và {cac_cum[j]}, khoảng cách = {d:.2f}")
        cac_cum[i] = cac_cum[i] + cac_cum[j]
        del cac_cum[j]
        print(f"   Các cụm hiện tại: {cac_cum}")

    return cac_cum


if __name__ == "__main__":
    print("--- LIÊN KẾT ĐƠN ---")
    gom_cum_phan_cap(DIEM, "don")

    print("\n--- LIÊN KẾT ĐẦY ĐỦ ---")
    gom_cum_phan_cap(DIEM, "day_du")
