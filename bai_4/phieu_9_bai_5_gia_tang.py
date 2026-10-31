# -*- coding: utf-8 -*-
"""
PHIẾU BÀI 9 - BÀI 5: Gom cụm gia tăng với dữ liệu bài 1, ngưỡng delta = 3.
So sánh kết quả với K-means.

Thuật toán gom cụm gia tăng xét lần lượt từng điểm, mỗi điểm chỉ xét một lần:
    1. Điểm đầu tiên tạo thành cụm đầu tiên.
    2. Với điểm tiếp theo, tìm cụm có tâm gần nhất.
       - Nếu khoảng cách không vượt ngưỡng: đưa điểm vào cụm và tính lại tâm.
       - Ngược lại: điểm đó tạo thành một cụm mới.

Số cụm do ngưỡng quyết định, không cần chọn trước K như K-means.

Chạy: python phieu_9_bai_5_gia_tang.py
"""

import math

from phieu_9_bai_1_kmeans import DIEM, in_cac_cum, k_means, tinh_loi, tinh_tam

NGUONG = 3


def gom_cum_gia_tang(diem, nguong):
    """Gom cụm gia tăng, trả về danh sách các cụm."""
    cac_cum = []

    for ten, toa_do in diem.items():
        # Điểm đầu tiên tạo cụm đầu tiên
        if not cac_cum:
            cac_cum.append([ten])
            print(f"{ten}: tạo cụm C1")
            continue

        # Tính khoảng cách từ điểm đến tâm của từng cụm
        khoang_cach = [math.dist(toa_do, tinh_tam(cum, diem)) for cum in cac_cum]
        d = min(khoang_cach)
        vi_tri = khoang_cach.index(d)

        if d <= nguong:
            cac_cum[vi_tri].append(ten)
            print(f"{ten}: cách C{vi_tri + 1} {d:.2f} <= {nguong}, đưa vào C{vi_tri + 1}")
        else:
            cac_cum.append([ten])
            print(f"{ten}: gần nhất cách {d:.2f} > {nguong}, tạo cụm C{len(cac_cum)}")

    return cac_cum


if __name__ == "__main__":
    print(f"--- GOM CỤM GIA TĂNG, ngưỡng = {NGUONG} ---")
    cum_gia_tang = gom_cum_gia_tang(DIEM, NGUONG)
    in_cac_cum(cum_gia_tang, DIEM)
    print(f"   Lỗi bình phương E = {tinh_loi(cum_gia_tang, DIEM):.2f}")

    # So sánh với K-means, lấy K bằng số cụm mà gom cụm gia tăng tìm được
    k = len(cum_gia_tang)
    print(f"\n--- K-MEANS, K = {k} ---")
    cum_k_means = k_means(DIEM, k)
    in_cac_cum(cum_k_means, DIEM)
    print(f"   Lỗi bình phương E = {tinh_loi(cum_k_means, DIEM):.2f}")

    # Sắp xếp để không phụ thuộc thứ tự các cụm khi so sánh
    giong_nhau = sorted(cum_gia_tang) == sorted(cum_k_means)
    print("\n--- SO SÁNH ---")
    print("Hai thuật toán cho cùng kết quả:", "Có" if giong_nhau else "Không")
    print("Gom cụm gia tăng: không cần biết trước K, duyệt mỗi điểm một lần nên nhanh,")
    print("   nhưng kết quả phụ thuộc thứ tự điểm và giá trị ngưỡng.")
    print("K-means: phải chọn trước K, lặp nhiều lần và phụ thuộc tâm ban đầu.")
