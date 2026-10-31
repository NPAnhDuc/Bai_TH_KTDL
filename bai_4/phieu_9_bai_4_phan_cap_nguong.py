# -*- coding: utf-8 -*-
"""
PHIẾU BÀI 9 - BÀI 4: Gom cụm phân cấp với dữ liệu bài 3, ngưỡng delta = 2,2.

Dùng lại hàm gom cụm phân cấp của bài 2. Thuật toán dừng khi hai cụm gần nhất
cách nhau xa hơn ngưỡng, số cụm còn lại chính là đáp án của đề.

Chạy: python phieu_9_bai_4_phan_cap_nguong.py
"""

from phieu_9_bai_2_phan_cap import gom_cum_phan_cap
from phieu_9_bai_3_kmeans_loi_binh_phuong import DIEM

NGUONG = 2.2

if __name__ == "__main__":
    print("--- LIÊN KẾT ĐƠN ---")
    cum_don = gom_cum_phan_cap(DIEM, "don", NGUONG)
    print(f"=> Với ngưỡng {NGUONG} gom được {len(cum_don)} cụm: {cum_don}")

    print("\n--- LIÊN KẾT ĐẦY ĐỦ ---")
    cum_day_du = gom_cum_phan_cap(DIEM, "day_du", NGUONG)
    print(f"=> Với ngưỡng {NGUONG} gom được {len(cum_day_du)} cụm: {cum_day_du}")
