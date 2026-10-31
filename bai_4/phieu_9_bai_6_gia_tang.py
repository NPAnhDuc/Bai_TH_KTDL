# -*- coding: utf-8 -*-
"""
PHIẾU BÀI 9 - BÀI 6: Cài đặt gom cụm gia tăng với dữ liệu bài 3.

Đề bài không cho ngưỡng, bài này dùng ngưỡng delta = 3 giống bài 5.

Chạy: python phieu_9_bai_6_gia_tang.py
"""

from phieu_9_bai_1_kmeans import in_cac_cum, tinh_loi
from phieu_9_bai_3_kmeans_loi_binh_phuong import DIEM
from phieu_9_bai_5_gia_tang import NGUONG, gom_cum_gia_tang

if __name__ == "__main__":
    cac_cum = gom_cum_gia_tang(DIEM, NGUONG)

    print("\nKết quả:")
    in_cac_cum(cac_cum, DIEM)
    print(f"   Lỗi bình phương E = {tinh_loi(cac_cum, DIEM):.2f}")
