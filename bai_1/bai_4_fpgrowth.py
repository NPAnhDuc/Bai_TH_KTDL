# -*- coding: utf-8 -*-
"""
BÀI 4: Dùng FP-Growth tìm tập phổ biến và luật kết hợp với minsup = 80%, minconf = 80%.

Thuật toán đã cài đặt ở bai_2_fpgrowth.py, bài này chỉ thay dữ liệu và ngưỡng.

Có 5 giao dịch nên minsup = 80% nghĩa là tập phổ biến phải xuất hiện ít nhất 4 lần.

Chạy: python bai_4_fpgrowth.py
"""

from bai_2_fpgrowth import giai_bai

# Cơ sở dữ liệu của đề bài (TID 100, 200, 300, 400, 500)
GIAO_DICH = [
    {"a", "d", "c", "q", "m", "p"},
    {"d", "b", "c", "a", "h", "m", "k"},
    {"b", "a", "g"},
    {"b", "c", "k", "p"},
    {"d", "a", "c", "e", "h", "p", "m", "g"},
]

MINSUP = 0.8
MINCONF = 0.8

if __name__ == "__main__":
    giai_bai(GIAO_DICH, MINSUP, MINCONF)
