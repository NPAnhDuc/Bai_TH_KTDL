# -*- coding: utf-8 -*-
"""
THỰC HÀNH 6-1: Tìm luật kết hợp trên dữ liệu store_data.csv bằng thư viện apyori.

Các bước:
    B1: Đặt giá trị tối thiểu của độ hỗ trợ và độ tin cậy
    B2: Tìm các tập thỏa mãn độ hỗ trợ
    B3: Tìm các luật thỏa mãn độ tin cậy
    B4: Sắp xếp các luật theo lift giảm dần

Cài thư viện: pip install pandas apyori
Chạy: python thuc_hanh_6_1_apriori.py
"""

from pathlib import Path

import pandas as pd
from apyori import apriori

# File dữ liệu nằm cùng thư mục với file code này
FILE_DU_LIEU = Path(__file__).parent / "store_data.csv"

# File không có dòng tiêu đề nên phải dùng header=None
store_data = pd.read_csv(FILE_DU_LIEU, header=None)
print("Kích thước dữ liệu (dòng, cột):", store_data.shape)
print(store_data.head())

# Thư viện apyori làm việc với list, nên chuyển mỗi dòng thành một list mặt hàng.
# Các ô trống (NaN) phải bỏ đi, nếu không "nan" sẽ bị coi là một mặt hàng.
records = []
for i in range(store_data.shape[0]):
    dong = store_data.iloc[i].dropna()
    records.append([str(mat_hang).strip() for mat_hang in dong])

print("\nGiao dịch đầu tiên:", records[0])
print("Giao dịch thứ hai:", records[1])

# Chọn tham số như bài thực hành:
#   min_support = 0.0045: mặt hàng được mua khoảng 35 lần/tuần, 35/7500 = 0.0045
#   min_confidence = 0.2: độ tin cậy ít nhất 20%
#   min_lift = 3
#   min_length = 2: luật có ít nhất 2 sản phẩm
association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)

# Mỗi kết quả gồm một tập mặt hàng và các cách tách tập đó thành luật X -> Y.
# Ở đây tách ra từng luật để in đúng vế trái, vế phải.
danh_sach_luat = []
for ket_qua in association_results:
    for luat in ket_qua.ordered_statistics:
        if len(luat.items_base) == 0:
            continue  # bỏ luật có vế trái rỗng

        danh_sach_luat.append({
            "ve_trai": ", ".join(sorted(luat.items_base)),
            "ve_phai": ", ".join(sorted(luat.items_add)),
            "support": ket_qua.support,
            "confidence": luat.confidence,
            "lift": luat.lift,
        })

# B4: sắp xếp theo lift giảm dần
danh_sach_luat.sort(key=lambda luat: luat["lift"], reverse=True)

print("\nSố tập mặt hàng thỏa mãn:", len(association_results))
print("Số luật thỏa mãn:", len(danh_sach_luat))

print("\n10 luật có lift cao nhất:")
for luat in danh_sach_luat[:10]:
    print("Rule:", luat["ve_trai"], "->", luat["ve_phai"])
    print("Support:", round(luat["support"], 4))
    print("Confidence:", round(luat["confidence"], 4))
    print("Lift:", round(luat["lift"], 4))
    print("=====================================")
