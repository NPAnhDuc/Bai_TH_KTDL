# -*- coding: utf-8 -*-
"""
THỰC HÀNH 6-3: Dùng cả Apriori và FP-Growth của thư viện mlxtend trên
Market_Basket_Optimisation.csv, sau đó so sánh thời gian chạy của hai thuật toán.

Các bước:
    1. Đọc và mã hóa dữ liệu thành bảng True/False
    2. Tìm tập phổ biến bằng Apriori, sinh luật kết hợp
    3. Lọc tập phổ biến theo số phần tử
    4. Tìm tập phổ biến bằng FP-Growth, sinh luật kết hợp
    5. Đo và vẽ biểu đồ thời gian chạy của hai thuật toán

Cài thư viện: pip install pandas matplotlib mlxtend
Chạy: python thuc_hanh_6_3_so_sanh.py
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

THU_MUC = Path(__file__).parent

# Hiển thị đủ các cột khi in bảng
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

# ----- 1. Đọc và mã hóa dữ liệu -----
# File không có dòng tiêu đề nên phải dùng header=None
data = pd.read_csv(THU_MUC / "Market_Basket_Optimisation.csv", header=None)
print("Kích thước dữ liệu (dòng, cột):", data.shape)

# Chuyển mỗi dòng thành list mặt hàng, bỏ các ô trống (NaN)
trans = []
for i in range(data.shape[0]):
    dong = data.iloc[i].dropna()
    trans.append([str(mat_hang).strip() for mat_hang in dong])

t = TransactionEncoder()
data = pd.DataFrame(t.fit_transform(trans), columns=t.columns_)
print("Kích thước sau khi mã hóa:", data.shape)

# ----- 2. Apriori -----
# Tìm các tập có độ hỗ trợ ít nhất 5%
freq_items = apriori(data, min_support=0.05, use_colnames=True)
print("\nApriori tìm được", len(freq_items), "tập phổ biến")

rules_apriori = association_rules(freq_items, metric="lift", min_threshold=1.3)
print("\nCác luật từ Apriori (lift >= 1.3):")
print(rules_apriori[["antecedents", "consequents", "support", "confidence", "lift"]])

# ----- 3. Lọc tập phổ biến theo số phần tử -----
freq_items["length"] = freq_items["itemsets"].apply(len)

print("\nCác tập phổ biến có 2 phần tử:")
print(freq_items[freq_items["length"] == 2])

# ----- 4. FP-Growth -----
res = fpgrowth(data, min_support=0.05, use_colnames=True)
print("\nFP-Growth tìm được", len(res), "tập phổ biến")

rules_fp = association_rules(res, metric="lift", min_threshold=1)
print("\nCác luật từ FP-Growth (lift >= 1):")
print(rules_fp[["antecedents", "consequents", "support", "confidence", "lift"]])

# ----- 5. So sánh thời gian chạy -----
cac_nguong = [0.01, 0.02, 0.03, 0.04, 0.05]
thoi_gian_apriori = []
thoi_gian_fp = []

for nguong in cac_nguong:
    bat_dau = time.time()
    apriori(data, min_support=nguong, use_colnames=True)
    thoi_gian_apriori.append((time.time() - bat_dau) * 1000)  # đổi sang mili giây

    bat_dau = time.time()
    fpgrowth(data, min_support=nguong, use_colnames=True)
    thoi_gian_fp.append((time.time() - bat_dau) * 1000)

print("\nThời gian chạy (ms):")
print(pd.DataFrame({
    "min_support": cac_nguong,
    "apriori": [round(ms, 1) for ms in thoi_gian_apriori],
    "fpgrowth": [round(ms, 1) for ms in thoi_gian_fp],
}))

plt.figure(figsize=(8, 5))
plt.plot(cac_nguong, thoi_gian_apriori, marker="o", linewidth=2, color="#2a78d6", label="Apriori")
plt.plot(cac_nguong, thoi_gian_fp, marker="o", linewidth=2, color="#eb6834", label="FP-Growth")
plt.xticks(cac_nguong)
plt.xlabel("min_support")
plt.ylabel("Thời gian chạy (ms)")
plt.title("So sánh thời gian chạy Apriori và FP-Growth")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(THU_MUC / "bieu_do_so_sanh_thoi_gian.png", dpi=150)
plt.close()
print("\nĐã lưu biểu đồ: bieu_do_so_sanh_thoi_gian.png")
