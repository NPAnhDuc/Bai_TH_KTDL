# -*- coding: utf-8 -*-
"""
THỰC HÀNH 6-2: Tìm tập phổ biến và luật kết hợp trên Market_Basket_Optimisation.csv
bằng hàm fpgrowth của thư viện mlxtend.

Các bước:
    1. Đọc dữ liệu và đếm số lần mua của từng mặt hàng
    2. Vẽ biểu đồ 20 mặt hàng bán chạy nhất
    3. Mã hóa dữ liệu thành bảng True/False
    4. Giữ lại 30 mặt hàng bán chạy nhất
    5. Chạy FP-Growth rồi sinh luật kết hợp

Cài thư viện: pip install pandas matplotlib mlxtend
Chạy: python thuc_hanh_6_2_fpgrowth.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

THU_MUC = Path(__file__).parent

# Hiển thị đủ các cột khi in bảng
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

# ----- 1. Đọc dữ liệu -----
# File không có dòng tiêu đề nên phải dùng header=None
dataset = pd.read_csv(THU_MUC / "Market_Basket_Optimisation.csv", header=None)
print("Kích thước dữ liệu (dòng, cột):", dataset.shape)

# Chuyển mỗi dòng thành list mặt hàng, bỏ các ô trống (NaN)
transaction = []
for i in range(dataset.shape[0]):
    dong = dataset.iloc[i].dropna()
    transaction.append([str(mat_hang).strip() for mat_hang in dong])

# Mỗi lần xuất hiện của một mặt hàng là một dòng, sau đó gom nhóm để đếm
tat_ca_mat_hang = [mat_hang for gd in transaction for mat_hang in gd]
df = pd.DataFrame(tat_ca_mat_hang, columns=["items"])
df["incident_count"] = 1
df_table = df.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()

print("\n5 mặt hàng bán chạy nhất:")
print(df_table.head(5))

# ----- 2. Vẽ biểu đồ 20 mặt hàng bán chạy nhất -----
top_20 = df_table.head(20).iloc[::-1]  # đảo ngược để mặt hàng lớn nhất nằm trên cùng

plt.figure(figsize=(9, 7))
plt.barh(top_20["items"], top_20["incident_count"], color="#2a78d6")
plt.xlabel("Số lần xuất hiện")
plt.title("20 mặt hàng bán chạy nhất")
plt.tight_layout()
plt.savefig(THU_MUC / "bieu_do_top_20_mat_hang.png", dpi=150)
plt.close()
print("\nĐã lưu biểu đồ: bieu_do_top_20_mat_hang.png")

# ----- 3. Mã hóa dữ liệu -----
# Mỗi cột là một mặt hàng, mỗi dòng là một giao dịch.
# True nghĩa là giao dịch đó có mua mặt hàng này.
te = TransactionEncoder()
te_ary = te.fit(transaction).transform(transaction)
dataset = pd.DataFrame(te_ary, columns=te.columns_)
print("\nKích thước sau khi mã hóa:", dataset.shape)

# ----- 4. Giữ lại 30 mặt hàng bán chạy nhất -----
first30 = df_table["items"].head(30).values
dataset = dataset.loc[:, first30]
print("Kích thước sau khi lấy top 30:", dataset.shape)

# ----- 5. Chạy FP-Growth và sinh luật -----
res = fpgrowth(dataset, min_support=0.05, use_colnames=True)
print("\nCác tập phổ biến với min_support = 0.05:")
print(res)

# Chỉ giữ các luật có lift từ 1 trở lên, sắp xếp theo confidence giảm dần
rules = association_rules(res, metric="lift", min_threshold=1)
rules = rules.sort_values("confidence", ascending=False)

print("\nCác luật kết hợp:")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
