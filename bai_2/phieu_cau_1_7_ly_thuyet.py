# -*- coding: utf-8 -*-
"""
PHIẾU BÀI TẬP CÂY QUYẾT ĐỊNH - CÂU 1 ĐẾN CÂU 7 (lý thuyết).

Các câu này chủ yếu là tính toán nhỏ và trả lời lý thuyết,
nên chương trình tính kết quả rồi in câu trả lời ra màn hình.

Chạy: python phieu_cau_1_7_ly_thuyet.py
"""

import math

DO_SAU = 5
SO_QUAN_SAT = 10000
SO_LOP = 5

# Câu 1: mỗi tầng số nút tăng gấp đôi, nên cây đầy đủ có 2^độ sâu nút lá
so_nut_la = 2 ** DO_SAU
print(f"Câu 1: Cây nhị phân độ sâu {DO_SAU} có tối đa 2^{DO_SAU} = {so_nut_la} nút lá.")

# Câu 2: chia đều số quan sát cho các nút lá
trung_binh = SO_QUAN_SAT / so_nut_la
print(f"\nCâu 2: Trung bình mỗi nút lá có {SO_QUAN_SAT} / {so_nut_la} = {trung_binh} quan sát.")
print("   Khoảng 312 quan sát là đủ lớn để tin cậy kết luận phân loại ở nút lá.")
print("   Tuy nhiên đây chỉ là trung bình, thực tế có nút lá chứa rất ít quan sát")
print("   và kết luận ở những nút đó kém tin cậy hơn.")

# Câu 3: đi từ gốc xuống lá, mỗi tầng trả lời một câu hỏi
print(f"\nCâu 3: Giá trị dự báo được tạo thành từ {DO_SAU} câu hỏi,")
print("   mỗi tầng của cây tương ứng với một câu hỏi.")

# Câu 4: entropy lớn nhất khi các lớp có xác suất bằng nhau, p = 1/n
entropy_lon_nhat = math.log2(SO_LOP)
print(f"\nCâu 4: Entropy lớn nhất khi {SO_LOP} lớp có xác suất bằng nhau (p = 1/{SO_LOP}):")
print(f"   H = log2({SO_LOP}) = {entropy_lon_nhat:.4f}")

# Câu 5: đề ghi "câu 3" nhưng bài toán phân loại 5 lớp nằm ở câu 4
gini_lon_nhat = 1 - SO_LOP * (1 / SO_LOP) ** 2
print(f"\nCâu 5: Xét bài toán {SO_LOP} lớp. Gini lớn nhất cũng khi các lớp bằng nhau:")
print(f"   Gini = 1 - {SO_LOP} * (1/{SO_LOP})^2 = {gini_lon_nhat:.1f}")

print("\nCâu 6: Khi quá khớp nên GIẢM độ sâu của cây để mô hình đơn giản hơn.")

print("\nCâu 7: Giá trị dự báo của các quan sát cùng một nút là giống nhau:")
print("   - Bài toán hồi quy: lấy trung bình biến mục tiêu của các quan sát trong nút.")
print("   - Bài toán phân loại: lấy nhãn xuất hiện nhiều nhất trong nút.")
