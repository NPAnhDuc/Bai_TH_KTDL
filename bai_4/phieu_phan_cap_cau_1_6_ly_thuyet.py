# -*- coding: utf-8 -*-
"""
PHIẾU BÀI TẬP GOM CỤM BẰNG PHÂN CẤP - CÂU 1 ĐẾN CÂU 6 (lý thuyết).

Chạy: python phieu_phan_cap_cau_1_6_ly_thuyet.py
"""

print("Câu 1: Có 2 chiến lược.")
print("   - Hợp nhất (agglomerative): ban đầu mỗi điểm là một cụm, gộp dần đến khi còn 1 cụm.")
print("   - Phân chia (divisive): ban đầu tất cả là một cụm, chia nhỏ dần.")

print("\nCâu 2: Ban đầu mỗi quan sát là một cụm, nên N quan sát có N cụm.")

print("\nCâu 3: Mỗi bước gộp giảm đi 1 cụm, nên đi từ N cụm về 1 cụm mất N - 1 bước gộp.")

print("\nCâu 4: Có 4 cách đo khoảng cách giữa hai cụm:")
print("   - Single linkage: khoảng cách giữa hai điểm gần nhất của hai cụm.")
print("   - Complete linkage: khoảng cách giữa hai điểm xa nhất của hai cụm.")
print("   - Group average: trung bình khoảng cách giữa mọi cặp điểm của hai cụm.")
print("   - Ward linkage: mức tăng tổng bình phương sai số khi gộp hai cụm.")

print("\nCâu 5: Ward linkage chỉ phù hợp với dữ liệu trong không gian Euclid.")

print("\nCâu 6: Các cách dừng phân cụm:")
print("   - Dừng khi đạt số cụm k xác định trước.")
print("   - Dừng khi khoảng cách giữa hai cụm gần nhất vượt ngưỡng cho trước.")
print("   - Dừng khi việc gộp tạo ra cụm có độ gắn kết thấp hơn.")
