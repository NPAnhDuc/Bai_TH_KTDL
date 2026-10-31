# -*- coding: utf-8 -*-
"""
THỰC HÀNH 7-1: Cài đặt thuật toán ID3 trên dữ liệu PlayTennis.csv.

Công thức:
    Entropy = - tổng (p * log2(p))
    Information Gain = Entropy(cha) - tổng (tỉ lệ mẫu của nhánh * Entropy(nhánh))

Thuật toán ID3:
    1. Tính Information Gain của từng thuộc tính.
    2. Chọn thuộc tính có Information Gain lớn nhất làm nút.
    3. Với mỗi giá trị của thuộc tính, lấy phần dữ liệu tương ứng rồi lặp lại.
    4. Dừng khi các mẫu cùng một lớp hoặc đã dùng hết thuộc tính.

Cài thư viện: pip install pandas
Chạy: python thuc_hanh_7_1_id3.py
"""

import math
from pathlib import Path

import pandas as pd

COT_NHAN = "play"


def tinh_entropy(nhan):
    """Tính entropy của một cột nhãn."""
    entropy = 0
    for so_luong in nhan.value_counts():
        p = so_luong / len(nhan)
        entropy -= p * math.log2(p)
    return entropy


def tinh_gain(du_lieu, thuoc_tinh):
    """Tính Information Gain khi chia dữ liệu theo một thuộc tính."""
    entropy_sau_khi_chia = 0
    for _, nhom in du_lieu.groupby(thuoc_tinh):
        ti_le = len(nhom) / len(du_lieu)
        entropy_sau_khi_chia += ti_le * tinh_entropy(nhom[COT_NHAN])
    return tinh_entropy(du_lieu[COT_NHAN]) - entropy_sau_khi_chia


def xay_dung_cay(du_lieu, cac_thuoc_tinh):
    """
    Xây dựng cây bằng đệ quy.
    Nút lá là tên nhãn, nút trong là dict {thuộc tính: {giá trị: cây con}}.
    """
    # Tất cả mẫu cùng một lớp thì tạo nút lá
    if du_lieu[COT_NHAN].nunique() == 1:
        return du_lieu[COT_NHAN].iloc[0]

    # Hết thuộc tính thì lấy nhãn xuất hiện nhiều nhất
    if not cac_thuoc_tinh:
        return du_lieu[COT_NHAN].mode()[0]

    # Chọn thuộc tính có Information Gain lớn nhất
    cac_gain = {tt: tinh_gain(du_lieu, tt) for tt in cac_thuoc_tinh}
    tot_nhat = max(cac_gain, key=cac_gain.get)

    print(f"Xét {len(du_lieu)} mẫu, entropy = {tinh_entropy(du_lieu[COT_NHAN]):.3f}")
    for thuoc_tinh, gain in cac_gain.items():
        print(f"   Gain({thuoc_tinh}) = {gain:.3f}")
    print(f"   => Chọn thuộc tính: {tot_nhat}")

    # Tạo một nhánh cho mỗi giá trị của thuộc tính được chọn
    thuoc_tinh_con_lai = [tt for tt in cac_thuoc_tinh if tt != tot_nhat]
    cay = {tot_nhat: {}}
    for gia_tri, nhom in du_lieu.groupby(tot_nhat):
        cay[tot_nhat][gia_tri] = xay_dung_cay(nhom, thuoc_tinh_con_lai)
    return cay


def in_cay(cay, muc=0):
    """In cây ra màn hình, nhánh con thụt vào so với nhánh cha."""
    thuoc_tinh = next(iter(cay))
    for gia_tri, cay_con in cay[thuoc_tinh].items():
        if isinstance(cay_con, dict):
            print("    " * muc + f"{thuoc_tinh} = {gia_tri}:")
            in_cay(cay_con, muc + 1)
        else:
            print("    " * muc + f"{thuoc_tinh} = {gia_tri} => {cay_con}")


def lay_luat(cay, dieu_kien=None):
    """Mỗi đường đi từ gốc đến nút lá là một luật IF ... THEN ..."""
    dieu_kien = dieu_kien or []
    if not isinstance(cay, dict):
        return [f"IF {' AND '.join(dieu_kien)} THEN {COT_NHAN} = {cay}"]

    thuoc_tinh = next(iter(cay))
    cac_luat = []
    for gia_tri, cay_con in cay[thuoc_tinh].items():
        cac_luat += lay_luat(cay_con, dieu_kien + [f"{thuoc_tinh} = {gia_tri}"])
    return cac_luat


def du_doan(cay, mau):
    """Đi từ gốc xuống nút lá theo giá trị của mẫu để lấy nhãn dự đoán."""
    while isinstance(cay, dict):
        thuoc_tinh = next(iter(cay))
        cay = cay[thuoc_tinh][mau[thuoc_tinh]]
    return cay


if __name__ == "__main__":
    # dtype=str để cột windy giữ nguyên chữ false/true
    data = pd.read_csv(Path(__file__).parent / "PlayTennis.csv", dtype=str)
    print("Kích thước dữ liệu:", data.shape)
    print(data.head(), "\n")

    thuoc_tinh = [cot for cot in data.columns if cot != COT_NHAN]
    cay_quyet_dinh = xay_dung_cay(data, thuoc_tinh)

    print("\n--- CÂY QUYẾT ĐỊNH ---")
    in_cay(cay_quyet_dinh)

    print("\n--- CÁC LUẬT ---")
    for so_thu_tu, luat in enumerate(lay_luat(cay_quyet_dinh), start=1):
        print(f"Rule {so_thu_tu}: {luat}")

    # Mẫu D15 trong bài thực hành: Overcast, Mild, Normal, gió nhẹ
    d15 = {"outlook": "overcast", "temp": "mild", "humidity": "normal", "windy": "false"}
    print("\nDự đoán cho mẫu D15:", du_doan(cay_quyet_dinh, d15))
