# -*- coding: utf-8 -*-
"""
PHIẾU BÀI 9 - BÀI 1: Cài đặt thuật toán K-means cho 7 điểm, K nhập từ bàn phím.

Các bước của K-means:
    1. Chọn K điểm làm tâm ban đầu (ở đây chọn K điểm đầu tiên).
    2. Gán mỗi điểm vào cụm có tâm gần nhất.
    3. Tính lại tâm của mỗi cụm bằng trung bình các điểm trong cụm.
    4. Lặp lại bước 2 và 3 cho đến khi các cụm không thay đổi.

Chạy: python phieu_9_bai_1_kmeans.py
"""

import math

# Dữ liệu của đề bài
DIEM = {
    "x1": (1, 1),
    "x2": (1, 3),
    "x3": (2, 2),
    "x4": (4, 4),
    "x5": (4, 5),
    "x6": (5, 4),
    "x7": (5, 5),
}


def tinh_tam(cum, diem):
    """Tâm của cụm là trung bình cộng tọa độ các điểm trong cụm."""
    toa_do = [diem[ten] for ten in cum]
    return tuple(sum(truc) / len(toa_do) for truc in zip(*toa_do))


def tinh_loi(cac_cum, diem):
    """Lỗi bình phương E = tổng bình phương khoảng cách từ mỗi điểm đến tâm cụm."""
    loi = 0
    for cum in cac_cum:
        tam = tinh_tam(cum, diem)
        loi += sum(math.dist(diem[ten], tam) ** 2 for ten in cum)
    return loi


def gan_vao_cum(cac_tam, diem):
    """Gán mỗi điểm vào cụm có tâm gần nhất."""
    cac_cum = [[] for _ in cac_tam]
    for ten, toa_do in diem.items():
        khoang_cach = [math.dist(toa_do, tam) for tam in cac_tam]
        cac_cum[khoang_cach.index(min(khoang_cach))].append(ten)
    return cac_cum


def in_cac_cum(cac_cum, diem):
    """In danh sách điểm và tâm của từng cụm."""
    for so_thu_tu, cum in enumerate(cac_cum, start=1):
        tam = tuple(round(x, 2) for x in tinh_tam(cum, diem))
        print(f"   C{so_thu_tu} = {cum}, tâm = {tam}")


def k_means(diem, k):
    """Thuật toán K-means, trả về danh sách các cụm."""
    # Bước 1: lấy K điểm đầu tiên làm tâm
    cac_tam = list(diem.values())[:k]
    cac_cum = None

    lan_lap = 0
    while True:
        lan_lap += 1

        # Bước 2: gán các điểm vào cụm gần nhất
        cum_moi = gan_vao_cum(cac_tam, diem)

        # Các cụm không đổi nghĩa là thuật toán đã hội tụ
        if cum_moi == cac_cum:
            print(f"Lần lặp {lan_lap}: các cụm không đổi, thuật toán dừng.")
            return cac_cum
        cac_cum = cum_moi

        print(f"Lần lặp {lan_lap}:")
        in_cac_cum(cac_cum, diem)

        # Bước 3: tính lại tâm. Cụm rỗng thì giữ nguyên tâm cũ
        cac_tam = [tinh_tam(cum, diem) if cum else tam
                   for cum, tam in zip(cac_cum, cac_tam)]


def nhap_k(so_diem):
    """Nhập số cụm K từ bàn phím, nhập sai thì yêu cầu nhập lại."""
    while True:
        try:
            k = int(input(f"Nhập số cụm K (1 - {so_diem}): "))
        except ValueError:
            print("Vui lòng nhập một số nguyên.")
            continue

        if 1 <= k <= so_diem:
            return k
        print("K nằm ngoài khoảng cho phép.")


if __name__ == "__main__":
    k = nhap_k(len(DIEM))
    ket_qua = k_means(DIEM, k)

    print(f"\nKết quả với K = {k}:")
    in_cac_cum(ket_qua, DIEM)
    print(f"   Lỗi bình phương E = {tinh_loi(ket_qua, DIEM):.2f}")
