# -*- coding: utf-8 -*-
"""
BÀI 1: Cài đặt thuật toán Apriori để tìm tập phổ biến, sau đó sinh luật kết hợp
với độ hỗ trợ (minsup) và độ tin cậy (minconf) nhập từ bàn phím.

Các công thức dùng trong bài:
    support(X)      = số giao dịch chứa X / tổng số giao dịch
    confidence(X→Y) = số giao dịch chứa cả X và Y / số giao dịch chứa X
    lift(X→Y)       = confidence(X→Y) / support(Y)

Ý tưởng của Apriori:
    - Một tập đã KHÔNG phổ biến thì mọi tập lớn hơn chứa nó cũng không phổ biến.
    - Vì vậy ta đi từ tập 1 phần tử, lên 2 phần tử, 3 phần tử...
      Mỗi vòng chỉ xét những ứng viên được ghép từ các tập phổ biến của vòng trước.

Chạy: python bai_1_apriori.py
"""

from itertools import combinations

# Cơ sở dữ liệu của đề bài, mỗi dòng là một giao dịch (TID từ 1 đến 6)
GIAO_DICH = [
    {"A", "B", "D", "E"},
    {"B", "C", "E"},
    {"A", "B", "D", "E"},
    {"A", "B", "C", "E"},
    {"A", "B", "C", "D", "E"},
    {"B", "C", "D", "E"},
]


def dem_so_lan_xuat_hien(ung_vien, giao_dich):
    """Đếm xem mỗi tập ứng viên nằm trong bao nhiêu giao dịch."""
    so_lan = {}
    for tap in ung_vien:
        so_lan[tap] = 0
        for gd in giao_dich:
            if tap <= gd:  # tap <= gd nghĩa là "tap là tập con của gd"
                so_lan[tap] += 1
    return so_lan


def tao_ung_vien(tap_pho_bien_truoc, k):
    """Tạo các ứng viên k phần tử từ các tập phổ biến (k-1) phần tử."""
    ung_vien = set()
    danh_sach = list(tap_pho_bien_truoc)

    # Bước ghép: hợp từng cặp tập, giữ lại những kết quả có đúng k phần tử
    for i in range(len(danh_sach)):
        for j in range(i + 1, len(danh_sach)):
            tap_moi = danh_sach[i] | danh_sach[j]
            if len(tap_moi) != k:
                continue

            # Bước tỉa: mọi tập con (k-1) phần tử của ứng viên đều phải phổ biến
            hop_le = True
            for tap_con in combinations(tap_moi, k - 1):
                if frozenset(tap_con) not in tap_pho_bien_truoc:
                    hop_le = False
                    break

            if hop_le:
                ung_vien.add(tap_moi)

    return ung_vien


def apriori(giao_dich, minsup):
    """
    Tìm tất cả tập phổ biến.
    Trả về dict: {tập phổ biến (frozenset): số giao dịch chứa tập đó}
    """
    so_giao_dich = len(giao_dich)
    ket_qua = {}

    # Vòng đầu tiên: ứng viên là từng mặt hàng riêng lẻ
    tat_ca_mat_hang = set()
    for gd in giao_dich:
        tat_ca_mat_hang = tat_ca_mat_hang | gd
    ung_vien = {frozenset([mat_hang]) for mat_hang in tat_ca_mat_hang}

    k = 1
    while ung_vien:
        so_lan = dem_so_lan_xuat_hien(ung_vien, giao_dich)

        # Chỉ giữ lại các tập đạt độ hỗ trợ tối thiểu
        tap_pho_bien = {}
        for tap, dem in so_lan.items():
            if dem / so_giao_dich >= minsup:
                tap_pho_bien[tap] = dem

        ket_qua.update(tap_pho_bien)

        # Sang vòng sau: tạo ứng viên có nhiều hơn 1 phần tử
        k += 1
        ung_vien = tao_ung_vien(set(tap_pho_bien), k)

    return ket_qua


def sinh_luat(tap_pho_bien, so_giao_dich, minconf):
    """
    Sinh luật kết hợp X → Y từ các tập phổ biến.
    Với mỗi tập phổ biến có từ 2 phần tử, ta thử tách nó thành 2 phần X và Y.
    """
    danh_sach_luat = []

    for tap, dem_tap in tap_pho_bien.items():
        if len(tap) < 2:
            continue  # tập 1 phần tử thì không tách được thành luật

        # Thử mọi tập con X có kích thước từ 1 đến len(tap) - 1
        for kich_thuoc in range(1, len(tap)):
            for ve_trai in combinations(tap, kich_thuoc):
                x = frozenset(ve_trai)
                y = tap - x

                confidence = dem_tap / tap_pho_bien[x]
                if confidence >= minconf:
                    support = dem_tap / so_giao_dich
                    lift = confidence / (tap_pho_bien[y] / so_giao_dich)
                    danh_sach_luat.append((x, y, support, confidence, lift))

    # Sắp xếp luật theo lift giảm dần, bằng nhau thì xét confidence
    danh_sach_luat.sort(key=lambda luat: (-luat[4], -luat[3], sorted(luat[0]), sorted(luat[1])))
    return danh_sach_luat


def viet_tap(tap):
    """Đổi tập thành chuỗi dễ đọc, ví dụ {A, B}."""
    return "{" + ", ".join(sorted(tap)) + "}"


def in_tap_pho_bien(tap_pho_bien, so_giao_dich):
    """In các tập phổ biến, nhóm theo số phần tử."""
    print(f"\n--- TẬP PHỔ BIẾN ({len(tap_pho_bien)} tập) ---")
    if not tap_pho_bien:
        print("Không có tập phổ biến nào.")
        return

    kich_thuoc_lon_nhat = max(len(tap) for tap in tap_pho_bien)
    for k in range(1, kich_thuoc_lon_nhat + 1):
        print(f"Tập {k} phần tử:")
        cac_tap = [tap for tap in tap_pho_bien if len(tap) == k]
        for tap in sorted(cac_tap, key=sorted):
            dem = tap_pho_bien[tap]
            print(f"   {viet_tap(tap)}: xuất hiện {dem} lần, support = {dem / so_giao_dich:.2f}")


def in_luat(danh_sach_luat):
    """In các luật kết hợp."""
    print(f"\n--- LUẬT KẾT HỢP ({len(danh_sach_luat)} luật) ---")
    if not danh_sach_luat:
        print("Không có luật nào thỏa mãn ngưỡng đã cho.")
        return

    for x, y, support, confidence, lift in danh_sach_luat:
        print(f"   {viet_tap(x)} -> {viet_tap(y)}: "
              f"support = {support:.2f}, confidence = {confidence:.2f}, lift = {lift:.2f}")


def nhap_nguong(thong_bao):
    """Nhập một ngưỡng từ bàn phím, nhập sai thì yêu cầu nhập lại."""
    while True:
        try:
            gia_tri = float(input(thong_bao))
        except ValueError:
            print("Vui lòng nhập một số, ví dụ 0.5")
            continue

        if 0 < gia_tri <= 1:
            return gia_tri
        print("Giá trị phải lớn hơn 0 và không vượt quá 1, ví dụ 0.5 tương ứng 50%")


if __name__ == "__main__":
    minsup = nhap_nguong("Nhập độ hỗ trợ tối thiểu minsup (0 - 1): ")
    minconf = nhap_nguong("Nhập độ tin cậy tối thiểu minconf (0 - 1): ")

    tap_pho_bien = apriori(GIAO_DICH, minsup)
    in_tap_pho_bien(tap_pho_bien, len(GIAO_DICH))

    luat = sinh_luat(tap_pho_bien, len(GIAO_DICH), minconf)
    in_luat(luat)
