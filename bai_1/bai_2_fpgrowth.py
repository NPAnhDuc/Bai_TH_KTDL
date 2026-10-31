# -*- coding: utf-8 -*-
"""
BÀI 2: Cài đặt thuật toán FP-Growth để tìm tập phổ biến, sau đó sinh luật kết hợp
với độ hỗ trợ (minsup) và độ tin cậy (minconf) nhập từ bàn phím.
Cơ sở dữ liệu giống bài 1.

Các bước của FP-Growth:
    1. Đếm số lần xuất hiện của từng mặt hàng, bỏ các mặt hàng không phổ biến.
    2. Sắp xếp mặt hàng trong mỗi giao dịch theo số lần xuất hiện giảm dần.
    3. Lần lượt đưa từng giao dịch vào cây FP. Các giao dịch có chung phần đầu
       sẽ dùng chung nhánh, nhờ đó dữ liệu được nén lại.
    4. Với từng mặt hàng, lấy các đường đi từ gốc tới mặt hàng đó
       (gọi là cơ sở mẫu có điều kiện), dựng một cây FP nhỏ từ các đường đi này
       rồi lặp lại việc khai thác trên cây nhỏ.

Ưu điểm so với Apriori: không phải sinh ứng viên và quét lại dữ liệu nhiều lần.

Phần sinh luật giống hệt bài 1 nên dùng lại từ file bai_1_apriori.py.

Chạy: python bai_2_fpgrowth.py
"""

from bai_1_apriori import GIAO_DICH, in_luat, in_tap_pho_bien, nhap_nguong, sinh_luat


class Nut:
    """Một nút của cây FP."""

    def __init__(self, ten, cha):
        self.ten = ten        # tên mặt hàng
        self.dem = 0          # số giao dịch đi qua nút này
        self.cha = cha        # nút cha, dùng để đi ngược lên gốc
        self.con = {}         # các nút con: {tên mặt hàng: nút}


def dung_cay(danh_sach_mau, so_giao_dich, minsup):
    """
    Dựng cây FP.

    danh_sach_mau: danh sách (các mặt hàng, số lần), ví dụ (["A", "B"], 2).
                   Với dữ liệu gốc, số lần của mỗi giao dịch là 1.

    Trả về:
        goc: nút gốc của cây
        bang_nut: {mặt hàng: danh sách các nút của mặt hàng đó trong cây}
        so_lan: {mặt hàng phổ biến: số lần xuất hiện}
    """
    # Bước 1: đếm số lần xuất hiện của từng mặt hàng
    so_lan = {}
    for mat_hang_list, dem in danh_sach_mau:
        for mat_hang in mat_hang_list:
            so_lan[mat_hang] = so_lan.get(mat_hang, 0) + dem

    # Chỉ giữ các mặt hàng đạt độ hỗ trợ tối thiểu
    so_lan = {m: d for m, d in so_lan.items() if d / so_giao_dich >= minsup}

    goc = Nut(None, None)
    bang_nut = {mat_hang: [] for mat_hang in so_lan}

    for mat_hang_list, dem in danh_sach_mau:
        # Bước 2: bỏ mặt hàng không phổ biến, sắp xếp theo số lần giảm dần
        # (bằng nhau thì xếp theo tên để kết quả luôn giống nhau mỗi lần chạy)
        da_loc = [m for m in mat_hang_list if m in so_lan]
        da_loc.sort(key=lambda m: (-so_lan[m], m))

        # Bước 3: đưa giao dịch vào cây, bắt đầu từ gốc
        nut_hien_tai = goc
        for mat_hang in da_loc:
            if mat_hang not in nut_hien_tai.con:
                nut_moi = Nut(mat_hang, nut_hien_tai)
                nut_hien_tai.con[mat_hang] = nut_moi
                bang_nut[mat_hang].append(nut_moi)
            nut_hien_tai = nut_hien_tai.con[mat_hang]
            nut_hien_tai.dem += dem

    return goc, bang_nut, so_lan


def lay_co_so_mau(cac_nut):
    """Lấy các đường đi từ gốc tới từng nút (cơ sở mẫu có điều kiện)."""
    co_so_mau = []
    for nut in cac_nut:
        duong_di = []
        cha = nut.cha
        while cha.ten is not None:  # chỉ nút gốc mới có tên là None
            duong_di.append(cha.ten)
            cha = cha.cha
        if duong_di:
            co_so_mau.append((duong_di, nut.dem))
    return co_so_mau


def khai_thac(danh_sach_mau, hau_to, so_giao_dich, minsup, ket_qua):
    """
    Khai thác đệ quy.
    hau_to là tập mặt hàng đã chọn, ban đầu là tập rỗng.
    """
    _, bang_nut, so_lan = dung_cay(danh_sach_mau, so_giao_dich, minsup)

    for mat_hang, dem in so_lan.items():
        # Mỗi mặt hàng phổ biến ghép với hậu tố tạo thành một tập phổ biến mới
        tap_moi = hau_to | {mat_hang}
        ket_qua[tap_moi] = dem

        # Khai thác tiếp trên cơ sở mẫu có điều kiện của mặt hàng này
        co_so_mau = lay_co_so_mau(bang_nut[mat_hang])
        khai_thac(co_so_mau, tap_moi, so_giao_dich, minsup, ket_qua)


def fp_growth(giao_dich, minsup):
    """
    Tìm tất cả tập phổ biến.
    Trả về dict: {tập phổ biến (frozenset): số giao dịch chứa tập đó}
    """
    # Ban đầu mỗi giao dịch được tính 1 lần
    danh_sach_mau = [(list(gd), 1) for gd in giao_dich]

    ket_qua = {}
    khai_thac(danh_sach_mau, frozenset(), len(giao_dich), minsup, ket_qua)
    return ket_qua


def in_cay(nut, muc=0):
    """In cây FP ra màn hình, nút con thụt vào so với nút cha."""
    for con in nut.con.values():
        print("   " + "    " * muc + f"{con.ten}:{con.dem}")
        in_cay(con, muc + 1)


def giai_bai(giao_dich, minsup, minconf):
    """Chạy toàn bộ các bước và in kết quả. Dùng chung cho bài 2, 3 và 4."""
    so_giao_dich = len(giao_dich)

    # In cây FP để xem cấu trúc dữ liệu sau khi nén
    danh_sach_mau = [(list(gd), 1) for gd in giao_dich]
    goc, _, _ = dung_cay(danh_sach_mau, so_giao_dich, minsup)
    print("\n--- CÂY FP (mặt hàng:số lần) ---")
    print("   (gốc)")
    in_cay(goc)

    tap_pho_bien = fp_growth(giao_dich, minsup)
    in_tap_pho_bien(tap_pho_bien, so_giao_dich)

    luat = sinh_luat(tap_pho_bien, so_giao_dich, minconf)
    in_luat(luat)


if __name__ == "__main__":
    minsup = nhap_nguong("Nhập độ hỗ trợ tối thiểu minsup (0 - 1): ")
    minconf = nhap_nguong("Nhập độ tin cậy tối thiểu minconf (0 - 1): ")
    giai_bai(GIAO_DICH, minsup, minconf)
