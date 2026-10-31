# -*- coding: utf-8 -*-
"""
BUỔI THỰC HÀNH 5 - BỔ SUNG: Mạng CNN nhận dạng mặt người.

Kiến trúc theo hình trong đề bài (giống VGG16), ảnh vào 224x224:
    2 lớp tích chập 3x3, 64 bộ lọc   + max-pool /2   -> 112x112
    2 lớp tích chập 3x3, 128 bộ lọc  + max-pool /2   -> 56x56
    3 lớp tích chập 3x3, 256 bộ lọc  + max-pool /2   -> 28x28
    3 lớp tích chập 3x3, 512 bộ lọc  + max-pool /2   -> 14x14
    3 lớp tích chập 3x3, 512 bộ lọc  + max-pool /2   -> 7x7
    3 lớp kết nối đầy đủ 4096 nơ-ron
    Thêm 1 lớp ra softmax để chọn người trong dữ liệu

Số lớp giữ đúng theo hình. Bài thêm BatchNormalization sau mỗi lớp tích chập và
Dropout sau mỗi lớp kết nối đầy đủ, vì nếu không có chúng mạng không học được
trên bộ dữ liệu nhỏ này (loss đứng yên, accuracy chỉ bằng đoán ngẫu nhiên).

Dữ liệu là bộ Olivetti Faces có sẵn trong scikit-learn:
400 ảnh xám 64x64 của 40 người, mỗi người 10 ảnh.
Lần chạy đầu tiên scikit-learn tự tải dữ liệu, cần có mạng Internet.

LƯU Ý: Mạng này có hơn 100 triệu tham số nên huấn luyện rất nặng.
Nên chạy trên máy có GPU hoặc Google Colab, chạy bằng CPU sẽ rất lâu.
Bộ dữ liệu chỉ có 400 ảnh, quá ít so với mạng lớn như vậy,
nên độ chính xác có thể không cao nếu huấn luyện ít epoch.

Cài thư viện: pip install numpy scikit-learn tensorflow
Chạy: python cnn_nhan_dang_mat_nguoi.py
"""

import tensorflow as tf
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import set_random_seed

KICH_THUOC_ANH = 224
SO_EPOCH = 40
BATCH_SIZE = 16


def khoi_tich_chap(so_lop, so_bo_loc):
    """Một khối gồm nhiều lớp tích chập 3x3 và một lớp max-pool giảm một nửa kích thước."""
    khoi = []
    for _ in range(so_lop):
        khoi.append(layers.Conv2D(so_bo_loc, kernel_size=3, padding="same"))
        # momentum nhỏ vì dữ liệu ít, số bước huấn luyện không nhiều
        khoi.append(layers.BatchNormalization(momentum=0.9))
        khoi.append(layers.ReLU())
    khoi.append(layers.MaxPooling2D(pool_size=2, strides=2))
    return khoi


def tao_mo_hinh(kich_thuoc_anh, so_nguoi):
    """Tạo mạng theo kiến trúc của đề bài."""
    return models.Sequential([
        layers.Input(shape=(kich_thuoc_anh, kich_thuoc_anh, 3)),
        *khoi_tich_chap(2, 64),
        *khoi_tich_chap(2, 128),
        *khoi_tich_chap(3, 256),
        *khoi_tich_chap(3, 512),
        *khoi_tich_chap(3, 512),
        layers.Flatten(),
        layers.Dense(4096, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(4096, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(4096, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(so_nguoi, activation="softmax"),   # mỗi nơ-ron ứng với một người
    ])


def chuan_bi_anh(X, kich_thuoc_anh):
    """Phóng to ảnh về kích thước của mạng và chuyển ảnh xám thành 3 kênh màu."""
    anh = tf.image.resize(X[..., None], (kich_thuoc_anh, kich_thuoc_anh))
    anh = tf.repeat(anh, 3, axis=-1).numpy()
    return anh - 0.5  # đưa giá trị điểm ảnh về quanh 0 để mạng học dễ hơn


if __name__ == "__main__":
    set_random_seed(1)  # cố định số ngẫu nhiên để kết quả ổn định hơn

    # Đọc dữ liệu, giá trị điểm ảnh đã nằm trong khoảng [0, 1]
    du_lieu = fetch_olivetti_faces()
    X, y = du_lieu.images, du_lieu.target
    so_nguoi = len(set(y))
    print("Số ảnh:", len(X), ", số người:", so_nguoi)

    # Chia 80% train, 20% test, mỗi người có số ảnh như nhau ở mỗi tập
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=1
    )
    X_train = chuan_bi_anh(X_train, KICH_THUOC_ANH)
    X_test = chuan_bi_anh(X_test, KICH_THUOC_ANH)
    print("Kích thước train:", X_train.shape)

    model = tao_mo_hinh(KICH_THUOC_ANH, so_nguoi)
    model.summary()

    # Learning rate nhỏ để mạng sâu học ổn định
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # validation_data chỉ để theo dõi sau mỗi epoch, không dùng để huấn luyện
    model.fit(X_train, y_train, epochs=SO_EPOCH, batch_size=BATCH_SIZE,
              validation_data=(X_test, y_test), verbose=2)

    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    print("\nAccuracy trên tập test:", round(test_acc, 4))
