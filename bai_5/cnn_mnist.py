# -*- coding: utf-8 -*-
"""
BUỔI THỰC HÀNH 5: Mạng CNN nhận dạng chữ số viết tay với bộ dữ liệu MNIST.

Kiến trúc theo hình trong đề bài:
    Vào 28x28
    Tích chập: kernel 5x5, 16 bộ lọc, bước 1  -> 28x28x16
    Max-pool: kernel 2x2, bước 2              -> 14x14x16
    Tích chập: kernel 5x5, 32 bộ lọc, bước 1  -> 14x14x32
    Max-pool: kernel 2x2, bước 2              -> 7x7x32
    Kết nối đầy đủ: 128 nơ-ron
    Lớp ra: 10 nơ-ron ứng với 10 chữ số

Keras tự tải bộ dữ liệu MNIST trong lần chạy đầu tiên, cần có mạng Internet.

Cài thư viện: pip install tensorflow matplotlib
Chạy: python cnn_mnist.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import set_random_seed

THU_MUC = Path(__file__).parent
SO_EPOCH = 5


def tao_mo_hinh():
    """Tạo mạng CNN theo kiến trúc của đề bài."""
    # padding="same" giữ nguyên kích thước ảnh sau lớp tích chập
    return models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, kernel_size=5, strides=1, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=2, strides=2),
        layers.Conv2D(32, kernel_size=5, strides=1, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=2, strides=2),
        layers.Flatten(),                                   # 7x7x32 thành vector
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),             # xác suất của 10 chữ số
    ])


if __name__ == "__main__":
    set_random_seed(1)  # cố định số ngẫu nhiên để kết quả ổn định hơn

    # Đọc dữ liệu: 60000 ảnh train, 10000 ảnh test
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    print("Kích thước train:", X_train.shape)
    print("Kích thước test:", X_test.shape)

    # Đưa giá trị điểm ảnh về [0, 1] và thêm chiều kênh màu: 28x28 thành 28x28x1
    X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

    model = tao_mo_hinh()
    model.summary()

    # Nhãn là số nguyên 0 - 9 nên dùng sparse_categorical_crossentropy
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Lấy 10% tập train làm tập validation để theo dõi quá trình học
    lich_su = model.fit(X_train, y_train, epochs=SO_EPOCH, batch_size=128,
                        validation_split=0.1, verbose=2)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print("\nAccuracy trên tập test:", round(test_acc, 4))

    # Vẽ accuracy qua các epoch
    cac_epoch = range(1, SO_EPOCH + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(cac_epoch, lich_su.history["accuracy"], marker="o", color="#2a78d6", label="Train")
    plt.plot(cac_epoch, lich_su.history["val_accuracy"], marker="s", color="#eb6834", label="Validation")
    plt.xticks(cac_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Quá trình học của mạng CNN")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(THU_MUC / "qua_trinh_hoc_mnist.png", dpi=150)
    plt.close()
    print("Đã lưu biểu đồ: qua_trinh_hoc_mnist.png")

    # Dự đoán thử 10 ảnh đầu tiên của tập test
    du_doan = model.predict(X_test[:10], verbose=0).argmax(axis=1)
    print("\nNhãn dự đoán:", du_doan)
    print("Nhãn thật:   ", y_test[:10])
