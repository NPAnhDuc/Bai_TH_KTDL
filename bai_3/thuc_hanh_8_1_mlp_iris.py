# -*- coding: utf-8 -*-
"""
THỰC HÀNH 8-1: Phân loại hoa iris bằng mạng nơ-ron.

Làm 2 cách giống bài thực hành:
    Cách 1: MLPClassifier của scikit-learn, 2 lớp ẩn 3 - 3
    Cách 2: Mạng truyền thẳng bằng Keras, 3 lớp ẩn 32 - 64 - 128

Các bước:
    1. Nạp dữ liệu iris
    2. Chia dữ liệu thành tập train và test
    3. Mã hóa nhãn về dạng one-hot (dùng cho Keras)
    4. Tạo và huấn luyện mô hình
    5. Đánh giá mô hình trên tập test

Cài thư viện: pip install numpy scikit-learn tensorflow
Chạy: python thuc_hanh_8_1_mlp_iris.py
"""

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tensorflow.keras import layers, models
from tensorflow.keras.utils import set_random_seed, to_categorical

# ----- 1. Nạp dữ liệu -----
iris = datasets.load_iris()
X = iris.data      # 4 thuộc tính của mỗi bông hoa
y = iris.target    # nhãn 0, 1, 2 ứng với 3 loài hoa
print("5 mẫu đầu tiên:")
print(X[:5, :])
print("Nhãn:", y[:5])

# ----- 2. Chia dữ liệu: 80% train, 20% test -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1
)
print("\nKích thước train:", X_train.shape)
print("Kích thước test:", X_test.shape)

# ----- 3. Mã hóa one-hot: nhãn 2 thành [0, 0, 1] -----
train_labels = to_categorical(y_train)
test_labels = to_categorical(y_test)
print("\n5 nhãn đầu sau khi mã hóa:")
print(train_labels[:5])

# ----- 4 và 5. Cách 1: MLPClassifier của scikit-learn -----
# max_iter=2000 để mô hình có đủ số vòng lặp để hội tụ
clf = MLPClassifier(solver="adam", alpha=1e-5, hidden_layer_sizes=(3, 3),
                    activation="relu", max_iter=2000, random_state=1)
clf.fit(X_train, y_train)

prediction = clf.predict(X_test)
print("\nNhãn dự đoán:", prediction)
print("Nhãn thật:   ", y_test)
print("Accuracy của MLPClassifier:", metrics.accuracy_score(y_test, prediction))

# ----- 4 và 5. Cách 2: mạng truyền thẳng bằng Keras -----
set_random_seed(1)  # cố định số ngẫu nhiên để các lần chạy cho kết quả giống nhau

network = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(3, activation="softmax"),  # 3 nơ-ron ứng với 3 loài hoa
])
network.summary()

# Khai báo hàm tối ưu, hàm mất mát và thước đo
network.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

network.fit(X_train, train_labels, epochs=20, batch_size=40, verbose=2)

test_loss, test_acc = network.evaluate(X_test, test_labels, verbose=0)
print("\nTest Accuracy:", test_acc)
print("Test Loss:", test_loss)
