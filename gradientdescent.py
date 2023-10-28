##############################################################
# Simple Linear Regression with Gradient Descent from Scratch
##############################################################

# Cost function
# Minimum değeri bulmak
import pandas as pd


def cost_function(Y, b, w, X):
    m = len(Y)  # Gözlem Sayısı
    sse = 0  # Hata kareler toplamı
    for i in range(0, m):  # Bütün gözlem birimlerini gez.
        y_hat = b + w * X[i]  # Tahmin edilen y değerleri hesaplanır.
        y = Y[i]  # Gerçek y değerleri
        sse += (
            y_hat - y
        ) ** 2  # Gerçek değerler ve tahmin değerlerin farkının karesini al hepsini topla
    mse = sse / m  # Gözlem sayısına böl.
    return mse  # Sonucu dön


# Update Weights (Ağırlıkları güncelleme)
# update_weights
def update_weights(
    Y, b, w, X, learning_rate
):  # parametreler : bağımlı değişken , sabit , ağırlık , bağımsız değişken , öğrenme oranı
    m = len(Y)  # Gözlem sayısı

    b_deriv_sum = 0
    w_deriv_sum = 0

    for i in range(0, m):
        y_hat = b + w * X[i]  # Tahmin edilen değer
        y = y[i]  # Gerçek değer
        b_deriv_sum += y_hat - y  # Farklar toplam ifadesine eklendi.
        w_deriv_sum += (y_hat - y) * X[i]  # Ağırlık için toplam ifadesine eklendi.

    new_b = b - (learning_rate * 1 / m * b_deriv_sum)  # ortalamalar alındı
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)  # ortalamalar alındı

    return new_b, new_w


# Train Fonksiyonu 
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print(
        "Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(
            initial_b, initial_w, cost_function(Y, initial_b, initial_w, X)
        )
    )
    # Eğitim başlamadan öncekş bağımlı bağımsız değişken ve başlangıç ağırlıkları ile ilgili rapor yazdırmak için

    b = initial_b
    w = initial_w

    cost_history = []  # Hata hesaplama

    for i in range(num_iters):  # adım sayısı kadar
        b, w = update_weights(Y, b, w, X, learning_rate)  # Ağırlıkları güncellemek için
        mse = cost_function(Y, b, w, X)  # Yeni ağırlıklar için hata kontrolü
        cost_history.append(mse)  # Hata listeye eklendi

        if i % 100 == 0:  # Her yüz iterasyonda bir raporlama yapmak için
            print("iter={:d}   b={:.2f}  w={:.4f}  mse={:.4}".format(i, b, w, mse))

    print(
        "After {0} iterations b = {1}, w = {2}, mse = {3}".format(
            num_iters, b, w, cost_function(Y, b, w, X)
        )
    )
    return cost_history, b, w


df = pd.read_csv("data/advertising.csv")

X = df["radio"]
Y = df["sales"]

# parametre : modelin veriyi kullanarak bulduğu değerler
# hiperparametre : veri setinden bulunamaz kullanıcı tarafından ayarlanmalıdır.

# hyperparameters  (örnek olarak verilen değerler)
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 10000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)
