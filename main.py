import matplotlib.pyplot as plt
import numpy as np
import math
import random

random.seed()
omega = 2
pi = math.pi
cos = math.cos
sin = math.sin
ceil = math.ceil
floor = math.floor
fabs = math.fabs
sqrt = math.sqrt


def Polyharmonic1(t):
    return (sin(2 * pi * 0.05 * t) +
            50 * sin(2 * pi * 0.1 * t) +
            10 * sin(2 * pi * 0.5 * t) +
            omega * random.randint(1, 500))


def Polyharmonic2(t):
    return sin(t) * 3


def low_pass_filter(n, Fc):
    # if fabs(i) >= fabs(floor(size / 2)):
    #     return 0
    if n == 0:
        return 2 * pi * Fc
    else:
        return sin(2 * pi * Fc * n) / (pi * n)


def high_pass_filter(n, Fc):
    if n == 0:
        return 1 - 2 * pi * Fc
    else:
        return -sin(2 * pi * Fc * n) / (pi * n)


def bandpass_filter(n, f1, f2, w1, w2):
    if n == 0:
        return 2 * (f2 - f1)
    else:
        return sin(w2 * n) / (pi * n) - sin(w1 * n) / (pi * n)


def barrage_filter(n, f1, f2, w1, w2):
    if n == 0:
        return 1 - 2 * (f2 - f1)
    else:
        return sin(w1 * n) / (pi * n) - sin(w2 * n) / (pi * n)


def kaiser_window(omega_s, omega_p, omega_a, A_a, is_high):
    H = []  # Импульсная характеристика фильтра
    H_d = []  # Идеальная импульсная характеристика
    W = []  # Весовая функция

    filter = high_pass_filter if is_high else low_pass_filter
    Fc = (omega_p + omega_a) / (2 * omega_s)  # Частота среза
    D = 0.9222 if (A_a <= 21) else (A_a - 7.95) / 14.36
    alpha = (0.0 if A_a <= 21 else
             0.5842 * pow(A_a - 21, 0.4) + 0.07886 * (A_a - 21) if A_a <= 50 else
             0.1102 * (A_a - 8.7))

    size = ceil(omega_s * D / (omega_a - omega_p) * 0.5) * 2 + 1  # Длина фильтра
    start, fin = -floor(size / 2), floor(size / 2)
    print("start, fin:", start, fin)

    for n in range(start, fin + 1):
        H_d.append(filter(n, Fc))
        # весовая функция Кайзера
        beta = alpha * sqrt(1 - pow(2 * n / (size - 1), 2))
        # i0 - модифицированная функция Бесселя 1-го рода 0-го порядка
        W.append(np.i0(beta) / np.i0(alpha))
        # Преобразование коэффициентов по принципу свертки
        H.append(H_d[-1] * W[-1])

    # Нормировка импульсной характеристики
    sum = 0.0
    for i in range(size):
        sum += H[i]
    for i in range(size):
        H[i] /= sum
    return H


def hamming_window(alpha, is_high):
    if alpha < 0 or alpha > 1:
        return False

    H = []  # Импульсная характеристика фильтра
    H_d = []  # Идеальная импульсная характеристика
    W = []  # Весовая функция

    filter = high_pass_filter if is_high else low_pass_filter
    # Fc = (omega_p + omega_a) / (2 * omega_s)
    Fc = 0.1245  # Частота среза
    # size = ceil(omega_s * D / (omega_a - omega_p) * 0.5) * 2 + 1  # Длина фильтра
    size = 21
    start, fin = -floor(size / 2), floor(size / 2)

    print("start, fin:", start, fin)

    for n in range(start, fin + 1):
        H_d.append(filter(n, Fc))
        # весовая функция Кайзера
        W.append(alpha - (1 - alpha) * cos(2 * pi * n / (size - 1)))
        # Преобразование коэффициентов по принципу свертки
        H.append(H_d[-1] * W[-1])

    # Нормировка импульсной характеристики
    sum = 0.0
    for i in range(size):
        sum += H[i]
    for i in range(size):
        H[i] /= sum
    return H


def response(input, H):
    if len(input) < 1:
        return 0

    output = 0.0
    inpSize = len(input)
    firSize = len(H)

    for i in range(firSize):
        if inpSize - 1 - i >= 0:
            output += H[i] * input[-1 - i]
    return output


H1 = kaiser_window(50, 20, 50, 30, False)
H2 = hamming_window(0.54, False)
t = list(range(100))
input = []
output1 = []
output2 = []

for i in t:
    input.append(Polyharmonic1(i))
    # input.append(Polyharmonic2(i))
    output1.append(response(input, H1))
    output2.append(response(input, H2))

deltaY = 1500
deltaX = 10
plt.xlim(0, len(t))
plt.ylim(-deltaY, deltaY)
plt.plot(t, input)
plt.show()

plt.xlim(0, len(t))
plt.ylim(-deltaY, deltaY)
plt.plot(t, output1)
plt.show()

plt.xlim(0, len(t))
plt.ylim(-deltaY, deltaY)
plt.plot(t, output2)
plt.show()
