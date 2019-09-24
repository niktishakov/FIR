import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np

import math
import random

random.seed()
pi = np.pi
cos = np.cos
sin = np.sin
ceil = math.ceil
floor = math.floor
fabs = math.fabs
sqrt = math.sqrt
lg = math.log10


def polyharmonic1(t):
    return (10 * sin(0.5 * pi * t) +
            10 * sin(0.3 * pi * t) +
            10 * sin(0.1 * pi * t))


def low_pass_filter(n, omega_c):
    if n == 0:
        return omega_c / pi
    else:
        return sin(omega_c * n) / (pi * n)


def high_pass_filter(n, omega_c):
    if n == 0:
        return 1 - omega_c / pi
    else:
        return - sin(omega_c * n) / (pi * n)


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


def_filters = {
    'low': low_pass_filter,
    'high': high_pass_filter,
    'band': bandpass_filter,
    'barrage': barrage_filter
}


def kaiser_window(omega_s, omega_p, omega_a, A_o, type="low"):
    # omega_s - частота дискретизации
    # omega_p - полоса пропускания
    # omega_a - полоса заграждения
    # A_o - параметр, определяющий A_a, A_p
    H = []  # Импульсная характеристика фильтра
    H_d = []  # Идеальная импульсная характеристика
    W = []  # Весовая функция

    try:
        filter = def_filters[type]
    except KeyError as e:
        raise ValueError('Undefined filter type: {}'.format(e.args[0]))

    delta1 = pow(10, -0.05 * A_o)
    delta2 = (pow(10, 0.05 * A_o) - 1) / (pow(10, 0.05 * A_o) + 1)
    delta = min(delta1, delta2)
    A_a = -20 * lg(delta)  # Минимальное затухание в полосе задерживания

    # Максимально допустимая пульсация в полосе пропускания
    # A_p = 20 * lg((1 + delta) / (1 - delta))

    omega_c = (omega_p + omega_a) / 2
    D = 0.9222 if (A_a <= 21) else (A_a - 7.95) / 14.36
    alpha = (0.0 if A_a <= 21 else
             0.5842 * pow(A_a - 21, 0.4) + 0.07886 * (A_a - 21) if A_a <= 50 else
             0.1102 * (A_a - 8.7))

    size = ceil(omega_s * D / (omega_a - omega_p) * 0.5) * 2 + 1  # Длина фильтра
    start, fin = -floor(size / 2), floor(size / 2)
    print("size, start, fin:", size, start, fin)

    for n in range(start, fin + 1):
        H_d.append(filter(n, omega_c))
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


def hamming_window(size, alpha, Fc, type="low"):
    if alpha < 0 or alpha > 1:
        return False

    H = []  # Импульсная характеристика фильтра
    H_d = []  # Идеальная импульсная характеристика
    W = []  # Весовая функция

    try:
        filter = def_filters[type]
    except KeyError as e:
        raise ValueError('Undefined filter type: {}'.format(e.args[0]))

    start, fin = -floor(size / 2), floor(size / 2)

    print("size, start, fin:", size, start, fin)

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


H1 = kaiser_window(1, 0.15, 0.3, 30, "low")
H2 = hamming_window(5, 0.54, 0.001, "low")
t = list(range(200))
input = []
output1 = []
output2 = []

for i in t:
    input.append(polyharmonic1(i))
    # if i >= len(H1):
    output1.append(response(input, H1))
    # if i >= len(H2):
    output2.append(response(input, H2))


def plotGraphic(func, delta_y, name="Function"):
    t = list(range(len(func)))
    plt.xlim(-2, len(func))
    plt.ylim(-delta_y, delta_y)
    plt.plot(t, func)
    plt.title(name)
    plt.show()


def plotFFTGraphic(N, T, y, name="FFT Function"):
    x = np.linspace(0.0, N * T, N)
    yf = fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    plt.grid()
    plt.title(name)
    plt.show()


plotGraphic(input, 35, "Input")
plotGraphic(output1, 35, "Output After Kaiser")
plotGraphic(output2, 35, "Output After Hamming")

plotFFTGraphic(len(input), 1, input, "FFT Input")
plotFFTGraphic(len(output1), 1, output1, "FFT Output After Kaiser")
plotFFTGraphic(len(output2), 1, output2, "FFT Output After Hamming")
