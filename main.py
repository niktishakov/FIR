import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np

import math
import random
import threading

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
        return omega_c[0] / pi
    else:
        return sin(omega_c[0] * n) / (pi * n)


def high_pass_filter(n, omega_c):
    if n == 0:
        return 1 - omega_c[0] / pi
    else:
        return - sin(omega_c[0] * n) / (pi * n)


def bandpass_filter(n, omega_c):
    if n == 0:
        return (omega_c[1] - omega_c[0]) / pi
    else:
        return sin(omega_c[1] * n) / (pi * n) - sin(omega_c[0] * n) / (pi * n)


def barrage_filter(n, omega_c):
    if n == 0:
        return 1 - (omega_c[1] - omega_c[0]) / pi
    else:
        return sin(omega_c[0] * n) / (pi * n) - sin(omega_c[1] * n) / (pi * n)


def_filters = {
    'low': low_pass_filter,
    'high': high_pass_filter,
    'band': bandpass_filter,
    'barrage': barrage_filter
}


def kaiser_window(type):
    N = 1 if type == "low" or type == "high" else 2

    omega_c = []
    order = 1
    print("Kaiser Window:")
    for i in range(N):
        print(i + 1, "| omega_s:")
        omega_s = float(input())  # omega_s - частота дискретизации
        print(i + 1, "| omega_p:")
        omega_p = float(input())  # omega_p - полоса пропускания
        print(i + 1, "| omega_a:")
        omega_a = float(input())  # omega_a - полоса заграждения
        print(i + 1, "| A_o:")
        A_o = float(input())  # A_o - параметр, определяющий A_a, A_p

        delta1 = pow(10, -0.05 * A_o)
        delta2 = (pow(10, 0.05 * A_o) - 1) / (pow(10, 0.05 * A_o) + 1)
        delta = min(delta1, delta2)
        A_a = -20 * lg(delta)  # Минимальное затухание в полосе задерживания

        # Максимально допустимая пульсация в полосе пропускания
        # A_p = 20 * lg((1 + delta) / (1 - delta))

        omega_c.append((omega_p + omega_a) / 2)
        D = 0.9222 if (A_a <= 21) else (A_a - 7.95) / 14.36
        alpha = (0.0 if A_a <= 21 else
                 0.5842 * pow(A_a - 21, 0.4) + 0.07886 * (A_a - 21) if A_a <= 50 else
                 0.1102 * (A_a - 8.7))

        order = max(order, ceil(omega_s * D / (omega_a - omega_p) * 0.5) * 2 + 1)  # Длина фильтра

    n = yield order, omega_c

    while True:
        beta = alpha * sqrt(1 - pow(2 * n / (order - 1), 2))
        n = yield np.i0(beta) / np.i0(alpha)


def hamming_window(type):
    alpha = 0.54
    N = 1 if type == "low" or type == "high" else 2
    omega_c = []

    print("Hamming Window:")
    for i in range(N):
        print(i, ": omega_c")
        omega_c.append(float(input()))

    print("order:")
    order = int(input())

    n = yield order, omega_c
    while True:
        n = yield alpha - (1 - alpha) * cos(2 * pi * n / (order - 1))


def_windows = {
    'kaiser': kaiser_window,
    'hamming': hamming_window
}


def getFir(w_type, f_type):
    H = []  # Импульсная характеристика фильтра
    H_d = []  # Идеальная импульсная характеристика
    W = []  # Весовая функция

    try:
        filter = def_filters[f_type]
        window = def_windows[w_type]
    except KeyError as e:
        raise ValueError('Undefined filter type: {}'.format(e.args[0]))

    win_gen = window(f_type)  # Init a generator for selected window
    order, omega_c = win_gen.send(None)  # First call: pre-calculation
    start, fin = -floor(order / 2), floor(order / 2)

    for n in range(start, fin + 1):
        H_d.append(filter(n, omega_c))
        # весовая функция Кайзера
        W.append(win_gen.send(n))
        # Преобразование коэффициентов по принципу свертки
        H.append(H_d[-1] * W[-1])

    win_gen.close()

    # Нормировка импульсной характеристики
    sum = 0.0
    for i in range(order):
        sum += H[i]
    for i in range(order):
        H[i] /= sum
    return H


def worker(output, H, input):
    output += H * input


def response(input, H, isParallel=False):
    if len(input) < 1:
        return 0

    output = 0.0
    inpSize = len(input)
    firSize = len(H)

    for i in range(firSize):
        if inpSize - 1 - i >= 0:
            if isParallel:
                t = threading.Thread(target=worker, args=(output, H[i], input[-1 - i]))
                t.start()
            else:
                output += H[i] * input[-1 - i]

    if isParallel:
        main_thread = threading.currentThread()
        for t in threading.enumerate():
            if t is main_thread:
                continue
            t.join()
    return output


def plotGraphic(func, delta_y, name="Function"):
    t = list(range(len(func)))
    plt.xlim(-2, len(func))
    plt.ylim(-delta_y, delta_y)
    plt.plot(t, func)
    plt.title(name)
    plt.show()


def plotFFTGraphic(y, T, name="FFT Function"):
    N = len(y)
    yf = fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    plt.grid()
    plt.title(name)
    plt.show()


def sequence(t, H1, H2):
    input = []
    output1 = [0.] * t
    output2 = [0.] * t

    for i in range(t):
        input.append(polyharmonic1(i))
        output1[i] = response(input, H1)
        output2[i] = response(input, H2)

    return input, output1, output2


def parallel(t, H1, H2):
    input = []
    output1 = []
    output2 = []

    for i in range(t):
        input.append(polyharmonic1(i))
        output1.append(response(input, H1))
        output2.append(response(input, H2))

    return input, output1, output2


def main():
    H1 = getFir("kaiser", "band")
    H2 = getFir("hamming", "band")

    # input, output1, output2 = sequence(200, H1, H2)
    input, output1, output2 = parallel(200, H1, H2)

    plotGraphic(input, 35, "Input")
    plotGraphic(output1, 35, "Output After Kaiser")
    plotGraphic(output2, 35, "Output After Hamming")

    plotFFTGraphic(input, 1, "FFT Input")
    plotFFTGraphic(output1, 1, "FFT Output After Kaiser")
    plotFFTGraphic(output2, 1, "FFT Output After Hamming")


if __name__ == "__main__":
    main()
