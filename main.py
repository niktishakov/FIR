import multiprocessing
from sys import stdin
from multiprocessing import Pool, Array, Process
import mymodule

import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np

import time
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
barrier = None


def polyharmonic(t):
    return (10 * sin(0.9 * pi * t) +
            10 * sin(0.6 * pi * t) +
            10 * sin(0.4 * pi * t) +
            10 * sin(0.1 * pi * t))


def delta_function(t):
    return 1 if t == 0 else 0


def low_pass_filter(n, omega_c):
    if n == 0:
        return omega_c[0] / pi
    else:
        return sin(omega_c[0] * n) / (pi * n)


def bandpass_filter(n, omega_c):
    if n == 0:
        return (omega_c[1] - omega_c[0]) / pi
    else:
        return (sin(omega_c[1] * n) - sin(omega_c[0] * n)) / (pi * n)


def_filters = {
    'low': low_pass_filter,
    'band': bandpass_filter,
}


def kaiser_window(f_type, args):
    d_omega, omega_1, omega_2 = args[0] * pi, args[1] * pi, args[2] * pi if f_type == "band" else 0
    omega_s = pi  # частота дискретизации
    A_o = 0.05  # определяет A_a, A_p

    delta1 = pow(10, -0.05 * A_o)
    delta2 = (pow(10, 0.05 * A_o) - 1) / (pow(10, 0.05 * A_o) + 1)
    delta = min(delta1, delta2)
    A_a = -20 * lg(delta)  # Минимальное затухание в полосе задерживания

    D = 0.9222 if (A_a <= 21) else (A_a - 7.95) / 14.36
    alpha = (0.0 if A_a <= 21 else
             0.5842 * pow(A_a - 21, 0.4) + 0.07886 * (A_a - 21) if A_a <= 50 else
             0.1102 * (A_a - 8.7))

    order = 1  # Длина фильтра
    omega_c = []  # Нормированные частоты срезов

    if f_type == "low":
        omega_p = omega_1  # omega_p - полоса пропускания
        omega_a = omega_1 + d_omega  # omega_a - полоса заграждения
        omega_c.append((omega_p + omega_a) / 2)
        order = max(order, ceil(omega_s * D / (omega_a - omega_p) * 0.5) * 2 + 1)

    if f_type == "band":
        omega_a1 = omega_1 - d_omega
        omega_a2 = omega_2 + d_omega
        omega_p1 = omega_1
        omega_p2 = omega_2

        omega_c.append((omega_p1 + omega_a1) / 2)
        omega_c.append((omega_p2 + omega_a2) / 2)
        order = max(ceil(omega_s * D / (omega_a1 - omega_p1)) + 1,
                    ceil(omega_s * D / (omega_a2 - omega_p2)) + 1)

    n = yield order, omega_c

    while True:
        beta = alpha * sqrt(1 - pow(2 * n / (order - 1), 2))
        n = yield np.i0(beta) / np.i0(alpha)


def hamming_window(f_type, args):
    alpha = 0.54
    order = args[0]
    omega_1 = args[1]*pi
    omega_2 = args[2]*pi if f_type == "band" else 0

    n = yield order, [omega_1, omega_2]
    while True:
        n = yield alpha + (1 - alpha) * cos(2 * pi * n / (order))


def_windows = {
    'kaiser': kaiser_window,
    'hamming': hamming_window
}


def getFir(w_type, f_type, w_args):
    H = []  # Импульсная характеристика фильтра
    H_d = []  # Идеальная импульсная характеристика
    W = []  # Весовая функция

    try:
        filter = def_filters[f_type]
        window = def_windows[w_type]
    except KeyError as e:
        raise ValueError('Undefined filter type: {}'.format(e.args[0]))

    win_gen = window(f_type, w_args)  # Init a generator for selected window
    order, omega_c = win_gen.send(None)  # First call: pre-calculation
    start, fin = -floor(order / 2), floor(order / 2)

    print(w_type, ":")
    print("Order = ", order)
    print("Omega 1 = ", w_args[1])
    print("Omega 2 = ", w_args[2] if f_type == "band" else "-")

    for n in range(start, fin + 1):
        H_d.append(filter(n, omega_c))  # Filter's function
        W.append(win_gen.send(n))  # Window's function
        H.append(H_d[-1] * W[-1])  # Сonvolution of coefficients

    win_gen.close()  # Generator shutdown

    # Impulse response normalization
    h_sum = 0.0
    for i in range(order):
        h_sum += H[i]
    for i in range(order):
        H[i] /= h_sum
    return H


def response(input, H):
    inp_size = len(input)
    fir_size = len(H)
    res = 0.

    for i in range(fir_size):
        if inp_size - 1 - i >= 0:
            res += H[i] * input[-1 - i]

    return res


def sequence(t, H, func):
    input = []
    output = []

    start_t = time.time() * 1.
    for i in range(t):
        input.append(func(i))
        output.append(response(input, H))
    end_t = time.time() * 1.

    return input, output, end_t - start_t


def worker(pos):
    fir_size = len(mymodule.H)
    res = 0

    for i in range(fir_size):
        if pos - 1 - i >= 0:
            res += mymodule.H[i] * mymodule.inp[pos - 1 - i]

    return res


def init_process(share1, share2):
    mymodule.inp = share1
    mymodule.H = share2


def parallel(t, H, f, thread_num):
    inp = []
    for i in range(t):
        inp.append(f(i))

    shared_inp = Array('d', len(inp), lock=False)
    shared_H = Array('d', len(H), lock=False)
    iterable = range(1, len(inp) + 1)

    # can set data after fork
    shared_inp = list(inp)
    shared_H = list(H)

    # fork
    pool = Pool(processes=thread_num, initializer=init_process, initargs=(shared_inp, shared_H))

    start_t = time.time() * 1.
    output = pool.map(worker, iterable)
    pool.close()
    pool.join()
    end_t = time.time() * 1.

    return inp, output, end_t - start_t


def plotGraphic(func, delta_y, name="Function"):
    t = list(range(len(func)))
    plt.xlim(0, 300)
    plt.ylim(-delta_y, delta_y)
    plt.plot(t, func)
    plt.title(name)
    plt.show()


def plotFFTGraphic(y, name="FFT Function"):
    N = len(y)
    yf = fft(y)
    xf = np.linspace(0.0, 1.0, N // 2)
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    plt.grid()
    plt.title(name)
    plt.show()


def error(arr1, arr2):
    if len(arr1) != len(arr2):
        return -1

    res = 0.
    for i in range(len(arr1)):
        if abs(arr1[i] - arr2[i]) - res > 0:
            res = abs(arr1[i] - arr2[i])

    return res


def main():
    H1 = getFir("kaiser", "band", [0.25, 0.2, 0.5])
    H2 = getFir("hamming", "band", [31, 0.2, 0.5])

    inp, output1, seq_time = sequence(1000, H1, polyharmonic)
    input, output2, seq_time = sequence(1000, H2, polyharmonic)

    plotGraphic(inp, 35, "input (Seq)")

    plotGraphic(output1, 35, "Low, Kaiser (Seq)")
    plotGraphic(output2, 35, "low, Hamming (Seq)")
    plotFFTGraphic(output1, "FFT low, Kaiser (Seq)")
    plotFFTGraphic(output2, "FFT low, Hamming (Seq)")

    # for t in range(2, 10):
    #     print("threads:", t)
    #     inp, output3, par_time = parallel(10000, H1, polyharmonic, t)
    #     print("error, SPEED-UP: ", error(output1, output3), seq_time / par_time)

    # plotGraphic(output3, 35, "Low, Kaiser (Par)")


if __name__ == "__main__":
    main()
