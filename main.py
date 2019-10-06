import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np

import time
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
        return sin(omega_c[1] * n) / (pi * n) - sin(omega_c[0] * n) / (pi * n)


def_filters = {
    'low': low_pass_filter,
    'band': bandpass_filter,
}


def kaiser_window(f_type):
    N = 1 if f_type == "low" else 2
    omega_c = []
    order = alpha = 1

    for i in range(N):
        omega_s = 1  # omega_s - частота дискретизации
        omega_p = 0.225 + i * 0.2  # omega_p - полоса пропускания
        omega_a = 0.275 + i * 0.2  # omega_a - полоса заграждения
        A_o = 30  # A_o - определяет A_a, A_p

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


def hamming_window(f_type):
    N = 1 if f_type == "low" else 2
    alpha = 0.54
    omega_c = []

    for i in range(N):
        omega_c.append(0.4 + 0.1 * i)
    order = 33

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

    print(w_type, "\nOrder = ", order)
    print("Omega_c:", omega_c[0], (omega_c[1] if len(omega_c) == 2 else "-"), "\n")

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
    if len(input) < 1:
        return 0

    output = 0.0
    inp_size = len(input)
    fir_size = len(H)

    for i in range(fir_size):
        if inp_size - 1 - i >= 0:
            output += H[i] * input[-1 - i]

    return output


def sequence(t, H1, H2, func):
    input = []
    output1 = [0.] * t
    output2 = [0.] * t

    for i in range(t):
        input.append(func(i))
        output1[i] = response(input, H1)
        output2[i] = response(input, H2)

    return input, output1, output2


def worker(out1, out2, input, inp_size, H1, H2):
    if inp_size < 1:
        return

    fir_size = len(H1)
    for i in range(fir_size):
        if inp_size - 1 - i >= 0:
            out1[inp_size - 1] += H1[i] * input[inp_size - 1 - i]

    fir_size = len(H2)
    for i in range(fir_size):
        if inp_size - 1 - i >= 0:
            out2[inp_size - 1] += H2[i] * input[inp_size - 1 - i]

    barrier.wait()


def parallel(t, H1, H2, func, thread_num):
    input = []
    output1 = [0.] * t
    output2 = [0.] * t

    global barrier
    barrier = threading.Barrier(thread_num + 1)
    thread = [None] * thread_num

    for i in range(floor(t / thread_num)):
        for p in range(thread_num):
            input.append(func(i * thread_num + p))
            thread[p] = threading.Thread(target=worker, args=(output1, output2, input, len(input), H1, H2))
            thread[p].start()
        barrier.wait()

    if len(input) < t:
        print("LAST PART..")
        start = len(input)
        barrier = threading.Barrier(t - start + 1)
        for p in range(start, t):
            input.append(polyharmonic(p))
            thread[p - start] = threading.Thread(target=worker, args=(output1, output2, input, p, H1, H2))
            thread[p - start].start()
        barrier.wait()

    return input, output1, output2


def plotGraphic(func, delta_y, name="Function"):
    t = list(range(len(func)))
    plt.xlim(-2, 200)
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


def main():
    H1 = getFir("kaiser", "band")
    H2 = getFir("hamming", "band")

    par_time = time.time() * 1.
    input, output1, output2 = parallel(4000, H1, H2, polyharmonic, 200)
    par_time = time.time() * 1. - par_time

    plotGraphic(input, 35, "Input")
    plotGraphic(output1, 35, "Output After Kaiser")
    plotGraphic(output2, 35, "Output After Hamming")

    plotFFTGraphic(input, "FFT Input")
    plotFFTGraphic(output1, "FFT Output After Kaiser")
    plotFFTGraphic(output2, "FFT Output After Hamming")

    seq_time = time.time() * 1.
    input, output3, output4 = sequence(4000, H1, H2, delta_function)
    seq_time = time.time() * 1. - seq_time

    plotFFTGraphic(input, "FFT Input")
    plotFFTGraphic(output3, "FFT Output After Kaiser")
    plotFFTGraphic(output4, "FFT Output After Hamming")

    print("BOOST: ", par_time / seq_time)


if __name__ == "__main__":
    main()
