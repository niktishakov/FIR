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


def Polyharmonic(t):
    return (sin(2 * pi * 0.05 * t) +
            50 * sin(2 * pi * 0.1 * t) +
            10 * sin(2 * pi * 0.5 * t) +
            omega * random.randint(1, 500))




def GetFreqResp(omegaS, omegaP, omegaA, Aa, Ap):
    H = []  # Импульсная характеристика фильтра
    H_d = []  # Идеальная импульсная характеристика
    W = []  # Весовая функция
    Fc = (omegaP + omegaA) / (2 * omegaS)
    D = 0.9222 if (Aa <= 21) else (Aa - 7.95) / 14.36
    alpha = (0.0 if Aa <= 21 else
             0.5842 * pow(Aa - 21, 0.4) + 0.07886 * (Aa - 21) if Aa <= 50 else
             0.1102 * (Aa - 8.7))

    size = ceil(omegaS * D / (omegaA - omegaP) * 0.5) * 2 + 1   # Длина фильтра
    start, fin = -floor(size/2), floor(size/2)
    print("start, fin:", start, fin)

    for i in range(start, fin+1):
        if fabs(i) == floor(size/2):
            H_d.append(0)
        elif i == 0:
            H_d.append(2 * pi * Fc)
        else:
            H_d.append(sin(2 * pi * Fc * i) / (pi * i))

        # весовая функция Кайзера
        beta = alpha * sqrt(1 - pow(2 * i / (size - 1), 2))
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


def Response(input, H):
    if len(input) < 1:
        return 0

    output = 0.0
    inpSize = len(input)
    firSize = len(H)

    for i in range(firSize):
        if inpSize - 1 - i >= 0:
            output += H[i] * input[-1-i]
    return output


H = GetFreqResp(128, 20, 50, 30, 30)
t = list(range(500))
x = []
y = []

for i in t:
    x.append(Polyharmonic(i))
    y.append(Response(x, H))

plt.plot(t, x)
plt.show()
plt.plot(t, y)
plt.show()
