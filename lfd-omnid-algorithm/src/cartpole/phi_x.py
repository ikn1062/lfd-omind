import numpy as np
from scipy.integrate import dblquad

"""
def calculate_Fk(x, k, L, n):
    hk = calculate_hk_l2(k, L)
    fourier_basis = 1
    for i in range(n):
        fourier_basis *= np.cos((k[i] * np.pi * x[i]) / L[i])
    Fk = (1 / hk) * fourier_basis
    return Fk


def calculate_hk_l2(k, L):
    # calculate_hk only works in 2-d currently -> using dblquad
    L1, L2 = L[0], L[1]
    k1, k2 = (k[0] * np.pi) / L1, (k[1] * np.pi) / L2
    hk = dblquad(lambda x1, x2: (np.cos(k1 * x1)) ** 2 + (np.cos(k2 * x2)) ** 2, 0, L1, lambda x1: 0, lambda x2: L2)
    hk = np.sqrt(hk)
    return hk


def calculate_ck(x_t, k, L, T, dt):
    n = len(x_t)
    dim = len(x_t[0])
    coeff_t = np.zeros(n)
    for i in range(n):
        coeff_t[i] = calculate_Fk(x_t[i], k, L, dim)
    ck = (1 / T) * np.trapz(coeff_t, dx=dt)
    return ck


def calculate_phik(D, E, W, k, L, dt):
    m = len(D)
    phik = 0
    for i in range(m):
        x_t = D[i]
        T = len(x_t) * dt
        phik += E[i] * W[i] * calculate_ck(x_t, k, L, T, dt)
    return phik
"""

if __name__ == "__main__":
    K = 6
    n = 4
    values = {}

    def func(arr, val):
        string = ''.join(str(x) for x in arr)
        values[string] = val

    def recc(k, n1, arr, val, f):
        if n1 > 0:
            for x in range(K):
                recc(k, n1 - 1, arr + [x], val + x, f)
        else:
            f(arr, val)


    recc(K + 1, n, [], 0, func)
    print(values)
