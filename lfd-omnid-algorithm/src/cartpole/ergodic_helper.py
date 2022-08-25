import numpy as np


class ErgodicHelper:
    def __init__(self, D, E, K, L, dt):
        # Creating Fourier Distributions
        self.D = D
        self.E = E

        # Ergodic Measure variables
        self.K = K
        self.n = len(D[0][0])
        self.L = L  # needs upper and lower bound (L0, L1)
        self.dt = dt

        # Weights for demonstration trajectories
        self.m = len(self.D)
        self.w = np.array([(1/self.m) for _ in range(self.m)])

        # Stores lambda_k, phi_k, and ck values
        self.lambdak_values = {}
        self.phik_values = {}
        self.hk_values = {}

    def calc_fourier_metrics(self):
        self.__recursive_wrapper(self.K + 1, [], self.n, self.calc_phik)
        self.__recursive_wrapper(self.K + 1, [], self.n, self.calc_lambda_k)
        return self.hk_values, self.phik_values, self.lambdak_values

    def calc_Fk(self, x, k):
        # Effect of different L value -> not bound from 0 to Li
        hk = self.calc_hk(k)
        fourier_basis = 1
        for i in range(len(x)):
            fourier_basis *= np.cos((k[i]*np.pi*x[i])/self.L[i][1])
        Fk = (1/hk)*fourier_basis
        return Fk

    def calc_hk(self, k):
        hk = 1
        for i in range(self.n):
            l0, l1 = self.L[i][0], self.L[i][1]
            if not k[i]:
                hk *= (l1 - l0)
                continue
            k_i = (k[i] * np.pi) / l1
            hk *= (2 * k_i * (l1 - l0) - np.sin(2 * k_i * l0) + np.sin(2 * k_i * l1)) / (4 * k_i)
        k_str = ''.join(str(i) for i in k)
        hk = np.sqrt(hk)
        self.hk_values[k_str] = hk
        return hk

    def calc_ck(self, x_t, k):
        x_len = len(x_t)
        T = x_len * self.dt
        Fk_x = np.zeros(x_len)
        for i in range(x_len):
            Fk_x[i] = self.calc_Fk(x_t[i], k)
        ck = (1 / T) * np.trapz(Fk_x, dx=self.dt)
        return ck

    def calc_phik(self, k):
        phik = 0
        for i in range(self.m):
            phik += self.E[i] * self.w[i] * self.calc_ck(self.D[i], k)
        k_str = ''.join(str(i) for i in k)
        self.phik_values[k_str] = phik
        # return phik

    def calc_lambda_k(self, k):
        s = (self.n + 1) / 2
        lamnbda_k = 1 / ((1 + np.linalg.norm(k) ** 2) ** s)
        k_str = ''.join(str(i) for i in k)
        self.lambdak_values[k_str] = lamnbda_k

    def __recursive_wrapper(self, K, k_arr, n, f):
        # When calling this function, call with K+1
        if n > 0:
            for k in range(K):
                self.__recursive_wrapper(K, k_arr + [k], n - 1, f)
        else:
            f(k_arr)
