import numpy as np


class ErgodicLFD:
    def __int__(self, D, E, K, n, L, T, dt):
        # Creating Fourier Distributions
        self.D = D
        self.E = E

        self.K = K
        self.n = n
        self.L = L  # needs upper and lower bound (L0, L1)

        # Controller Variables:
        self.T = T
        self.dt = dt
        self.x_t = []

        # Weights for demonstration trajectories
        self.m = len(self.D)
        self.w = np.array([(1/self.m) for _ in range(self.m)])

        # Stores lambda_k, phi_k, and ck values
        self.epsilon = 0
        self.lambdak_values = {}
        self.phik_values = {}
        self.ck_values = {}
        self.hk_values = {}

    def calc_fourier_metrics(self):



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
            l0, l1 = self.L[0], self.L[1]
            k = (k[i] * np.pi) / l1
            hk *= (2 * k * (l1 - l0) - np.sin(2 * k * l0) + np.sin(2 * k * l1)) / (4 * k)
        return np.sqrt(hk)

    def calc_ck(self, k, x_t=None, T=None):
        if not x_t:
            # Takes trajectory if one is inputted -> calculating for phik
            # ck will use self.x_t
            x_t = self.x_t
        if not T:
            # Assumes self.dt is the same for the controller
            T = len(x_t) * self.dt
        Fk_x = np.zeros(T)
        for i in range(T):
            Fk_x[i] = self.calc_Fk(x_t[i], k)
        ck = (1 / T) * np.trapz(Fk_x, dx=self.dt)
        k_str = ''.join(str(i) for i in k)
        self.ck_values[k_str] = ck
        return ck

    def calc_phik(self, k):
        phik = 0
        for i in range(self.m):
            phik += self.E * self.w[i] * self.calc_ck(self.D[i], k)
        k_str = ''.join(str(i) for i in k)
        self.phik_values[k_str] = phik
        # return phik

    def calc_lambda_k(self, k):
        s = (self.n + 1) / 2
        lamnbda_k = 1 / ((1 + np.linalg.norm(k) ** 2) ** s)
        k_str = ''.join(str(i) for i in k)
        self.lambdak_values[k_str] = lamnbda_k

    def calc_epsilon(self, k):
        k_str = ''.join(str(i) for i in k)
        lambdak = self.lambdak_values[k_str]
        ck = self.ck_values[k_str]
        phik = self.phik_values[k_str]
        eps = lambdak * (np.abs(ck - phik)) ** 2
        # self.epsilon += eps
        return eps

    def __recursive_wrapper(self, K, k_arr, n, f):
        # When calling this function, call with K+1
        if n > 0:
            for k in range(K):
                self.__recursive_wrapper(K, n - 1, k_arr + [k], f)
        else:
            f(k_arr)
