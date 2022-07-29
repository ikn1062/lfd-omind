import numpy as np

"""
QUESTIONS:::
1. How should I define u?
2. Dynamics resources for cartpole
3. What is DF
4. Size of a(t)
"""


class MPC:
    def __init__(self, x0, t0, tf):
        self.x0 = x0
        self.u = []

        self.t0, self.tf = t0, tf
        self.dt = 0.1

        self.K = 6
        self.L = []
        self.n = len(x0)
        self.q = 10
        self.R = []

        self.hk_values = {}
        self.ck_values = {}
        self.phik_values = {}
        self.lambdak_values = {}

        self.trajectory = []

        self.at = []

    def grad_descent(self):
        self.trajectory = self.make_trajectory(self.x0, self.u)
        self.__recursive_wrapper(self.K + 1, [], self.n, self.calc_ck)
        # WTF is the shape of a????
        at = self.calc_at()
        bt = self.calc_b()

        return 0

    def __cart_pole_dyn(self, x, u):
        # Cart pole dynamics should take state x(t) and u(t) and return array corresponding to dynamics
        return 0

    def __integrate(self, x_t, u_t):
        # Finds x_t(t + dt) given dynamcis f, and x_t, u_t, and dt
        return 0

    def make_trajectory(self, x0, u):
        # Creates a trajectory given initial state and controls
        return 0

    def calc_Fk(self, xt, k):
        # hk = self.calc_hk(k)
        fourier_basis = 1
        for i in range(len(xt)):
            fourier_basis *= np.cos((k[i] * np.pi * xt[i]) / self.L[i][1])
        k_str = ''.join(str(i) for i in k)
        hk = self.hk_values[k_str]
        Fk = (1 / hk) * fourier_basis
        return Fk

    def calc_DFk(self, k):
        xt = self.trajectory
        k_str = ''.join(str(i) for i in k)
        hk = self.hk_values[k_str]

        dfk = [[] for _ in range(len(xt))]
        for t, x in enumerate(xt):
            for i in x:
                ki = (k[i] * np.pi)/self.L[i]
                dfk_xi = (1/hk) * -ki * np.cos(ki * x[i]) * np.sin(ki * x[i])
                dfk[t].append(dfk_xi)
        return dfk

    def calc_ck(self, k):
        x_t = self.trajectory
        Fk_x = np.zeros(self.tf)
        for i in range(self.tf):
            Fk_x[i] = self.calc_Fk(x_t[i], k)
        ck = (1 / self.tf) * np.trapz(Fk_x, dx=self.dt)
        k_str = ''.join(str(i) for i in k)
        self.ck_values[k_str] = ck
        return ck

    def calc_at(self):
        self.__recursive_wrapper(self.K+1, [], self.n, self.calc_a)
        return self.q * self.at

    def calc_a(self, k):
        # at should be of size (trajectory length, n dimensions)
        k_str = ''.join(str(i) for i in k)
        lambdak = self.lambdak_values[k_str]
        ck = self.ck_values[k_str]
        phik = self.phik_values[k_str]
        # DFk should be of size (trajectory length, n dimensions), maybe change to matrix addition
        self.at += (lambdak * 2 * (ck - phik) * (1/self.tf)) * self.calc_DFk(k)

    def calc_b(self):
        u, R = self.u, self.R
        return np.transpose(u)@R

    def __recursive_wrapper(self, K, k_arr, n, f):
        # When calling this function, call with K+1
        if n > 0:
            for k in range(K):
                self.__recursive_wrapper(K, n - 1, k_arr + [k], f)
        else:
            f(k_arr)
