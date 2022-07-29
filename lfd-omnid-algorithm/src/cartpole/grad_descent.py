import numpy as np


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
        self.Q = []
        self.P = []
        self.A = []
        self.B = []

        self.hk_values = {}
        self.ck_values = {}
        self.phik_values = {}
        self.lambdak_values = {}

        self.trajectory = []

        self.at = []

    def grad_descent(self):
        self.trajectory = self.make_trajectory(self.x0, self.u)
        self.__recursive_wrapper(self.K + 1, [], self.n, self.calc_ck)
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
        fourier_basis = 1
        for i in range(len(xt)):
            fourier_basis *= np.cos((k[i] * np.pi * xt[i]) / self.L[i][1])
        hk = self.hk_values[self.__k_str(k)]
        Fk = (1 / hk) * fourier_basis
        return Fk

    def calc_DFk(self, k):
        xt = self.trajectory
        hk = self.hk_values[self.__k_str(k)]
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
        self.ck_values[self.__k_str(k)] = ck
        return ck

    def calc_at(self):
        self.__recursive_wrapper(self.K+1, [], self.n, self.calc_a)
        return self.q * self.at

    def calc_a(self, k):
        # at should be of size (trajectory length, n dimensions)
        k_str = self.__k_str(k)
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

    def __k_str(self, k):
        return ''.join(str(i) for i in k)

    def calc_P_r(self, at, bt):
        P, A, B, Q = self.P, self.A, self.B, self.Q
        t = np.arange(self.t0, self.tf, self.dt)
        listP, listr = np.zeros((len(t)+1, self.n, self.n)), np.zeros((len(t)+1, self.n, 1))
        listP[0] = np.zeros(np.shape(P))
        listr[0] = -np.array([[0.]*self.n]).T
        Rinv = np.linalg.inv(self.R)
        for i in range(len(t)):
            P_dot = P@A + np.transpose(A)@P - P@(B @ Rinv @ np.transpose(B)) @ P + Q
            a, b = at[i], bt[i]
            r_dot = np.transpose(A - B @ Rinv @ np.transpose(B) @ P) @ listr[i] + a - (P @ B @ Rinv) @ b
            listP[i+1] = self.dt * P_dot + listP[i]
            listr[i+1] = self.dt * r_dot + listr[i]
        listP, listr = np.flip(listP, 0), np.flip(listr, 0)
        return listP, listr

    def desc_dir(self, listP, listr, bt):
        A, B = self.A, self.B
        t = np.arange(self.t0, self.tf, self.dt)
        z = np.array([[0.0] * self.n])
        Rinv = np.linalg.inv(self.R)
        # size of zeta needs to change (stacks z and v)
        zeta = np.zeros((len(t)+1))
        for i in range(len(t)):
            P, r, b = listP[i], listr[i], bt[i]
            v = -Rinv @ np.transpose(B) @ P @ z - Rinv @ B.T @ r - Rinv @ b
            zeta[i] = (z, v)
            zdot = A @ z + B @ v
            z += zdot * self.dt
        return zeta

    def DJ(self, zeta, at, bt):
        t = np.arange(self.t0, self.tf, self.dt)
        J = np.zeros((len(t)+1))
        for i in range(len(t)):
            # size of zeta will change here depending on desc direction function
            z, v = zeta[i][0], zeta[i][1]
            a = np.transpose(at[i])
            b = np.transpose(bt[i])  # might be wrong need to double-check -> b instead of u @ R
            J_val = a @ z + b @ v    # might be wrong due to b -> u @ R @ v
            J[i] = J_val[0][0] # need to double-check this
        integrate_J = np.trapz(J, dx=self.dt)
        return integrate_J

