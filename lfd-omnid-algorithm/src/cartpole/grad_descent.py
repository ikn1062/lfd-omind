import numpy as np
import matplotlib.pyplot as plt


class MPC:
    def __init__(self, x0, t0, tf, L, hk, phik, lambdak, dt=0.01, K=6):
        # System variables
        self.n = len(x0)
        self.t0, self.tf = t0, tf
        self.dt = dt

        # Initialize x_t and u_t variables
        self.u = np.array([[0], [0]])
        self.x_t = x0

        # Control Constants
        self.K = K
        self.q = 10
        self.R = 2
        # self.Q = 10*np.eye(self.n, dtype=float)
        self.Q = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 10, 0],
                           [0, 0, 0, 10]])
        # self.P = 10 * np.eye(self.n)  # P(t) is P1
        self.P = 10 * np.array([[0.01, 0, 0, 0],
                                [0, 0.01, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
        self.L = L

        # Grad descent
        self.beta = 0.15

        # Control Variables
        self.at = np.zeros((np.shape(self.x_t)))
        self.bt = np.zeros(np.shape(self.u))

        # Dynamics constants
        self.M, self.m, self.l = 20, 20, 1.0
        self.A, self.B = self.__calc_A_B()

        # Variable values as a function of k
        self.hk_values = hk
        self.phik_values = phik
        self.lambdak_values = lambdak
        self.ck_values = {}

    def grad_descent(self):
        gamma = self.beta
        t0 = self.t0

        output_x, output_u = [], []

        while t0 < self.tf:
            at, bt = self.calc_at(), self.calc_b()
            listP, listr = self.calc_P_r(at, bt)
            zeta = self.desc_dir(listP, listr, bt)

            v = zeta[:][1]
            self.u = self.u + gamma * v
            if self.u > 200:
                self.u = 200
            if self.u < -200:
                self.u = -200

            self.x_t = self.integrate(self.x_t, self.u)

            while self.x_t[2] < 0:
                self.x_t[2] += 2 * np.pi
            while self.x_t[2] > 2 * np.pi:
                self.x_t[2] -= 2 * np.pi
            if self.x_t[2] > np.pi:
                self.x_t[2] -= 2 * np.pi

            output_x.append(self.x_t)
            output_u.append(self.u)
            print(f"x: {self.x_t}, u: {self.u}")
            print(f"DJ: {self.DJ(zeta, at, bt)}")

            t0 += self.dt

        output_x = np.array(output_x)
        output_u = np.array(output_u)
        t = np.arange(self.t0, self.tf + self.dt, self.dt)
        plt.plot(t, output_x[:, 0])
        plt.show()
        plt.plot(t, output_x[:, 1])
        plt.show()
        plt.plot(t, output_x[:, 2])
        plt.show()
        plt.plot(t, output_x[:, 3])
        plt.show()
        plt.plot(t, output_u[:, 0])
        plt.show()
        return 0

    def make_trajectory(self, xt, u, ti, tf):
        # Creates a trajectory given initial state and controls
        t = np.arange(ti, tf, self.dt)
        x_traj = np.zeros((len(t) + 1, len(xt)))
        x_traj[0, :] = np.transpose(xt[:])
        for i in range(1, len(t) + 1):
            xt_1 = self.integrate(x_traj[i - 1, :], u[i, :])
            x_traj[i, :] = np.transpose(xt_1)
        return x_traj

    def desc_dir(self, listP, listr, bt):
        A, B = self.A, self.B
        z = np.array([[0.0] * self.n]).T
        Rinv = (-1 / self.R)
        zeta = []
        for i in range(len(bt)):
            P, r, b = listP[i], listr[i], bt[i]
            v = -Rinv * np.transpose(B) @ P @ z - Rinv * np.transpose(B) @ r - Rinv * b
            zeta.append((z, v))
            zdot = A @ z + B @ v
            z += zdot * self.dt
        return zeta

    def DJ(self, zeta, at, bt):
        J = np.zeros((len(zeta)))
        for i in range(len(zeta)):
            z, v = zeta[i][0], zeta[i][1]
            a_T = np.transpose(at[i])
            b_T = np.transpose(bt[i])
            J_val = a_T @ z + b_T @ v
            J[i] = J_val
        J_integral = np.trapz(J, dx=self.dt)
        return J_integral

    def calc_P_r(self, at, bt):
        P, A, B, Q = self.P, self.A, self.B, self.Q
        dim_len = len(at)
        listP, listr = np.zeros((dim_len, self.n, self.n)), np.zeros((dim_len, self.n, 1))
        listP[0] = np.zeros(np.shape(P))
        listr[0] = -np.array([[0.] * self.n]).T
        Rinv = -1/self.R
        for i in range(dim_len-1):
            # difference in Todds lecture notes for Pdot
            P_dot = P @ (B * Rinv * np.transpose(B)) @ P - Q - P @ A + np.transpose(A) @ P
            a, b = np.transpose([at[i]]), bt[i]
            r_dot = - np.transpose(A - B * Rinv * np.transpose(B) @ P) @ listr[i] - a + (P @ B * Rinv) * b
            listP[i + 1] = self.dt * P_dot + listP[i]
            listr[i + 1] = self.dt * r_dot + listr[i]
        return listP, listr

    def dynamics(self):
        # https://sites.wustl.edu/slowfastdynamiccontrolapproaches/cart-pole-system/cart-pole-dynamics-system/
        M, m, l = self.M, self.m, self.l
        g = 9.81  # gravitational constant
        I = (m * (l ** 2)) / 12
        denominator = 1 / (I * (M + m) + M * m * (l ** 2))
        a = denominator * (g * (m ** 2) * (l ** 2))
        b = denominator * (-g * (M + m))
        c = denominator * (I + m * (l ** 2))
        d = denominator * (-m * l)
        return a, b, c, d

    def __calc_A_B(self):
        a, b, c, d = self.dynamics()
        A = np.array([[0, 1, 0, 0],
                      [0, 0, a, 0],
                      [0, 0, 0, 1],
                      [0, 0, b, 0]])
        B = np.array([[0],
                      [c],
                      [0],
                      [d]])
        return A, B

    def cart_pole_dyn(self, X, U):
        # Cart pole dynamics should take state x(t) and u(t) and return array corresponding to dynamics
        A, B = self.A, self.B
        Xdot = A @ X + B * U
        return Xdot

    def integrate(self, xi, ui):
        # Finds x_t(t + dt) given dynamcis f, and x_t, u_t, and dt
        if np.shape(xi) != (4, 1):
            xi = np.transpose([xi])
        f = self.cart_pole_dyn
        k1 = self.dt * f(xi, ui)
        k2 = self.dt * f(xi + k1 / 2, ui)
        k3 = self.dt * f(xi + k2 / 2, ui)
        k4 = self.dt * f(xi + k3, ui)
        x_i_next = xi + (1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        return x_i_next

    def __calc_Fk(self, xt, k):
        fourier_basis = 1
        for i in range(len(xt)):
            fourier_basis *= np.cos((k[i] * np.pi * xt[i]) / self.L[i][1])
        hk = self.hk_values[self.__k_str(k)]
        Fk = (1 / hk) * fourier_basis
        return Fk

    def calc_DFk(self, k):
        x_t = self.x_t
        hk = self.hk_values[self.__k_str(k)]
        dfk = np.zeros(np.shape(x_t))
        for t, x in enumerate(x_t):
            for i in range(len(x)):
                ki = (k[i] * np.pi) / self.L[i][1]
                dfk_xi = (1 / hk) * -ki * np.cos(ki * x[i]) * np.sin(ki * x[i])
                dfk[t, i] = dfk_xi
        return dfk

    def calc_ck(self, k):
        x_t = self.x_t
        T = len(x_t) * self.dt
        Fk_x = np.zeros(len(x_t))
        for i in range(len(x_t)):
            Fk_x[i] = self.__calc_Fk(x_t[i], k)
        ck = (1 / T) * np.trapz(Fk_x, dx=self.dt)
        self.ck_values = ck
        return ck

    def calc_at(self):
        self.at = np.zeros((np.shape(self.x_t)))
        self.__recursive_wrapper(self.K + 1, [], self.n, self.__calc_a)
        self.at *= self.q
        return self.at

    def __calc_a(self, k):
        k_str = self.__k_str(k)
        lambdak = self.lambdak_values[k_str]
        ck = self.calc_ck(k)
        phik = self.phik_values[k_str]
        # DFk should be of size (xt), maybe change to matrix addition
        self.at = ((lambdak * 2 * (ck - phik) * (1 / self.tf)) * self.calc_DFk(k)) + self.at

    def calc_b(self):
        bt = np.zeros((len(self.u), 1))
        for i, u in enumerate(self.u):
            bt[i] = u * self.R
        return bt

    def __recursive_wrapper(self, K, k_arr, n, f):
        # When calling this function, call with K+1
        if n > 0:
            for k in range(K):
                self.__recursive_wrapper(K, k_arr + [k], n - 1, f)
        else:
            f(k_arr)

    def __k_str(self, k):
        return ''.join(str(i) for i in k)
