import numpy as np


"""
TODO:
1. Add armijo line search -> changing definitions of how reccursive wrapper is used
2. Test with actual demonstrations
3. Integrate into a script
4. Fix DJ
"""


class MPC:
    def __init__(self, x0, t0, tf):
        # System variables
        self.x0 = x0
        self.n = len(x0)
        self.t0, self.tf = t0, tf
        self.dt = 0.1
        self.t = np.arange(self.t0, self.tf, self.dt)
        self.it = len(self.t)

        # Initialize x_t and u_t variables
        self.u = np.zeros((self.it, self.n, 1))
        self.x_trajectory = []

        # Control Constants
        self.K = 6
        self.q = 1100
        self.R = 2*np.eye(self.n, dtype=float)
        self.Q = 10*np.eye(self.n, dtype=float)
        self.P = np.zeros((self.n, self.n))
        self.L = []

        # Grad descent
        self.beta = 0.95
        self.eps = 0.00001

        # Control Variables
        self.at = np.zeros((np.shape(self.x_trajectory)))
        self.bt = np.zeros((np.shape(self.u)))

        # Dynamics constants
        self.M, self.m, self.l = 0, 0, 0
        self.A, self.B = self.__calc_A_B()

        # Variable values as a function of k
        self.hk_values = {}
        self.ck_values = {}
        self.phik_values = {}
        self.lambdak_values = {}

    def grad_descent(self):
        self.x_trajectory = self.make_trajectory(self.x0, self.u)
        self.__recursive_wrapper(self.K + 1, [], self.n, self.__calc_ck)
        gamma = self.beta

        # Need to add armijo line search
        while True:
            at, bt = self.__calc_at(), self.__calc_b()
            listP, listr = self.calc_P_r(at, bt)
            zeta = self.desc_dir(listP, listr, bt)

            x0 = self.x_trajectory[0, :]
            v = zeta[:][1]
            u_new = self.u + gamma * v
            u_new = u_new[:, 0]

            self.x_trajectory = self.make_trajectory(x0, u_new)

            if self.DJ(zeta, at, bt) < self.eps:
                break
        return 0

    def make_trajectory(self, x0, u):
        # Creates a trajectory given initial state and controls
        t = np.arange(self.t0, self.tf, self.dt)
        x_traj = np.zeros((len(t)+1, len(x0)))
        x_traj[0, :] = x0[:]
        for i in range(1, len(t)+1):
            x_traj[:, i] = self.__integrate(x_traj[i-1, :], u[i, :])
        return x_traj

    def desc_dir(self, listP, listr, bt):
        A, B = self.A, self.B
        z = np.array([[0.0] * self.n])
        Rinv = np.linalg.inv(self.R)
        zeta = []
        for i in range(self.it):
            P, r, b = listP[i], listr[i], bt[i]
            v = -Rinv @ np.transpose(B) @ P @ z - Rinv @ B.T @ r - Rinv @ b
            zeta.append((z, v))
            zdot = A @ z + B @ v
            z += zdot * self.dt
        return zeta

    def DJ(self, zeta, at, bt):
        J = np.zeros((self.it+1))
        for i in range(self.it):
            z, v = zeta[i][0], zeta[i][1]
            a = np.transpose(at[i])
            b = np.transpose(bt[i])  # might be wrong need to double-check -> b instead of u @ R
            J_val = a @ z + b @ v    # might be wrong due to b -> u @ R @ v
            J[i] = J_val[0][0] # need to double-check this
        J_integral = np.trapz(J, dx=self.dt)
        return J_integral

    def calc_P_r(self, at, bt):
        P, A, B, Q = self.P, self.A, self.B, self.Q
        listP, listr = np.zeros((self.it+1, self.n, self.n)), np.zeros((self.it+1, self.n, 1))
        listP[0] = np.zeros(np.shape(P))
        listr[0] = -np.array([[0.]*self.n]).T
        Rinv = np.linalg.inv(self.R)
        for i in range(self.it):
            P_dot = P@A + np.transpose(A)@P - P@(B @ Rinv @ np.transpose(B)) @ P + Q
            a, b = at[i], bt[i]
            r_dot = np.transpose(A - B @ Rinv @ np.transpose(B) @ P) @ listr[i] + a - (P @ B @ Rinv) @ b
            listP[i+1] = self.dt * P_dot + listP[i]
            listr[i+1] = self.dt * r_dot + listr[i]
        listP, listr = np.flip(listP, 0), np.flip(listr, 0)
        return listP, listr

    def __dynamics(self):
        # https://sites.wustl.edu/slowfastdynamiccontrolapproaches/cart-pole-system/cart-pole-dynamics-system/
        M, m, l = self.M, self.m, self.l
        g = 9.81  # gravitational constant
        I = (m * (l**2))/12
        denominator = 1 / (I*(M+m) + M*m*(l**2))
        a = denominator * (g * (m**2) * (l**2))
        b = denominator * (-g * (M + m))
        c = denominator * (I + m * (l**2))
        d = denominator * (-m * l)
        return a, b, c, d

    def __calc_A_B(self):
        a, b, c, d = self.__dynamics()
        A = np.array([[0, 1, 0, 0],
                      [0, 0, a, 0],
                      [0, 0, 0, 1],
                      [0, 0, b, 0]])
        B = np.array([[0],
                      [c],
                      [0],
                      [d]])
        return A, B

    def __cart_pole_dyn(self, X, U):
        # Cart pole dynamics should take state x(t) and u(t) and return array corresponding to dynamics
        A, B = self.A, self.B
        Xdot = A@X + B@U
        return Xdot

    def __integrate(self, xi, ui):
        # Finds x_t(t + dt) given dynamcis f, and x_t, u_t, and dt
        f = self.__cart_pole_dyn
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

    def __calc_DFk(self, k):
        xt = self.x_trajectory
        hk = self.hk_values[self.__k_str(k)]
        dfk = np.zeros(np.shape(xt))
        for t, x in enumerate(xt):
            for i in x:
                ki = (k[i] * np.pi)/self.L[i]
                dfk_xi = (1/hk) * -ki * np.cos(ki * x[i]) * np.sin(ki * x[i])
                dfk[t, :] = dfk_xi
        return dfk

    def __calc_ck(self, k):
        x_t = self.x_trajectory
        Fk_x = np.zeros(len(x_t))
        for i in range(len(x_t)):
            Fk_x[i] = self.__calc_Fk(x_t[i], k)
        ck = (1 / self.tf) * np.trapz(Fk_x, dx=self.dt)
        self.ck_values[self.__k_str(k)] = ck
        return ck

    def __calc_at(self):
        self.at = np.zeros((np.shape(self.x_trajectory)))
        self.__recursive_wrapper(self.K+1, [], self.n, self.__calc_a)
        self.at *= self.q
        return self.at

    def __calc_a(self, k):
        k_str = self.__k_str(k)
        lambdak = self.lambdak_values[k_str]
        ck = self.ck_values[k_str]
        phik = self.phik_values[k_str]
        # DFk should be of size (trajectory), maybe change to matrix addition
        self.at = ((lambdak * 2 * (ck - phik) * (1/self.tf)) * self.__calc_DFk(k)) + self.at

    def __calc_b(self):
        u, R = self.u, self.R
        return np.transpose(u)@R

    def __recursive_wrapper(self, K, k_arr, n, f):
        # When calling this function, call with K+1
        if n > 0:
            for k in range(K):
                self.__recursive_wrapper(K, k_arr + [k], n - 1, f)
        else:
            f(k_arr)

    def __k_str(self, k):
        return ''.join(str(i) for i in k)
