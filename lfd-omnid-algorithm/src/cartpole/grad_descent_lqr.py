import numpy as np
import matplotlib.pyplot as plt
from time import sleep


class LQR_mpc:
    def __init__(self, x0, t0, tf, L, hk, phik, lambdak, A, B, dt=0.01, K=6, max_u=100, rospy_pub=None):
        # System variables
        self.n = len(x0)
        self.t0, self.tf = t0, tf
        self.dt = dt

        # Initialize x_t and u_t variables
        self.u = np.array([[32], [32]])
        self.max_u = max_u
        self.x0 = np.transpose(x0)
        self.x_t = np.transpose(x0)

        # Control Constants
        self.K = K
        self.q = 1
        self.R = np.array([[0.01]])
        # self.Q = 10*np.eye(self.n, dtype=float)
        self.Q = np.diag([0.1, 1.0, 100.0, 5.0])

        self.P = np.zeros((self.n, self.n))
        self.r = np.array([[0.] * self.n]).T

        self.L = L

        # Grad descent
        self.beta = 0.15

        # Control Variables
        self.at = np.zeros((np.shape(self.x_t)))
        self.bt = np.zeros(np.shape(self.u))

        # Dynamics constants
        self.A, self.B = A, B

        # Variable values as a function of k
        self.hk_values = hk
        self.phik_values = phik
        self.lambdak_values = lambdak
        self.ck_values = {}

        # Set up Rospy Publisher
        self.rospy_pub = rospy_pub

    def grad_descent(self):
        """
        Uses iterative gradient descent to find optimal control at each time step
        - Compares spatial statistics of current trajectory to the spatial distributions of demonstrations

        - Creates plot of position state vector x over a period of time
        - Creates plot of control state vector u over a period of time

        :return: None
        """
        gamma = self.beta
        t0, dt = self.t0, self.dt
        self.x_t = self.make_trajectory(self.x0, self.u)

        t = np.arange(self.t0, self.tf + 2 * self.dt, self.dt)
        x_values = np.zeros((len(t), self.n))
        u_values = np.zeros((len(t), 1))

        x_values[0, :] = self.x_t[1, :]
        u_values[0] = self.u[0]

        ii = 1
        while t0 < self.tf:
            print("New loop")
            print(f"x_trajec:\n {self.x_t}")
            at, bt = self.calc_at(), self.calc_b()
            listP, listr = self.calc_P_r(at, bt)
            zeta = self.desc_dir(listP, listr, bt)
            DJ = self.DJ(zeta, at, bt)
            print(f"DJ:\n {DJ}")
            v = zeta[1]
            print(v)
            u_new = self.u + gamma * v

            if self.rospy_pub:
                self.rospy_pub.publish(u_new[1, 0])

            print(f"u_new:\n {u_new}")
            self.x_t = self.make_trajectory(self.x_t[1], u_new, t0, t0 + dt)

            self.u = u_new[:]

            x_values[ii, :] = self.x_t[1, :]
            u_values[ii] = u_new[1]

            t0 += self.dt
            ii += 1

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(t, x_values[:, 0])
        axs[0, 1].plot(t, x_values[:, 1])
        axs[1, 0].plot(t, x_values[:, 2])
        axs[1, 1].plot(t, x_values[:, 3])
        plt.show()

        plt.plot(t, u_values)
        plt.show()
        return 0

    def make_trajectory(self, x0, u):
        """
        Creates a trajectory of x from a given initial state, xt, and control input u
        - The output is a trajectory over the time horizon T of the input control

        :param x0: Initial position vector (np array of shape (1, n))
        :param u: Control input (np array of shape (T, m)
        :return: A trajectory of x over time horizon T of the input control (np array of shape (T, n))
        """
        x_traj = np.zeros((len(u), len(x0)))
        x_traj[0, :] = np.transpose(x0[:])
        for i in range(1, len(u)):
            xt_1 = self.integrate(x_traj[i - 1, :], u[i - 1])
            x_traj[i, :] = np.transpose(xt_1)
        return x_traj

    def desc_dir(self, listP, listr, bt):
        """
        Calculates the descent direction for position vector x and control vector u over the time horizon T

        - Descent direction for control is given by vector v:
        v = -Rinv @ np.transpose(B) @ P @ z - Rinv @ np.transpose(B) @ r - Rinv * b

        - Descent direction for trajectory is given by vector z:
        z = (A @ z[i-1] + B @ v) * dt

        z, v are the size of trajectory X and control U over time horizon T, respectively

        :param listP: P over time horizon T (np array of shape (T, n, n)
        :param listr: r over time horizon T (np array of shape (T, m)
        :param bt: b vector over time horizon T (np array of shape (T, m)) - calculated from self.calc_b
        :return: zeta (tuple of z and v)
        """
        A, B = self.A, self.B
        z = np.zeros((np.shape(self.x_t)))
        v = np.zeros((np.shape(bt)))
        Rinv = np.linalg.inv(self.R)
        for i in range(len(bt)):
            P, r, b = listP[i], listr[i], bt[i]
            v[i] = -Rinv @ np.transpose(B) @ P @ z[i] - Rinv @ np.transpose(B) @ r - Rinv * b
            zdot = A @ z[i] + B @ v[i]
            z[i] += zdot * self.dt
        zeta = (z, v)
        return zeta

    def DJ(self, zeta, at, bt):
        """
        Finds Direction of steepest descent DJ given at and bt

        :param zeta: Tuple of descent directions (z, v)
        :param at: a vector over time horizon T (np array of shape (T, n)) - calculated from self.calc_at
        :param bt: b vector over time horizon T (np array of shape (T, m)) - calculated from self.calc_b
        :return: DJ value (float)
        """
        J = np.zeros((len(zeta)))
        z, v = zeta[0], zeta[1]
        for i in range(len(zeta[0])):
            a_T = np.transpose(at[i])
            b_T = np.transpose(bt[i])
            J_val = a_T @ z[i] + b_T @ v[i]
            J[i] = J_val
        DJ_integral = np.trapz(J, dx=self.dt)
        return DJ_integral

    def calc_P_r(self, at, bt):
        """
        Calculates P and r

        :param at:
        :param bt:
        :return:
        """
        P, A, B, Q = self.P, self.A, self.B, self.Q
        dim_len = len(at)
        listP, listr = np.zeros((dim_len, self.n, self.n)), np.zeros((dim_len, self.n, 1))
        listP[0] = self.P[:]
        listr[0] = self.r[:]
        Rinv = np.linalg.inv(self.R)
        for i in range(dim_len - 1):
            # difference in Todds lecture notes for Pdot
            P_dot = P @ (B @ Rinv @ np.transpose(B)) @ P - Q - P @ A + np.transpose(A) @ P
            a, b = np.transpose([at[i]]), bt[i]
            r_dot = - np.transpose(A - B @ Rinv @ np.transpose(B) @ P) @ listr[i] - a + (P @ B @ Rinv) * b
            listP[i + 1] = self.dt * P_dot + listP[i]
            listr[i + 1] = self.dt * r_dot + listr[i]

        return listP, listr

    def cart_pole_dyn(self, X, U):
        """
        Calculate Xdot (Change in X from timestep i-1 to i)

        Xdot is defined by the forward dynamics equations:
        Xdot = A @ X + B @ U

        :param X: Array of position vector X over time horizon T (np array of shape (T, n))
        :param U: Array of control vector X over time horizon T (np array of shape (T, m))
        :return: Array of velocity vector Xdot over time horizon T (np array of shape (T, n))
        """
        A, B = self.A, self.B
        Xdot = A @ X + B @ U
        return Xdot

    def integrate(self, xi, ui):
        """
        Finds the next state vector x using the Runge Kutta integral method

        :param xi:
        :param ui:
        :return:
        """
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
            bt[i] = self.u @ self.R
        return bt

    def __recursive_wrapper(self, K, k_arr, n, f):
        # When calling this function, call with K+1
        if n > 0:
            for k in range(K):
                self.__recursive_wrapper(K, k_arr + [k], n - 1, f)
        else:
            f(k_arr)

    @staticmethod
    def __k_str(k):
        return ''.join(str(i) for i in k)
