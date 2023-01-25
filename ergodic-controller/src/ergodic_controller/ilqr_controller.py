import numpy as np
import matplotlib.pyplot as plt
from time import sleep


class iLQR:
    def __init__(self, x0, t0, tf, L, hk, phik, lambdak, A, B, u_init=1, dt=0.01, K=6, max_u=100):
        """
        iLQR Controller Class which generates controls for ergodic system in loop

        :param x0: Initial State of position in trajectory (np array of shape (1, 4))
        :param t0: Starting time of trajectory (int)
        :param tf: Finish time of trajectory (int)
        :param L: Size of boundaries for dimensions, listed as [Lower boundary, Higher Boundary] (list)
        :param hk: Normalizing factor for Fk - generated from ErgodicMeasure Class (dict)
        :param phik: Spatial Distribution of demonstrations - generated from ErgodicMeasure Class (dict)
        :param lambdak: Coefficient of Hilbert Space - generated from ErgodicMeasure Class (dict)
        :param u_init: Initial Control for time period (np array)
        :param A: A matrix for dynamic system (np array)
        :param B: B matrix for dynamic system (np array)
        :param dt: Time difference (float)
        :param K: Size of series coefficient (int)
        :param max_u: Max controller force applied to system (float)
        :param rospy_pub: Rospy Publisher Object (rospy.Publisher object)
        """
        # System variables
        self.n = len(x0)
        self.t0, self.tf = t0, tf
        self.dt = dt
        self.timespan = np.arange(t0, tf+dt, dt)

        # Dynamics
        self.A, self.B = A, B

        # Initialize x_t and u_t variables
        self.u_init = u_init
        self.max_u = max_u
        self.x0 = np.transpose(x0)
        self.u_t = self.u_init * np.ones((len(self.timespan), 1))
        self.x_t = self.make_trajectory(x0, self.u_t)

        # Control Constants
        self.K = K
        self.q = 1000
        self.R = np.array([[2]])
        self.Q = np.diag([0.1, 0.1, 2.0, 2.0])

        self.P = np.zeros((self.n, self.n))
        self.r = np.array([[0.] * self.n]).T

        self.L = L

        # Grad descent
        self.beta = 0.15
        self.eps = 0.01

        # Control Variables
        self.at = np.zeros((np.shape(self.x_t)))
        self.bt = np.zeros(np.shape(self.u_t))

        # Variable values as a function of k
        self.hk_values = hk
        self.phik_values = phik
        self.lambdak_values = lambdak

        # Lambda Helper function
        self.k_to_str = lambda k: ''.join(str(i) for i in k)

    def grad_descent(self, plot=True, plot_freq=15):
        """
        Uses iterative gradient descent to find optimal control at each time step
        - Compares spatial statistics of current trajectory to the spatial distributions of demonstrations

        - Creates plot of position state vector x over a period of time
        - Creates plot of control state vector u over a period of time

        :param plot: Whether to plot state vector x and control u over entire time period (bool)
        :param plot_freq: Plotting frequency for state and control vector plots (int)
        :return: None
        """
        dj = np.inf
        ii = 0

        while abs(dj) > self.eps:
            if plot and not ii % plot_freq:
                fig, axs = plt.subplots(4, 1)
                axs[0].plot(self.timespan, self.x_t[:, 0])
                axs[0].set_ylabel('x (m)')
                axs[1].plot(self.timespan, self.x_t[:, 1])
                axs[1].set_ylabel('x dot (m/s)')
                axs[2].plot(self.timespan, self.x_t[:, 2])
                axs[2].set_ylabel('Theta (rad)')
                axs[3].plot(self.timespan, self.x_t[:, 3])
                axs[3].set_ylabel('Theta dot (rad/s)')
                axs[3].set_xlabel('Time (s)')
                axs[0].set_title("State system over Time Horizon")
                plt.tight_layout()

                plt.show()

                plt.plot(self.timespan, self.u_t)
                plt.xlabel("Time (s)")
                plt.ylabel("Control Force (N)")
                plt.title("Control Applied over Time Horizon")
                plt.show()
            try:
                at, bt = self.calc_at(), self.calc_b()
                listP, listr = self.calc_P_r(at, bt)
                zeta = self.desc_dir(listP, listr, bt)
                dj = self.DJ(zeta, at, bt)
                print(f"DJ: {dj}")
            except Exception as e:
                print(f"Exception Occurred: {e}")
                break

            v = zeta[1]
            self.u_t = self.u_t + self.beta * v
            self.x_t = self.make_trajectory(self.x0, self.u_t)
            ii += 1

    def make_trajectory(self, x0, u_t, circular=True):
        """
        Creates a trajectory of x from a given initial state, xt, and control input u
        - The output is a trajectory over the time horizon T of the input control

        :param x0: Initial position vector (np array of shape (1, n))
        :param u_t: Control input (np array of shape (T, m)
        :param circular: Property to define circular nature of a given
        :return: A trajectory of x over time horizon T of the input control (np array of shape (T, n))

        """
        N = len(self.timespan)
        x_traj = np.zeros((N, len(x0)))
        x_traj[0, :] = np.transpose(x0[:])
        for i in range(1, N):
            xi_new = self.__integrate(x_traj[i - 1, :], u_t[i - 1])
            if circular:
                while xi_new[2] > np.pi:
                    xi_new[2] -= 2 * np.pi
                while xi_new[2] < -np.pi:
                    xi_new[2] += 2 * np.pi
            x_traj[i, :] = np.transpose(xi_new)
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
        v = np.zeros((np.shape(self.u_t)))
        Rinv = np.linalg.inv(self.R)
        for i in range(1, len(self.timespan)):
            P, r, b = listP[i], listr[i], bt[i]
            v[i] = -Rinv @ np.transpose(B) @ P @ z[i-1] - Rinv @ np.transpose(B) @ r - Rinv * b
            zdot = A @ z[i-1] + B @ v[i]
            z[i] = z[i-1] + zdot * self.dt
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
        J = np.zeros((len(self.timespan)))
        z, v = zeta[0], zeta[1]
        for i in range(len(self.timespan)):
            a_T = np.transpose(at[i])
            b_T = np.transpose(bt[i])
            J_val = a_T @ z[i] + b_T @ v[i]
            J[i] = J_val
        DJ_integral = np.trapz(J, dx=self.dt)
        return DJ_integral

    def calc_P_r(self, at, bt):
        """
        Calculates P and r used to solve the Ricatti Equations

        :param at: a vector over time horizon T (np array of shape (T, n)) - calculated from self.calc_at
        :param bt: b vector over time horizon T (np array of shape (T, m)) - calculated from self.calc_b
        :return: list of P matrix and r matrix over the time horizon (tuple)
        """
        A, B, Q = self.A, self.B, self.Q
        listP, listr = np.zeros((len(self.timespan), self.n, self.n)), np.zeros((len(self.timespan), self.n, 1))
        listP[0], listr[0] = self.P[:], self.r[:]
        Rinv = np.linalg.inv(self.R)

        for i in range(1, len(self.timespan)):
            P, r = listP[i-1], listr[i-1]
            a, b = np.transpose([at[i]]), [bt[i]]

            P_dot = P @ B @ Rinv @ np.transpose(B) @ P - Q - P @ A - np.transpose(A) @ P
            r_dot = - np.transpose(A - B @ Rinv @ np.transpose(B) @ P) @ r - a + P @ B @ Rinv @ b

            listP[i] = self.dt * P_dot + listP[i - 1]
            listr[i] = self.dt * r_dot + listr[i - 1]

        return listP, listr

    def dynamics(self, X, U):
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

    def __integrate(self, xi, ui):
        """
        Finds the next state vector x using the Runge Kutta integral method

        :param xi: State position vector x (np array with shape (1, n))
        :param ui: State control vector u (np array with shape (1, m))
        :return: State position vector x at next time step (np array with shape (1, n))
        """
        k1 = self.dt * self.dynamics(xi, ui)
        k2 = self.dt * self.dynamics(xi + k1 / 2, ui)
        k3 = self.dt * self.dynamics(xi + k2 / 2, ui)
        k4 = self.dt * self.dynamics(xi + k3, ui)
        x_i_next = xi + (1/6 * (k1 + 2 * k2 + 2 * k3 + k4))
        return x_i_next

    def __calc_Fk(self, xt, k):
        """
        Helper function to calculate normalized fourier coeffecient using basis function metric

        Fk is defined by the following:
        Fk = 1/hk * product(cos(k[i] *x[i])) where i ranges for all dimensions of x
        - Where k[i] = (K[i] * pi) / L[i]
        - Where L[i] is the bounds of the variable dimension i

        :param x: Position vector x (np array of shape (1, n))
        :param k: The series coefficient given as a list of length dimensions (list)
        :return: Fk Value (float)
        """
        fourier_basis = 1
        for i in range(self.n):
            fourier_basis *= np.cos((k[i] * np.pi * xt[i]) / (self.L[i][1] - self.L[i][0]))
        hk = self.hk_values[self.k_to_str(k)]
        Fk = (1 / hk) * fourier_basis
        return Fk

    def calc_DFk(self, k):
        """
        Calculates directional derivative of fourier coeffecient using basis function metric

        DFk is defined by the following:
        DFk = 1/hk * product(-k * np.cos(ki * x) * np.sin(k * x))
        - Where k = (K * pi) / L
        - Where L is the bounds of the variable dimensions

        :param k: The series coefficient given as a list of length dimensions (list)
        :return: DFk value (np array of size (1, n))
        """
        x_t = self.x_t
        hk = self.hk_values[self.k_to_str(k)]
        dfk = np.zeros(np.shape(x_t))
        for t, x in enumerate(x_t):
            for i in range(len(x)):
                ki = (k[i] * np.pi) / (self.L[i][1] - self.L[i][0])
                dfk_xi = (1 / hk) * - ki * np.cos(ki * x[i]) * np.sin(ki * x[i])
                dfk[t, i] = dfk_xi
        return dfk

    def calc_ck(self, k):
        """
        Calculates spacial statistics for a given trajectory and series coefficient value

        ck is given by:
        ck = integral Fk(x(t)) dt from t=0 to t=T
        - where x(t) is a trajectory, mapping t to position vector x

        :param k: The series coefficient given as a list of length dimensions (list)
        :return: ck value (float)
        """
        x_t = self.x_t
        Fk_x = np.zeros(len(self.timespan))
        for i in range(len(self.timespan)):
            Fk_x[i] = self.__calc_Fk(x_t[i], k)
        ck = (1 / self.tf) * np.trapz(Fk_x, dx=self.dt)
        return ck

    def calc_at(self):
        """
        Calculates coefficient at for solving ricatti equations

        at is calculated using helper function self.__calc_a

        :return: at coefficients (np array of shape (T, n)) - same shape as x
        """
        self.at = np.zeros((np.shape(self.x_t)))
        self.__recursive_wrapper(self.K + 1, [], self.n, self.__calc_a)
        self.at *= self.q
        return self.at

    def __calc_a(self, k):
        """
        Calculates coefficient a for solving ricatti equations

        a is defined by the equation below:
        a_k = ((lambda_k * 2 * (c_k - phi_k) * (1 / self.tf)) * DF_k) + self.at

        :param k: The series coefficient given as a list of length dimensions (list)
        :return: at coefficients for a given k (np array of shape (T, n)) - same shape as x
        """
        k_str = self.k_to_str(k)
        lambdak, phik = self.lambdak_values[k_str], self.phik_values[k_str]
        ck = self.calc_ck(k)
        self.at = (lambdak * 2 * (ck - phik) * (1 / self.tf) * self.calc_DFk(k)) + self.at

    def calc_b(self):
        """
        Calculates coefficient b for solving ricatti equations

        b is defined by the equation below:
        b = transpose(u) @ R

        :param k: The series coefficient given as a list of length dimensions (list)
        :return: at coefficients for a given k (np array of shape (T, n)) - same shape as x
        """
        bt = np.zeros(np.shape(self.u_t))
        for i, u in enumerate(self.u_t):
            bt[i] = np.transpose(self.u_t[i]) @ self.R
        return bt

    def __recursive_wrapper(self, K, k_arr, n, f):
        """
        Recurrsive wrapper allowing for to calculate various permuations of K

        :param K: K Value - Needs to be passed as K+1 (int)
        :param k_arr: array of traversed K values (list)
        :param n: count of dimensions left to iterate through (int)
        :param f: function f to call with k_arr (function)
        :return:
        """
        if n > 0:
            for k in range(K):
                self.__recursive_wrapper(K, k_arr + [k], n - 1, f)
        else:
            f(k_arr)

    # @staticmethod
    # def __k_str(k):
    #     # make this a lambda function??
    #     """
    #     Takes k arr and returns a string
    #     :param k: The series coefficient given as a list of length dimensions (list)
    #     :return: Series coefficient as a string (str)
    #     """
    #     return ''.join(str(i) for i in k)

