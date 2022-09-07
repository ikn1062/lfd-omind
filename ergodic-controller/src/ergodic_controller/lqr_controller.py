import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are as lqr


class LQR:
    def __init__(self, x0, t0, tf, dt, A, B):
        self.x0 = x0
        self.t0, self.tf, self.dt = t0, tf, dt
        self.timespan = np.arange(t0, tf+dt, dt)

        self.n = len(x0)

        self.A = A
        self.B = B

        Q = np.diag([0.1, 0.1, 2.0, 2.0])
        R = np.eye(1, dtype=int)
        P = lqr(A, B, Q, R)
        self.K = np.dot(np.linalg.inv(R), np.dot(B.T, P))

    def controller(self, show_plot=False):
        t0, tf, dt = self.t0, self.tf, self.dt
        x = self.x0

        x_trajec = np.zeros((len(self.timespan), self.n))
        u_control = np.zeros((len(self.timespan), np.shape(self.B)[1]))
        ii = 0

        while t0 < tf:
            u = -np.dot(self.K, x)

            x_trajec[ii, :] = x
            u_control[ii, :] = u

            x = self.__integrate(x, u)
            t0 += dt
            ii += 1

        if show_plot:
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].plot(self.timespan, x_trajec[:, 0])
            axs[0, 1].plot(self.timespan, x_trajec[:, 1])
            axs[1, 0].plot(self.timespan, x_trajec[:, 2])
            axs[1, 1].plot(self.timespan, x_trajec[:, 3])
            plt.show()

            plt.plot(self.timespan, u_control)
            plt.show()

    def dynamics(self, X, U):
        """
        Calculate Xdot (Change in X from previous to next timestep)

        Xdot is defined by the forward dynamics equations:
        Xdot = A @ X + B @ U

        :param X: Array of position vector X (np array of shape (1, n))
        :param U: Array of control vector X (np array of shape (1, m))
        :return: Array of velocity vector Xdot (np array of shape (1, n))
        """
        A, B = self.A, self.B
        Xdot = A @ X + B @ U
        return Xdot

    def __integrate(self, x, u):
        """
        Finds the next state vector x using the Runge Kutta integral method

        :param x: State position vector x (np array with shape (1, n))
        :param u: State control vector u (np array with shape (1, m))
        :return: State position vector x at next time step (np array with shape (1, n))
        """
        k1 = self.dt * self.dynamics(x, u)
        k2 = self.dt * self.dynamics(x + k1 / 2, u)
        k3 = self.dt * self.dynamics(x + k2 / 2, u)
        k4 = self.dt * self.dynamics(x + k3, u)
        x_next = x + (1/6 * (k1 + 2 * k2 + 2 * k3 + k4))
        return x_next
