from .ergodic_measure import ErgodicMeasure
import numpy as np
import matplotlib.pyplot as plt


class Plot2DMetric:
    def __init__(self, D, E, K, L, dt, dim1, dim2, interpolation='none', vminmax=(-0.1, 0.5), sizexy=(15, 30)):
        """
        Plots Ergodic Spatial Metrix Phix using the Demonstrations and the ErgodicMeasure Class
        - Can also plot the Spatial Information Density of Trajectory as well as trajectory plots over state space

        :param D: List of demonstrations (list)
        :param E: Weights that describe whether a demonstration D[i] is good [1] or bad [-1] (list)
        :param K: Size of series coefficient (int)
        :param L: Size of boundaries for dimensions, listed as [Lower boundary, Higher Boundary] (list)
        :param dt: Time difference (float)
        :param dim1: Dimension 1 to visualize State space - column # (int)
        :param dim2: Dimension 2 to visualize State space - column # (int)
        :param interpolation: Interpolation for Ergodic Metric Visualization - matplotlib imshow interpolation (str)
        :param vminmax: Min/Max Value for matplotlib interpolation (vmin, vmax) (tuple)
        :param sizexy: Size of bins for x and y for visualize trajectory (dim1 bins, dim2 bins) (tuple)
        """
        self.K = K
        self.L = L

        self.dim1, self.dim2 = self.L[dim1], self.L[dim2]

        self.E = E
        D = np.array(D, dtype=object)
        self.D = [np.array(d)[:, [dim1, dim2]] for d in D]
        self.ergodic_measure = ErgodicMeasure(self.D, E, K, L, dt)
        self.phik_dict = self.ergodic_measure.get_phik(calc=True)

        self.Z = np.array([])

        self.interpolation = interpolation
        self.vminmax = vminmax
        self.sizexy = sizexy

    def visualize_ergodic(self):
        """
        Visualizes ergodic spatial metric Phix using inverse fourier transform of ergodic fourier metrix Phik

        :return: None
        """
        Z = self.__calc_phix()
        plt.imshow(Z, interpolation=self.interpolation, vmin=self.vminmax[0], vmax=self.vminmax[1],
                   extent=[self.dim1[0], self.dim1[1], self.dim2[0], self.dim2[1]], aspect='auto')
        plt.title('Ergodic Metric Spatial Distribution')
        plt.xlabel("dim 1")
        plt.ylabel("dim 2")
        plt.show()

    def visualize_trajectory(self, show_trajectory=True, show_information_density=True):
        """
        Creates a plot of all trajectory demonstrations

        - Good demonstrations are blue o, bad demonstrations are red +
        - Shows information content of trajectory, where positive values relate to positive demonstrations

        :param show_trajectory:
        :param show_information_density:
        :return: None
        """
        if show_trajectory:
            for i in range(len(self.D)):
                if self.E[i] == 1:
                    plt.plot(self.D[i][:, 0], self.D[i][:, 1], 'bo', markersize=0.2, linestyle='None')
                else:
                    plt.plot(self.D[i][:, 0], self.D[i][:, 1], 'r+', markersize=0.2, linestyle='None')

        if show_information_density:
            contour_count = self.__calculate_contour_count()

            bin_theta = np.linspace(self.dim1[0], self.dim1[1], self.sizexy[0])
            bin_thetadot = np.linspace(self.dim2[0], self.dim2[1], self.sizexy[1])

            plt.contourf(bin_theta, bin_thetadot, contour_count, 100, cmap='RdBu', vmin=-6, vmax=6)
        plt.title('Spatial information density and trajectories')
        plt.xlabel("dim 1")
        plt.ylabel("dim 2")
        plt.show()

    def __calc_phix_x(self, x):
        """
        Calculates Phix variable using inverse fourier transform for a given position state vector x

        Phix is defined by the function:
        Phix = 2 * Sum (phi_k * Fkx) for all k

        :param x: Position state vector x (np array of shape (1, 2))
        :return: Phi_x (float)
        """
        phix = 0
        for ki in range(self.K):
            for kj in range(self.K):
                k_str = f"{ki}{kj}"
                phik = self.phik_dict[k_str]
                fk = self.ergodic_measure.calc_Fk(x, [ki, kj])
                phix += phik * fk
        phix *= 2
        return phix

    def __calc_phix(self, axis_res=50):
        """
        Calculates Phix for all possible values of position state vector x within bounds L1 and L2, with resolution
        axis_res

        :param axis_res: Number of discrete values for plot axis x and y (int)
        :return: Ergodic Measure Phix for all coordinates of dimension 1 and 2 in bounds L1 and L2
        """
        x_axis = np.linspace(self.dim1[0], self.dim1[1], axis_res)
        y_axis = np.linspace(self.dim2[0], self.dim2[1], axis_res)
        z = np.array([self.__calc_phix_x([i, j]) for j in y_axis for i in x_axis])
        Z = z.reshape(axis_res, axis_res)
        return Z

    def __calculate_contour_count(self):
        """
        Calculates the information of all the trajectory demonstrations D using digitization of trajectory within
        buckets

        - Uses dimension 1 and 2 as bounds for bins, with self.sizexy[0] and self.sizexy[1] bins for each axis

        :return: contour_count - 2d information content in digitized buckets (np array of shape (self.sizexy[0], self.sizexy[1])
        """
        bin_theta = np.linspace(self.dim1[0], self.dim1[1], self.sizexy[0])
        bin_thetadot = np.linspace(self.dim2[0], self.dim2[1], self.sizexy[1])
        contour_count = np.zeros((self.sizexy[1] + 1, self.sizexy[0] + 1))

        for i in range(len(self.D)):
            digitize_theta = np.digitize(self.D[i][:, 0], bin_theta)
            digitize_thetadot = np.digitize(self.D[i][:, 1], bin_thetadot)
            trajec_len = len(self.D[i])
            for ii in digitize_theta:
                for jj in digitize_thetadot:
                    contour_count[jj][ii] += (1 / trajec_len) * self.E[i]

        for i, row in enumerate(contour_count):
            for j, val in enumerate(row):
                if val > 10:
                    contour_count[i, j] = np.log10(val)
                elif val < -10:
                    contour_count[i, j] = -1 * np.log10(abs(val))
        contour_count = contour_count[:-1, :-1]
        return contour_count
