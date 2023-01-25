import numpy as np


class ErgodicMeasure:
    def __init__(self, D, E, K, L, dt):
        """
        Ergodic Helper Class to calculate the following metrics:
        - Phi_k: Spatial Distribution of demonstrations
        - h_k: Normalizing factor for Fk
        - Lambda_k: Coefficient of Hilbert Space placing higher importance on lower freq information

        :param D: List of demonstrations (list)
        :param E: Weights that describe whether a demonstration D[i] is good [1] or bad [-1] (list)
        :param K: Size of series coefficient (int)
        :param L: Size of boundaries for dimensions, listed as [Lower boundary, Higher Boundary] (list)
        :param dt: Time difference (float)
        """
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
        """
        Calculates Phi_k, lambda_k, and h_k values based on demonstrations
        :return: (hk_value dictionary (dict), phi_k value dictionary (dict), h_k value dictionary (dict))
        """
        self.__recursive_wrapper(self.K + 1, [], self.n, self.calc_phik)
        self.__recursive_wrapper(self.K + 1, [], self.n, self.calc_lambda_k)
        return self.hk_values, self.phik_values, self.lambdak_values

    def calc_phik(self, k):
        """
        Calculates coefficients that describe the task definition, phi_k
        - Spatial distribution of demonstrations

        phi_k is defined by the following:
        phi_k = sum w_j * c_k_j where j ranges from 1 to num_trajectories
        - w_j is initialized as 1/num_trajectories, the weighting of each trajectory in the spatial coefficient

        - Sets self.phi_k value in dictionary using k as a string
        :param k: k: The series coefficient given as a list of length dimensions (list)
        :return: None
        """
        phik = 0
        for i in range(self.m):
            phik += self.E[i] * self.w[i] * self.__calc_ck(self.D[i], k)
        k_str = ''.join(str(i) for i in k)
        self.phik_values[k_str] = phik

    def calc_lambda_k(self, k):
        """
        Calculate lambda_k places larger weights on lower coefficients of information

        lambda_k is defined by the following:
        lambda_k = (1 + ||k||2) − s where s = n+1/2

        - Sets self.lambda_k value in dictionary using k as a string
        :param k: The series coefficient given as a list of length dimensions (list)
        :return: None
        """
        s = (self.n + 1) / 2
        lamnbda_k = 1 / ((1 + np.linalg.norm(k) ** 2) ** s)
        k_str = ''.join(str(i) for i in k)
        self.lambdak_values[k_str] = lamnbda_k

    def __calc_ck(self, x_t, k):
        """
        Calculates spacial statistics for a given trajectory and series coefficient value

        ck is given by:
        ck = integral Fk(x(t)) dt from t=0 to t=T
        - where x(t) is a trajectory, mapping t to position vector x

        :param x_t: x(t) function, mapping position vectors over a period of time (np array)
        :param k: k: The series coefficient given as a list of length dimensions (list)
        :return: ck value (float)
        """
        x_len = len(x_t)
        T = x_len * self.dt
        Fk_x = np.zeros(x_len)
        for i in range(x_len):
            Fk_x[i] = self.calc_Fk(x_t[i], k)
        ck = (1 / T) * np.trapz(Fk_x, dx=self.dt)
        return ck

    def calc_Fk(self, x, k):
        """
        Calculates normalized fourier coeffecient using basis function metric

        Fk is defined by the following:
        Fk = 1/hk * product(cos(k[i] *x[i])) where i ranges for all dimensions of x
        - Where k[i] = (K[i] * pi) / L[i]
        - Where L[i] is the bounds of the variable dimension i

        :param x: Position vector x (np array)
        :param k: The series coefficient given as a list of length dimensions (list)
        :return: Fk Value (float)
        """
        hk = self.__calc_hk(k)
        fourier_basis = 1
        for i in range(len(x)):
            fourier_basis *= np.cos((k[i]*np.pi*x[i])/(self.L[i][1] - self.L[i][0]))
        Fk = (1/hk)*fourier_basis
        return Fk

    def __calc_hk(self, k):
        """
        Normalizing factor for Fk

        hk is defined as:
        hk = Integral cos^2(k[i] * x[i]) dx from L[i][0] to L[i][1]

        :param k: The series coefficient given as a list of length dimensions (list)
        :return: hk value (float)
        """
        hk = 1
        for i in range(self.n):
            l0, l1 = self.L[i][0], self.L[i][1]
            if not k[i]:  # if k[i] is 0, we continue to avoid a divide by 0 error
                hk *= (l1 - l0)
                continue
            k_i = (k[i] * np.pi) / l1
            hk *= (2 * k_i * (l1 - l0) - np.sin(2 * k_i * l0) + np.sin(2 * k_i * l1)) / (4 * k_i)
        k_str = ''.join(str(i) for i in k)
        hk = np.sqrt(hk)
        self.hk_values[k_str] = hk
        return hk

    def __recursive_wrapper(self, K, k_arr, count, f):
        """
        Recurrsive wrapper allowing for to calculate various permuations of K

        :param K: K Value - Needs to be passed as K+1 (int)
        :param k_arr: array of traversed K values (list)
        :param n: count of dimensions left to iterate through (int)
        :param f: function f to call with k_arr (function)
        :return:
        """
        if count > 0:
            for k in range(K):
                self.__recursive_wrapper(K, k_arr + [k], count - 1, f)
        else:
            print(k_arr)
            f(k_arr)

    def get_phik(self, calc=False):
        if calc:
            self.__recursive_wrapper(self.K + 1, [], self.n, self.calc_phik)
        return self.phik_values

    def get_hk(self):
        return self.hk_values

    def get_lambdak_values(self, calc=False):
        if calc:
            self.__recursive_wrapper(self.K + 1, [], self.n, self.calc_lambda_k)
        return self.lambdak_values
