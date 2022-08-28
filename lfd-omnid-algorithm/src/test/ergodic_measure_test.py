import numpy as np
from src.ergodic_controller import ErgodicMeasure


class TestErgodicMeasure:
    def __init__(self):
        self.D = []
        for i in range(1, 7):
            self.D.append(np.genfromtxt(f'src/cartpole_gazebo/dynamics/test{i}.csv', delimiter=','))
        self.E, self.K, self.dt = [1, -1, -1, -1, -1, -1], 6, 0.01
        self.L = [[-np.pi, np.pi], [-11, 11], [-15, 15], [-15, 15]]

    def run_tests(self):
        self.__test_init()
        # self.__test_calc_hk()
        self.__test_calc_Fk()
        # self.__test_calc_ck()
        self.__test_calc_phik()
        self.__test_lambda_k()
        self.__test_ergodic_measure()

    def __test_init(self):
        """
        Tests the initialization of functions in the ErgodicMeasure Class
        :return: 0 (if test pass)
        """
        D, E, K, L, dt = self.D, self.E, self.K, self.L, self.dt
        ergodic_test = ErgodicMeasure(D, E, K, L, dt)
        n = ergodic_test.n
        assert n == 4, f"dimensions for input should be 4, got: {n}"
        m = ergodic_test.m
        assert m == 7, f"number of demonstrations for input should be 7, got: {n}"
        w = ergodic_test.w[0]
        assert abs(w - 1 / 7) < 0.1, f"weight of demonstrations for input should be 1/7, got: {w}"
        print("test_init pass")
        return 0

    def __test_calc_hk(self):
        """
        Tests the clac_hk method in the ErgodicMeasure class, compares the results to the math derivation

        Note: Test is currently deprecated as calc_hk is a private method

        :return: 0 (if test pass)
        """
        D, E, K, L, dt = self.D, self.E, self.K, self.L, self.dt
        ergodic_test = ErgodicMeasure(D, E, K, L, dt)
        hk = ergodic_test.calc_hk([0, 0, 0, 0])
        assert abs(hk - 352.7) < 0.1, f"expected 352.7, got: {hk}"
        # Based on how hk is defined, it is a constant dependent on L
        hk = ergodic_test.calc_hk([1, 1, 1, 1])
        assert abs(hk - 88.17) < 0.1, f"expected 7775, got: {hk}"
        hk = ergodic_test.calc_hk([4, 5, 4, 5])
        assert abs(hk - 88.17) < 0.1, f"expected 7775, got: {hk}"
        hk = ergodic_test.calc_hk([3, 2, 1, 6])
        assert abs(hk - 88.17) < 0.1, f"expected 7775, got: {hk}"
        hk = ergodic_test.calc_hk([1, 1, 2, 2])
        assert abs(hk - 88.17) < 0.1, f"expected 7775, got: {hk}"
        hk = ergodic_test.calc_hk([1, 3, 3, 1])
        assert abs(hk - 88.17) < 0.1, f"expected 7775, got: {hk}"
        print("test_calc_hk pass")
        return 0

    def __test_calc_Fk(self):
        """
        Tests the clac_Fk method in the ErgodicMeasure class, compares the results to the math derivation

        :return: 0 (if test pass)
        """
        D, E, K, L, dt = self.D, self.E, self.K, self.L, self.dt
        ergodic_test_init = ErgodicMeasure(D, E, K, L, dt)
        x_i = D[0][1900]
        # Comparing Fk to hand calculated values
        Fk = ergodic_test_init.calc_Fk(x_i, [1, 3, 1, 1])
        assert abs(Fk - -0.0061) < 0.0001, f"expected -0.0061, got: {Fk}"
        Fk = ergodic_test_init.calc_Fk(x_i, [2, 2, 4, 4])
        assert abs(Fk - 0.0015) < 0.0001, f"expected 0.0015, got: {Fk}"
        print("test_calc_Fk pass")
        return 0

    def __test_calc_ck(self):
        """
        Tests the clac_ck method in the ErgodicMeasure class, compares the results to the math derivation

        Note: Test is currently deprecated as calc_ck is a private method
        Note: It is hard to compare ck values as it takes an input as an entire trajectory, this test will check for
        values and will be evaluated based on plots

        :return: 0 (if test pass)
        """
        D, E, K, L, dt = self.D, self.E, self.K, self.L, self.dt
        ergodic_test = ErgodicMeasure(D, E, K, L, dt)
        x_t = D[0]
        ck = ergodic_test.calc_ck(x_t, [1, 2, 1, 2])
        assert ck, f"expected ck value, got: {ck}"
        ck = ergodic_test.calc_ck(x_t, [3, 5, 5, 3])
        assert ck, f"expected ck value, got: {ck}"
        ck = ergodic_test.calc_ck(x_t, [1, 1, 6, 6])
        assert ck, f"expected ck value, got: {ck}"
        print("test_calc_ck pass")
        return 0

    def __test_calc_phik(self):
        """
        Tests the clac_ck method in the ErgodicMeasure class, compares the results to the math derivation

        Note: Test is currently deprecated as calc_Fk is a private method
        Note: It is hard to compare ck values as it takes an input as an entire trajectory, this test will check for
        values and will be evaluated based on plots

        :return: 0 (if test pass)
        """
        D, E, K, L, dt = self.D, self.E, self.K, self.L, self.dt
        ergodic_test = ErgodicMeasure(D, E, K, L, dt)
        ergodic_test.calc_phik([1, 2, 3, 4])
        ergodic_test.calc_phik([2, 1, 5, 1])
        ergodic_test.calc_phik([1, 5, 5, 5])
        val_1234 = ergodic_test.phik_values["1234"]
        val_2151 = ergodic_test.phik_values["2151"]
        val_1555 = ergodic_test.phik_values["2151"]
        assert val_1234, f"expected phik value, got: {val_1234}"
        assert val_2151, f"expected phik value, got: {val_2151}"
        assert val_1555, f"expected phik value, got: {val_1555}"
        print("test_calc_phik pass")
        return 0

    def __test_lambda_k(self):
        """
        Tests the calc_lambda_k method in the ErgodicMeasure class, compares the results to the math derivation

        :return: 0 (if test pass)
        """
        D, E, K, L, dt = self.D, self.E, self.K, self.L, self.dt
        ergodic_test = ErgodicMeasure(D, E, K, L, dt)
        ergodic_test.calc_lambda_k([1, 1, 1, 1])
        ergodic_test.calc_lambda_k([1, 5, 2, 3])
        lambda_1111 = ergodic_test.lambdak_values["1111"]
        lambda_1523 = ergodic_test.lambdak_values["1523"]
        assert abs(lambda_1111 - 0.0178) < 0.001, f"expected lambdak value, got: {lambda_1111}"
        assert abs(lambda_1523 - 0.00009) < 0.0001, f"expected lambda value, got: {lambda_1523}"
        print("test_lambda_k pass")
        return 0

    def __test_ergodic_measure(self):
        """
        Tests the outut of the ErgodicMeasure class, compares the results to the math derivation

        Note: It is hard to compare output values as it takes an input as an entire trajectory, this test will check for
        values and will be evaluated based on plots

        :return: 0 (if test pass)
        """
        D, E, K, L, dt = self.D, self.E, self.K, self.L, self.dt
        ergodic_test = ErgodicMeasure(D, E, K, L, dt)
        hk, lambdak, phik = ergodic_test.calc_fourier_metrics()
        assert hk["1234"], f"expected hk value, got: {None}"
        assert hk["3322"], f"expected hk value, got: {None}"
        assert lambdak["2131"], f"expected hk value, got: {None}"
        assert lambdak["2344"], f"expected hk value, got: {None}"
        assert phik["4431"], f"expected hk value, got: {None}"
        assert phik["1422"], f"expected hk value, got: {None}"
        print("test_ergodic_calc pass")
        return 0


if __name__ == "__main__":
    test_ergodic_measure = TestErgodicMeasure
    TestErgodicMeasure.run_tests()
