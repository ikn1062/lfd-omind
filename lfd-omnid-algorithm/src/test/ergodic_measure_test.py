import numpy as np
from src.cartpole import ErgodicHelper


def test_init(D, E, K, L, dt):
    ergodic_test = ErgodicHelper(D, E, K, L, dt)
    n = ergodic_test.n
    assert n == 4, f"dimensions for input should be 4, got: {n}"
    m = ergodic_test.m
    assert m == 7, f"number of demonstrations for input should be 7, got: {n}"
    w = ergodic_test.w[0]
    assert abs(w - 1/7) < 0.1, f"weight of demonstrations for input should be 1/7, got: {w}"
    print("test_init pass")
    return 0


def test_calc_hk(D, E, K, L, dt):
    ergodic_test = ErgodicHelper(D, E, K, L, dt)
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


def test_calc_Fk(D, E, K, L, dt):
    ergodic_test_init = ErgodicHelper(D, E, K, L, dt)
    x_i = D[0][1900]
    # Comparing Fk to hand calculated values
    Fk = ergodic_test_init.calc_Fk(x_i, [1, 3, 1, 1])
    assert abs(Fk - -0.0061) < 0.0001, f"expected -0.0061, got: {Fk}"
    Fk = ergodic_test_init.calc_Fk(x_i, [2, 2, 4, 4])
    assert abs(Fk - 0.0015) < 0.0001, f"expected 0.0015, got: {Fk}"
    print("test_calc_Fk pass")
    return 0


def test_calc_ck(D, E, K, L, dt):
    # this is hard to hand calculate because we are calculating values over an entire trajectory
    # we're going to see if it prints a value, and will evaluate based off behavior of plots
    ergodic_test = ErgodicHelper(D, E, K, L, dt)
    x_t = D[0]
    ck = ergodic_test.calc_ck(x_t, [1, 2, 1, 2])
    assert ck, f"expected ck value, got: {ck}"
    ck = ergodic_test.calc_ck(x_t, [3, 5, 5, 3])
    assert ck, f"expected ck value, got: {ck}"
    ck = ergodic_test.calc_ck(x_t, [1, 1, 6, 6])
    assert ck, f"expected ck value, got: {ck}"
    print("test_calc_ck pass")
    return 0


def test_calc_phik(D, E, K, L, dt):
    # this is hard to hand calculate because we are calculating values over an entire trajectory
    # we're going to see if it prints a value, and will evaluate based off behavior of plots
    ergodic_test = ErgodicHelper(D, E, K, L, dt)
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


def test_lambda_k(D, E, K, L, dt):
    # this is hard to hand calculate because we are calculating values over an entire trajectory
    # we're going to see if it prints a value, and will evaluate based off behavior of plots
    ergodic_test = ErgodicHelper(D, E, K, L, dt)
    ergodic_test.calc_lambda_k([1, 1, 1, 1])
    ergodic_test.calc_lambda_k([1, 5, 2, 3])
    lambda_1111 = ergodic_test.lambdak_values["1111"]
    lambda_1523 = ergodic_test.lambdak_values["1523"]
    assert abs(lambda_1111 - 0.0178) < 0.001, f"expected lambdak value, got: {lambda_1111}"
    assert abs(lambda_1523 - 0.00009) < 0.0001, f"expected lambda value, got: {lambda_1523}"
    print("test_lambda_k pass")
    return 0


def test_ergodic_calc(D, E, K, L, dt):
    # this is hard to hand calculate because we are calculating values over an entire trajectory
    # we're going to see if it prints a value, and will evaluate based off behavior of plots
    print("Start Ergodic Calc test")
    ergodic_test = ErgodicHelper(D, E, K, L, dt)
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
    D = []
    for i in range(1, 7):
        D.append(np.genfromtxt(f'src/cartpole_gazebo/dynamics/test{i}.csv', delimiter=','))
    E, K, dt = [1, -1, -1, -1, -1, -1], 6, 0.01
    L = [[-np.pi, np.pi], [-11, 11], [-15, 15], [-15, 15]]
    test_init(D, E, K, L, dt)
    test_calc_hk(D, E, K, L, dt)
    test_calc_Fk(D, E, K, L, dt)
    test_calc_ck(D, E, K, L, dt)
    test_calc_phik(D, E, K, L, dt)
    test_lambda_k(D, E, K, L, dt)
    test_ergodic_calc(D, E, K, L, dt)
