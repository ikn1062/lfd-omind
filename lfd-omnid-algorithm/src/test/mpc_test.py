import numpy as np
from src.cartpole import ErgodicHelper, MPC


def test_mpc_init(mpc):
    print("start test_mpc_init")
    n = mpc.n
    assert n == 4, f"dimensions for input should be 4, got: {n}"
    at, bt = mpc.at, mpc.bt
    assert len(at) == 4, f"dimensions for at should be 4, got: {len(at)}"
    assert bt == 0, f"dimensions for bt should be 4, got: {bt}"
    # lambdak = mpc.lambdak_values
    # assert lambdak["1234"] == lambdak_in["1234"], f"dimensions for lambdak should be the same, got: {lambdak['1234']}"
    print("test_mpc_init pass")


def test_dynamics_1(mpc):
    print("start test_dynamics_1")
    a, b, c, d = mpc.dynamics()
    assert abs(8.408 - a) < 0.001, f"dynamics variable a is wrong, got: {a}"
    assert abs(-0.840 - b) < 0.001, f"dynamics variable a is wrong, got: {b}"
    assert abs(0.05 - c) < 0.01, f"dynamics variable a is wrong, got: {c}"
    assert abs(-0.0428 - d) < 0.001, f"dynamics variable a is wrong, got: {d}"
    A, B = mpc.A, mpc.B
    A_check = np.array([[0, 1, 0, 0], [0, 0, 8.408, 0], [0, 0, 0, 1], [0, 0, 0.8408, 0]])
    B_check = np.array([[0], [0.05], [0], [-0.0428]])
    A_eps = 0.01*np.ones((np.shape(A)))
    B_eps = 0.01 * np.ones((np.shape(B)))
    assert (abs(A - A_check) < A_eps).all, f"Matrix A is wrong, got: {A}"
    assert (abs(B - B_check) < B_eps).all, f"Matrix B is wrong, got: {B}"
    print("test_dynamics_1 pass")


def test_dynamics_2(mpc):
    print("start test_dynamics_2")
    x = np.array([[1], [0], [3.14], [0]])
    u = 32
    A = np.array([[0, 1, 0, 0], [0, 0, 8.408, 0], [0, 0, 0, 1], [0, 0, 0.8408, 0]])
    B = np.array([[0], [0.05], [0], [-0.0428]])
    x_dot = A@x + B*u
    mpc_x_dot = mpc.cart_pole_dyn(x, u)
    x_dot_eps = 0.01 * np.ones((np.shape(x)))
    assert (abs(x_dot - mpc_x_dot) < x_dot_eps).all, f"Matrix B is wrong, got: {mpc_x_dot}"

    dt = 0.1
    x_new = x + x_dot*dt
    mpc_x_new = mpc.integrate(x, u)
    assert (abs(x_new - mpc_x_new) < x_dot_eps).all, f"Matrix B is wrong, got: {mpc_x_new}"
    print("test_dynamics_2 pass")


def test_ck_DFk(mpc):
    print("start test_ck_DFk")
    mpc.x_t = np.array([[1], [0], [3.14], [0]])
    k = np.array([0, 2, 2, 1])
    mpc_dfk = mpc.calc_DFk(k)
    assert len(mpc_dfk) == 4, f"dfk should be of size xt, got {len(mpc_dfk)}"
    assert mpc_dfk.any(), f"dfk should return value, got {mpc_dfk}"

    ck = mpc.calc_ck(k)
    assert ck, f"ck should return value, got {ck}"
    print("test_ck_DFk pass")


def test_at_bt(mpc):
    print("start test_at_bt")
    mpc.x_t = np.array([[1], [0], [3.14], [0]])

    at = mpc.calc_at()
    assert len(at) == 4, f"at should be of size xt, got {len(at)}"
    assert at.any(), f"at should return value, got {at}"

    mpc.u = 32
    bt = mpc.calc_b()
    u, R = 32, 2
    utR = (-1/u)*R
    utR_eps = 0.01
    assert bt, f"at should return value, got {bt}"
    assert abs(utR - bt) < utR_eps, f"Matrix bt is wrong, got: {bt}"
    print("test_at_bt pass")


def test_desc_dir(mpc):
    print("start test_P_r")
    mpc.x_t, mpc.u = np.array([[1], [0], [3.14], [0]]), 32
    at, bt = mpc.calc_at(), mpc.calc_b()
    P, r = mpc.calc_P_r(at, bt)
    assert np.shape(P) == (4, 4), f"P should have shape 4,4, but got {np.shape(P)}"
    assert np.shape(r) == (4, 1), f"r should have shape 4,1, but got {np.shape(r)}"

    zeta = mpc.desc_dir(P, r, bt)
    assert np.shape(zeta[0]) == (4, 1), f"z should have shape 4,1, but got {np.shape(zeta[0])}"
    assert np.shape(zeta[1]) == (1, 1), f"v should have shape 1,1 but got {np.shape(zeta[1])}"

    J = mpc.DJ(zeta, at, bt)
    assert np.shape(J) == (1, 1), f"J should have shape 1,1 but got {np.shape(J)}"
    print("test_P_r pass")


if __name__ == "__main__":
    print("Start test")
    L = [[-np.pi, np.pi], [-11, 11], [-15, 15], [-15, 15]]
    hk, phik, lambdak = {}, {}, {}

    print("Get MPC model - no ergodic variables")
    x0 = [-np.pi, 0, 0, 0]
    t0, tf = 0, 30
    mpc_model = MPC(x0, t0, tf, L, hk, phik, lambdak)


    print("Start MPC test")
    test_mpc_init(mpc_model)
    test_dynamics_1(mpc_model)
    test_dynamics_2(mpc_model)


    D = []
    for i in range(1, 7):
        D.append(np.genfromtxt(f'src/cartpole_gazebo/dynamics/test{i}.csv', delimiter=','))
    E, K, dt = [1, -1, -1, -1, -1, -1], 2, 0.01
    # E, K, dt = [1, -1, -1, -1, -1, -1], 6, 0.01
    ergodic_test = ErgodicHelper(D, E, K, L, dt)
    print("Getting Ergodic Helpers")
    hk, lambdak, phik = ergodic_test.calc_fourier_metrics()
    mpc_model_1 = MPC(x0, t0, tf, L, hk, phik, lambdak, dt=dt, K=K)


    test_ck_DFk(mpc_model_1)
    test_at_bt(mpc_model_1)
    test_desc_dir(mpc_model_1)



