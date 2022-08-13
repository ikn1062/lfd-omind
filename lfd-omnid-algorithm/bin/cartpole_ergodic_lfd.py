import numpy as np
from src.cartpole import ErgodicHelper, MPC


def main():
    print("Calculate Ergodic Helpers")

    D = []
    for i in range(1, 10):
        demonstration = np.genfromtxt(f'src/cartpole_gazebo/dynamics/test{i}.csv', delimiter=',')
        demonstration = np.hstack((demonstration[:, 2:], demonstration[:, :2]))
        D.append(demonstration)
    E, K, dt = [1, -1, -1, -1, -1, -1, 1, -1, -1], 6, 0.01
    L = [[-15, 15], [-15, 15], [-np.pi, np.pi], [-11, 11]]
    ergodic_test = ErgodicHelper(D, E, K, L, dt)
    hk, lambdak, phik = ergodic_test.calc_fourier_metrics()

    print("Starting Grad Descent")

    x0 = [0, 0, -np.pi, 0]
    t0, tf = 0, 15

    mpc_model_1 = MPC(x0, t0, tf, L, hk, phik, lambdak, dt=dt, K=K)
    mpc_model_1.grad_descent()


if __name__ == "__main__":
    main()
