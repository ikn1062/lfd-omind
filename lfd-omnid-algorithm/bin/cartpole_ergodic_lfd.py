import numpy as np
from src.cartpole import ErgodicHelper, MPC


def main():
    print("Getting Demonstrations")
    num_trajectories = 13

    # demonstration_list = [0, 6, 7, 10, 11, 12]
    demonstration_list = [0]

    D = []
    # E, new_E = [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1], []
    E, new_E = [1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1], []
    for i in range(num_trajectories):
        if i not in demonstration_list:
            continue
        new_E.append(E)
        demonstration = np.genfromtxt(f'src/cartpole_gazebo/dynamics/test{i}.csv', delimiter=',')
        demonstration[:, 0] = np.pi - np.abs(demonstration[:, 0])
        demonstration = np.hstack((demonstration[:, 2:], demonstration[:, :2]))
        D.append(demonstration)

    print("Calculating Ergodic Helpers")

    K = 2
    dt = 0.01
    L = [[-15, 15], [-15, 15], [-np.pi, np.pi], [-11, 11]]
    ergodic_test = ErgodicHelper(D, E, K, L, dt)
    hk, lambdak, phik = ergodic_test.calc_fourier_metrics()

    print("Starting Grad Descent")

    x0 = [0, 0, 0, 0]
    t0, tf = 0, 15

    mpc_model_1 = MPC(x0, t0, tf, L, hk, phik, lambdak, dt=dt, K=K)
    mpc_model_1.grad_descent()


if __name__ == "__main__":
    main()
