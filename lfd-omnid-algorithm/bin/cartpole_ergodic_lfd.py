import numpy as np
from src.ergodic_controller import ErgodicMeasure, controlleriLQR, Plot2DMetric


def main():
    print("Getting Demonstrations")
    num_trajectories = 14

    demonstration_list = [13]

    D = []
    E, new_E = [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1], []
    for i in range(num_trajectories):
        if i not in demonstration_list:
            continue
        new_E.append(E)
        demonstration = np.genfromtxt(f'src/cartpole_gazebo/dynamics/test{i}.csv', delimiter=',')
        demonstration[:, 0] = np.pi - np.abs(demonstration[:, 0])
        demonstration = np.hstack((demonstration[:, 2:], demonstration[:, :2]))
        D.append(demonstration)

    K = 6
    dt = 1
    L = [[-15, 15], [-15, 15], [-np.pi, np.pi], [-11, 11]]

    print("Visualize Ergodic Metric")
    plot_phix_metric = Plot2DMetric(K, L, dt, E, D, 0, 1)
    plot_phix_metric.visualize_ergodic()
    plot_phix_metric.visualize_trajectory()

    print("Calculating Ergodic Helpers")

    ergodic_test = ErgodicMeasure(D, E, K, L, dt)
    hk, lambdak, phik = ergodic_test.calc_fourier_metrics()

    M, m, l = 20, 20, 0.75
    A, B = cartpole_AB_matrix(M, m, l)
    print("Starting Grad Descent")

    x0 = [0, 0, 0, 0]
    t0, tf = 0, 15

    mpc_model_1 = controlleriLQR(x0, t0, tf, L, hk, phik, lambdak, A, B, dt=dt, K=K)
    # mpc_model_1 = controllerLQR(x0, t0, tf, L, hk, phik, lambdak, A, B, dt=dt, K=K)
    mpc_model_1.grad_descent()


def cartpole_AB_matrix(M, m, l):
    g = 9.81  # gravitational constant
    I = (m * (l ** 2)) / 12
    denominator = 1 / (I * (M + m) + M * m * (l ** 2))
    a = denominator * (g * (m ** 2) * (l ** 2))
    b = denominator * (-g * (M + m))
    c = denominator * (I + m * (l ** 2))
    d = denominator * (-m * l)

    A = np.array([[0, 1, 0, 0],
                  [0, 0, a, 0],
                  [0, 0, 0, 1],
                  [0, 0, b, 0]])
    B = np.array([[0],
                  [c],
                  [0],
                  [d]])
    return A, B


if __name__ == "__main__":
    main()
