import numpy as np
from src.cartpole import ErgodicHelper, MPC


if __name__ == "__main__":
    D = []
    for i in range(1, 8):
        D.append(np.genfromtxt(f'src/cartpole_gazebo/dynamics/test{i}.csv', delimiter=','))
    E, K, dt = [-1, -1, 1, -1, -1, -1, -1], 5, 0.1
    L = [[-np.pi, np.pi], [-11, 11], [-15, 15], [-15, 15]]
    ergodic_test = ErgodicHelper(D, E, K, L, dt)
    hk, lambdak, phik = ergodic_test.calc_fourier_metrics()
