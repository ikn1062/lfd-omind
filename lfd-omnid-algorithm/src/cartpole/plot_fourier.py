from .ergodic_helper import ErgodicHelper
import numpy as np
import matplotlib.pyplot as plt


def get_phix(x, K, phik_dict, ergodic_helper):
    # Negative Euler Identity? Multiply by 2 for real function?
    phix = 0
    for ki in range(K):
        for kj in range(K):
            k_str = f"{ki}{kj}"
            phik = phik_dict[k_str]
            fk = ergodic_helper.calc_Fk(x, [ki, kj])
            phix += phik * fk
    phix *= 2
    return phix


def fourier_plot_2dim():
    num_trajec = 13
    demonstration_list = [0, 6, 7, 10, 11, 12]

    K = 8
    L = [[-np.pi, np.pi], [-11, 11]]
    dt = 0.01
    E = [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1]
    new_E = []
    D = []
    for i in range(num_trajec):
        if i not in demonstration_list:
            continue
        new_E.append(E[i])
        demonstration = np.genfromtxt(f'src/cartpole_gazebo/dynamics/test{i}.csv', delimiter=',')
        D.append(demonstration)
    D = [np.array(d)[:, :2] for d in D]
    # E = E[:num_trajec]

    ergodic_helper = ErgodicHelper(D, new_E, K, L, dt)
    _, phik_dict, _ = ergodic_helper.calc_fourier_metrics()

    x = np.linspace(-np.pi, np.pi, 50)
    y = np.linspace(-15, 15, 50)
    z = np.array([get_phix([i, j], K, phik_dict, ergodic_helper) for j in y for i in x])
    Z = z.reshape(50, 50)

    plt.imshow(Z, interpolation='none', vmin=-0.1, vmax=0.5, extent=[-3.14159, 3.14159, -15, 15], aspect='auto')
    plt.show()


if __name__ == "__main__":
    fourier_plot_2dim()
