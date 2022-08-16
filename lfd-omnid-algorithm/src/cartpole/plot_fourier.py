import matplotlib.pyplot as plt
import numpy as np


def plot_trajec(show_info=True, show_trajec=True):
    D = []
    for i in range(1, 10):
        demonstration = np.genfromtxt(f'src/cartpole_gazebo/dynamics/test{i}.csv', delimiter=',')
        D.append(demonstration)
    E = [1, -1, -1, -1, -1, -1, 1, -1, -1]

    if show_trajec:
        for i in range(len(D)):
            if E[i] == 1:
                plt.plot(D[i][:, 0], D[i][:, 1], 'bo', markersize=0.2, linestyle='None')
            else:
                plt.plot(D[i][:, 0], D[i][:, 1], 'r+', markersize=0.2, linestyle='None')

    if show_info:
        sizet, sizetd = 15, 40
        bin_theta = np.linspace(-3.14, 3.14, sizet)
        bin_thetadot = np.linspace(-20, 20, sizetd)
        contour_count = np.zeros((sizetd + 1, sizet + 1))

        for i in range(len(D)):
            digitize_theta = np.digitize(D[i][:, 0], bin_theta)
            digitize_thetadot = np.digitize(D[i][:, 1], bin_thetadot)
            trajec_len = len(D[i])
            for ii in digitize_theta:
                for jj in digitize_thetadot:
                    contour_count[jj][ii] += (1/trajec_len) * E[i]

        for i in contour_count:
            print(i)
        contour_count = contour_count[1:, 1:]
        plt.contourf(bin_theta, bin_thetadot, contour_count, 200, cmap='RdBu')
    plt.show()


if __name__ == "__main__":
    plot_trajec()
