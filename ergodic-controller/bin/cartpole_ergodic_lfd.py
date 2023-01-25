import os
import numpy as np
import json

from src.ergodic_controller import ErgodicMeasure, Plot2DMetric, iLQR


def main():
    print("Getting Demonstrations")

    # read values from json
    with open(os.path.join("..", "ergodic_system_properties.json"), 'r') as f:
        properties = json.load(f)
        ergodic_properties = properties["ergodic_system"]
        dynamic_properties = properties["dynamic_system"]

    file_path = os.path.join("..", properties["demonstration_path"])
    K, L, dt, E = ergodic_properties["K"], ergodic_properties["L"], ergodic_properties["dt"], ergodic_properties["E"]
    x0, t0, tf = dynamic_properties["x0"], dynamic_properties["t0"], dynamic_properties["tf"]
    A, B = dynamic_properties["A"], dynamic_properties["B"]

    demonstration_list, D, new_E = [], [], []

    input_demonstration = input("Please input the demonstrations you would like to use for training [list] (if empty, all demonstrations are used) \ninput: ")
    if input_demonstration != "q" or len(input_demonstration) != 0:
        input_demonstration = input_demonstration.split(",")
        for num in input_demonstration:
            if num.isnumeric():
                demonstration_list.append(int(num))

    sorted_files = sorted(os.listdir(file_path), key=lambda x: int(x[4:-4]))
    for i, file in enumerate(sorted_files):
        if (len(demonstration_list) != 0 and i not in demonstration_list) or "csv" not in file:
            continue
        new_E.append(E[i])
        demonstration_path = os.path.join(file_path, file)
        demonstration = np.genfromtxt(demonstration_path, delimiter=',')
        demonstration = np.hstack((demonstration[:, 2:], demonstration[:, :2]))
        D.append(demonstration)
    print(new_E)
    if len(D) == 0:
        raise FileNotFoundError("No files found in demonstration folder")

    print("Calculating Ergodic Helpers")
    ergodic_test = ErgodicMeasure(D, new_E, K, L, dt)
    hk, lambdak, phik = ergodic_test.calc_fourier_metrics()

    print("Visualize Ergodic Metric")
    plot_phix_metric = Plot2DMetric(D, new_E, K, L, dt, 0, 1, interpolation='bilinear')

    vis_ergodic = input("Would you like to visualize the Ergodic Metric Spatial Distribution [yY/nN]: ")
    if vis_ergodic in {"y", "Y"}:
        plot_phix_metric.visualize_ergodic()
    vis_trajec = input("Would you like to visualize the Distribution Trajectories [yY/nN]: ")
    if vis_trajec in {"y", "Y"}:
        plot_phix_metric.visualize_trajectory()

    input("Enter to start controller: ")
    print("Starting Grad Descent")
    mpc_model_1 = iLQR(x0, t0, tf, L, hk, phik, lambdak, A, B, dt=dt, K=K)
    mpc_model_1.grad_descent()


if __name__ == "__main__":
    main()
