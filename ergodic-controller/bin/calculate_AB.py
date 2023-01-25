import numpy as np


def cartpole_AB_matrix(M, m, l):
    g = 9.81  # gravitational constant
    I = (m * (l ** 2)) / 12

    """
    denominator = 1 / (13*M + m)
    a = denominator * -12 * m * g
    b = (denominator * 12 * g * (M + m)) / l
    c = denominator * 13
    d = denominator * (-12 / l)

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
    """
    A = [[0, 1, 0, 0],
         [0, -3 / M, 1 * m * g / M, 0],
         [0, 0, 0, 1],
         [0, -1 * 3 / M / l, -1 * (m + M) * g / M / l, 0]]
    B = np.transpose(np.array([[0, 1 / M, 0, 1 / M / l]]))
    print(A)
    print(B)

    """
    a = g / (l * (4.0 / 3 - m / (m + M)))
    A = np.array([[0, 1, 0, 0],
                  [0, 0, a, 0],
                  [0, 0, 0, 1],
                  [0, 0, a, 0]])

    # input matrix
    b = -1 / (l * (4.0 / 3 - m / (m + M)))
    B = np.array([[0], [1 / (m + M)], [0], [b]])
    """
    return A, B


if __name__ == "__main__":
    M, m, l = 20, 20, 0.75
    A, B = cartpole_AB_matrix(M, m, l)
    print(f"A: {A}")
    print(f"B: {B}")
