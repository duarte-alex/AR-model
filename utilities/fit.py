import numpy as np
import sys
import matplotlib.pyplot as plt


class polynomial_fit:
    def __init__(self, points):
        points = np.array(points)
        dim = points.shape[0]
        self.dim = dim
        A = np.zeros((dim, dim))
        B = np.zeros((dim))
        A[:, 0] = 1
        for i in range(1, dim):
            A[:, i] = points[:, 0] ** i
            B[:] = points[:, 1]

        self.C = np.linalg.solve(A, B)

    def f(self, x):
        x_powers = np.zeros((self.dim))
        for i in range(self.dim):
            x_powers[i] = x ** i
        return np.sum(x_powers * self.C)
