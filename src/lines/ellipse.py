import numpy as np
import math


class Ellipse:
    def __init__(self, points: list):
        self._calculate_equation(points)

    def _calculate_equation(self, points: list):
        A_mat = []
        B_mat = []
        for (x, y) in points:
            A_mat.append([x ** 2, x * y, y ** 2, x, y])
            B_mat.append(1)

        A_mat = np.array(A_mat)
        B_mat = np.array(B_mat)

        A, B, C, D, E = np.linalg.solve(A_mat, B_mat)
        self._a: np.float64 = A
        self._b: np.float64 = B
        self._c: np.float64 = C
        self._d: np.float64 = D
        self._e: np.float64 = E

    def get_equation_parameters(self) -> tuple:
        return self._a, self._b, self._c, self._d, self._e

    # TODO metody stricte do wyświetlania, poprawić jak będzie chwila
    def get_center(self) -> tuple:
        x: np.float64 = (self._b * self._e - 2 * self._c * self._d) / (4 * self._a * self._c - self._b ** 2)
        y: np.float64 = (self._b * self._d - 2 * self._a * self._e) / (4 * self._a * self._c - self._b ** 2)
        return x, y

    def get_axes(self, func: tuple) -> tuple:
        M = np.array([[self._a, self._b / 2], [self._b / 2, self._c]])

        eigenvalues: np.ndarray = np.linalg.eigvals(M)
        lambda1: np.float64 = eigenvalues[0]
        lambda2: np.float64 = eigenvalues[1]

        delta: np.float64 = (self._b ** 2 - 4 * self._a * self._c)
        if delta == 0:
            raise ValueError("Nie jest to równanie elipsy lub opis jest niewłaściwy.")

        a: np.float64 = np.sqrt(-delta / lambda1)
        b: np.float64 = np.sqrt(-delta / lambda2)
        return a, b

    def is_point_on(self, point: tuple) -> bool:
        result: np.float64 = (point[0] ** 2 * self._a + point[0] * point[1] * self._b + point[1] ** 2 * self._c +
                              point[0] * self._d + point[1] * self._e - 1)
        if result < 0.005:
            return True
        return False
