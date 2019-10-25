import copy
import time


import tabulate as tb
import numpy as np


def generate_matrix(n):
    """Генерация случайной матрицы"""
    return np.round(np.random.randint(-10, 10, (10,10)) + np.random.rand(n, n), 2)


def generate_vector(n):
    """Генерация случайного вектора"""
    return np.round(np.random.randint(-10, 10, 1) + np.random.rand(n, 1), 2)


def solve_lin_system(A, f):
    """Решение СЛАУ"""
    a = copy.deepcopy(A)
    b = copy.deepcopy(f)
    augmented_A = np.concatenate((a, b), axis=1)
    if (det(A) == 0) or (det(A) != det(augmented_A)):
        raise Exception("Нет решения")
    n = augmented_A.shape[0]

    for i in range(0, n):
        # Search for maximum in this column
        maxEl = abs(augmented_A[i][i])
        maxRow = i
        for k in range(i + 1, n):
            if abs(augmented_A[k][i]) > maxEl:
                maxEl = abs(augmented_A[k][i])
                maxRow = k

        # Swap maximum row with current row (column by column)
        augmented_A[[maxRow, i]] = augmented_A[[i, maxRow]]

        # Make all rows below this one 0 in current column
        for k in range(i + 1, n):
            c = -augmented_A[k][i] / augmented_A[i][i]
            for j in range(i, n + 1):
                if i == j:
                    augmented_A[k][j] = 0
                else:
                    augmented_A[k][j] += c * augmented_A[i][j]

    # Solve equation Ax=b for an upper triangular matrix A
    x = [0 for i in range(n)]
    for i in range(n - 1, -1, -1):
        x[i] = augmented_A[i][n] / augmented_A[i][i]
        for k in range(i - 1, -1, -1):
            augmented_A[k][n] -= augmented_A[k][i] * x[i]
    return np.asarray(x)


def max_norm(vector):
    """Максимум-норма"""
    return max(map(abs, [vector.min(), vector.max()]))


def det(matrix):
    """Вычисление определителя"""
    A = copy.deepcopy(matrix)
    n = len(A)
    for i in range(n):
        for k in range(i + 1, n):
            c = -A[k][i] / A[i][i]
            for j in range(i, n):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += c * A[i][j]
    return np.prod([A[i][i] for i in range(n)])


def inverse(matrix):
    """Вычисление обратной матрицы"""
    if det(matrix) == 0:
        raise Exception("Для данной матрицы нет обратной")
    n = matrix.shape[0]
    augmented_matrix = np.concatenate((matrix, np.identity(n)), axis=1)
    for i in range(n):
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i][i]
        newlist = list(range(n))
        for k in (newlist[:i] + newlist[i + 1:]):
            c = -augmented_matrix[k][i]
            for j in range(i, 2 * n):
                if i == j:
                    augmented_matrix[k][j] = 0
                else:
                    augmented_matrix[k][j] += c * augmented_matrix[i][j]
    # result = augmented_matrix[:][n:]
    result = [row[n:] for row in augmented_matrix]
    return np.asarray(result)


# Исходные данные
matrix_size = 10
np.random.seed(round(time.time()))
A = generate_matrix(matrix_size)
x = generate_vector(matrix_size)
f = np.matmul(A, x)

# Решение СЛАУ
try:
    approximate_x = solve_lin_system(A, f)
    approximate_f = np.matmul(A, approximate_x)
    print("Матрица A: \n", tb.tabulate(A, tablefmt="fancy_grid"))
    print(tb.tabulate(zip(x, f, approximate_x, approximate_f),
                      headers=["Вектор x", "Вектор f = A*x", "Приближенное решение",
                               "Приближенное f"],
                      tablefmt="fancy_grid", floatfmt=".14f"))

    max_norm_of_residual = max_norm(f.T - approximate_f)  # Максимум-норма невязки
    max_norm_of_error = max_norm(x.T - approximate_x)  # Максимум-норма погрешности
    det_A = det(A)
    inverse_A = inverse(A)
    output = [["det(A)", det_A], ["Максимум-норма невязки", max_norm_of_residual],
              ["Максимум-норма погрешности", max_norm_of_error]]
    print(tb.tabulate(output, tablefmt="fancy_grid", floatfmt=".14f"))
    print("A^(-1):\n", tb.tabulate(inverse_A, tablefmt="fancy_grid"))
    print("A^(-1)*A: \n", tb.tabulate(np.matmul(inverse_A, A), tablefmt="fancy_grid"))
except Exception as error:
    print(error.args)
