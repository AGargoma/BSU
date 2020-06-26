import copy
import time
import random
import tabulate as tb
import numpy as np


def convert_diagonals_to_matrix(a, b, c):
    """Преобразование диагоналей a, b, c к
    трехдиагональной матрице"""
    n = len(c)
    matrix = np.zeros((n, n))
    for i in range(n - 1):
        matrix[i + 1][i] = a[i]
    for i in range(n):
        matrix[i][i] = c[i]
    for i in range(n - 1):
        matrix[i][i + 1] = b[i]
    return matrix


def generate(num, start=-10, stop=10):
    """Генерация числа а: abs(a)>abs(num), -10<=a<=10"""
    x = num
    while (abs(x) <= abs(num)):
        x = round(random.uniform(abs(num), stop), 2)

    return x * (-1) ** random.randint(0, 1)


def generate_vector(n):
    """Генерация случайного вектора"""
    return np.round(np.random.randint(-10, 10, 1) + np.random.rand(n, 1), 2)


def generate_vectors(n):
    """
    Генерация диагоналей a, b, c, удовлетворющих условию корректности
    и устойчивости метода прогонки
    """
    a, b, c = [generate(0) for i in range((n - 1))], [generate(0) for i in range(n - 1)], [0] * (n)
    c[0] = generate(abs(b[0]))
    for i in range(1, n - 2):
        c[i] = generate(abs(a[i]) + abs(b[i]), stop=20)
    c[-1] = (generate(abs(a[-1])))
    return (a, b, c)


def max_norm(vector):
    """Максимум-норма"""
    return max(map(abs, [vector.min(), vector.max()]))


def solve_lin_system(a, b, c, f, det_A):
    """Решение СЛАУ методом прогонки"""

    n = len(c)
    x = [0 for i in range(n)]
    alpha = [0 for i in range(n - 1)]
    betta = [0 for i in range(n)]
    # Прямая прогонка
    alpha[0] = -b[0] / c[0]
    betta[0] = f[0] / c[0]
    for i in range(1, n - 1):
        alpha[i] = -b[i] / (c[i] - alpha[i - 1] * (-a[i - 1]))
        betta[i] = (f[i] - a[i - 1] * betta[i - 1]) / (c[i] + a[i - 1] * alpha[i - 1])
    betta[-1] = (f[-1] - a[-1] * betta[-2]) / (c[-1] + a[-1] * alpha[-1])

    det_A[0] = c[0]*np.prod([c[i+1]+a[i]*alpha[i] for i in range(n-1)])
    # Обратная прогонка
    x[-1] = betta[-1]
    for i in reversed(range(n - 1)):
        x[i] = alpha[i] * x[i + 1] + betta[i]
    return np.asarray(x)


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


# Исходные данные
matrix_size = 10
np.random.seed(round(time.time()))
a, b, c = generate_vectors(matrix_size)
x = generate_vector(matrix_size)
A = convert_diagonals_to_matrix(a, b, c)
f = np.matmul(A, x)

# Решение СЛАУ
try:
    det_A=[0]
    approximate_x = solve_lin_system(a, b, c, f, det_A)
    approximate_f = np.matmul(A, approximate_x)
    print("Матрица A: \n", tb.tabulate(A, tablefmt="fancy_grid"))
    print(tb.tabulate(zip(x, f, approximate_x, approximate_f),
                      headers=["Вектор x", "Вектор f = A*x", "Приближенное решение",
                               "Приближенное f"],
                      tablefmt="fancy_grid", floatfmt=".16f"))

    max_norm_of_residual = max_norm(f - approximate_f)  # Максимум-норма невязки
    max_norm_of_error = max_norm(x - approximate_x)  # Максимум-норма погрешности
    det_A1 = det(A)
    #assert (det_A1 == det_A[0])
    output = [["det(A)", det_A[0]], ["Максимум-норма невязки", max_norm_of_residual],
              ["Максимум-норма погрешности", max_norm_of_error]]
    print(tb.tabulate(output, tablefmt="fancy_grid", floatfmt=".14f"))
except Exception as error:
    print(error.args)
