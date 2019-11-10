import tabulate as tb
import numpy as np
import time
import copy


def generate_matrix(n):
    """Генерация случайной матрицы"""
    return np.round(np.random.randint(-10, 10, (10, 10)) + np.random.rand(n, n),
                    2)


def generate_vector(n):
    """Генерация случайного вектора"""
    return np.round(np.random.randint(-10, 10, 1) + np.random.rand(n, 1), 2)


def max_norm(vector):
    """Максимум-норма"""
    return max(map(abs, [vector.min(), vector.max()]))


# Решение СЛАУ
def solve_lin_system(A, f, eps, w, count=0):
    n = A.shape[0]
    x_cur = np.zeros(n)
    x_next = np.zeros(n)
    norm = float('inf')
    # Итерационный процесс метода релаксации
    while (norm >= eps):
        for i in range(n):
            x_next[i] = (1 - w) * x_cur[i] + (w / A[i][i]) * \
                        (f[i] - sum(x_next[:i] * A[i][:i]) - sum(
                            x_cur[i + 1:] * A[i][i + 1:]))
        norm = max_norm(abs(x_cur - x_next))
        x_cur = copy.deepcopy(x_next)
        count += 1  # +1 итерация
    return x_next, count, norm


# Решение СЛАУ с различными параметрами релаксации w
def test(A, f, eps):
    params = [0.2, 0.5, 0.8, 1, 1.3, 1.5, 1.8]
    amount_of_iterations = [0] * len(params)
    norm = [0] * len(params)
    result = ()
    for i in range(len(params)):
        count = 0
        result = solve_lin_system(np.matmul(np.transpose(A), A),
                                  np.matmul(np.transpose(A), f), eps, params[i],
                                  count)
        amount_of_iterations[i] = result[1]
        norm[i] = result[2]
    print(tb.tabulate(zip(params, amount_of_iterations, norm),
                      headers=["w", "Количество итераций", "Максимум-норма", ],
                      tablefmt="fancy_grid", floatfmt=".14f"))
    return result[0]


# Исходные данные
matrix_size = 10
np.random.seed(round(time.time()))
A = generate_matrix(matrix_size)
x = generate_vector(matrix_size)
f = np.matmul(A, x)

try:
    accuracy = .1e-4
    approximate_x = test(A, f, accuracy)
    approximate_f = np.matmul(A, approximate_x)
    print("Матрица A: \n", tb.tabulate(A, tablefmt="fancy_grid"))
    print("Точность: {0}\n".format(accuracy),
          tb.tabulate(zip(x, approximate_x, f, approximate_f),
                      headers=["Вектор x", "Приближенное решение",
                               "Вектор f = A*x",
                               "Приближенное f"],
                      tablefmt="fancy_grid", floatfmt=".14f"))

    max_norm_of_error = max_norm(
        x.T - approximate_x)  # Максимум-норма погрешности
    print("Максимум-норма погрешности(w = 1.8): {0}".format(max_norm_of_error))
except Exception as error:
    print(error.args)
