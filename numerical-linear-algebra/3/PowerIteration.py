# Вариант 2
import numpy as np
import tabulate as tb
import sympy as sp

# Степенной метод со скоростью сходимости (lambda_2/lambda_1)^k
def power_iteration1(B,u_k):
    A = np.copy(B)
    v_k = A@u_k
    u_k1 = (v_k)/np.linalg.norm(v_k,np.inf) # нормирование
    l1 = v_k/u_k
    u_k = u_k1
    v_k = A@u_k
    u_k1 = (v_k)/np.linalg.norm(v_k,ord=np.inf)
    l2 = v_k/u_k
    count = 2
    # Итерационный процесс
    while(np.linalg.norm(l2-l1,ord=np.inf)>=accuracy):
        u_k = u_k1
        v_k = A @ u_k
        u_k1 = (v_k) / np.linalg.norm(v_k, ord=np.inf)
        l1 = l2
        l2 = v_k/u_k
        count+=1
    return sum(l2)/len(l2),u_k1, count

# Степенной метод со скоростью сходимости (lambda_2/lambda_1)^2k
def power_iteration2(B,u_k):
    A = np.copy(B)
    v_k = A @ u_k
    u_k1 = (v_k) / np.linalg.norm(v_k, ord=np.inf)
    l1 = (v_k @ u_k)/(u_k@u_k)
    u_k = u_k1
    v_k = A @ u_k
    u_k1 = (v_k) / np.linalg.norm(v_k, ord=np.inf)
    l2 = (v_k @ u_k)/(u_k@u_k)
    count = 2
    # Итерационный процесс
    while (abs(l2 - l1) >= accuracy):
        u_k = u_k1
        v_k = A @ u_k
        u_k1 = (v_k) / np.linalg.norm(v_k, ord=np.inf)
        l1 = l2
        l2 = (v_k @ u_k)/(u_k@u_k)
        count += 1
    return l2,u_k1, count


B = np.array([[1.342, 0.432, 0.599, 0.202, 0.603, 0.202],
              [0.432, 1.342, 0.256, 0.599, 0.204, 0.304],
              [0.599, 0.256, 1.342, 0.532, 0.101, 0.506],
              [0.202, 0.599, 0.532, 1.342, 0.106, 0.311],
              [0.603, 0.204, 0.101, 0.106, 1.342, 0.102],
              [0.202, 0.304, 0.506, 0.311, 0.102, 1.342]])
C = np.array([[0.05, 0, 0, 0, 0, 0],
              [0, 0.03, 0, 0, 0, 0],
              [0, 0, 0.02, 0, 0, 0],
              [0, 0, 0, 0.04, 0, 0],
              [0, 0, 0, 0, 0.06, 0],
              [0, 0, 0, 0, 0, 0.07]])

k = 2
accuracy = .1e-5
A = B + k*C
print('Матрица A: \n',tb.tabulate(A))
u_k = np.asarray([1,1,1,1,1,1])
answer1 = power_iteration1(A,u_k)
u_k = np.asarray([1,1,1,1,1,1])
answer2 = power_iteration2(A,u_k)
print('Начальный вектор: ', u_k)
print(tb.tabulate([answer1[:2],answer2[:2]],headers=['Макс. собств. зн.',
"Собственный вектор"],floatfmt=".10f"))
print('Количество итераций для достижения заданной точности в двух способах'
      ' соответсвенно: ',
      answer1[2],' ',answer2[2])


