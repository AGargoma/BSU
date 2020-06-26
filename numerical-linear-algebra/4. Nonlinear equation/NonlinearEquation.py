# coding: utf-8
import itertools as it
import sympy as sp
from math import *
import matplotlib.pylab as pl
import numpy as np
import scipy.optimize as opt
from tabulate import tabulate


def f(x):
    return log(x + 2) - x ** 2

from IPython.display import display, Markdown, Latex

# Локализация корня методом дихотомии
def localize_root(func, a0, b0, epsilon):
    k = [0]
    a = [a0]
    b = [b0]
    f_a = [func(a0)]
    f_b = [func(b0)]
    med = [(a[-1] + b[-1]) / 2]
    f_med = [func(med[-1])]
    error = [abs(a0 - b0) / 2]
    while (error[-1] >= epsilon):
        if (f_med[-1] > 0):
            a.append(a[-1])
            b.append(med[-1])
        elif (f_med[-1] < 0):
            b.append(b[-1])
            a.append(med[-1])
        else:
            break
        k.append(k[-1] + 1)
        f_a.append(func(a[-1]))
        f_b.append(func(b[-1]))
        med.append((a[-1] + b[-1]) / 2)
        f_med.append(func(med[-1]))
        error.append(abs(a[-1] - b[-1]) / 2)
    display(Markdown(tabulate([['$' + column + '$' for column in
                                ['k', 'a_k', 'b_k', 'f(a_k)', 'f(b_k)', '\\frac{a_k+b_k}{2}', 'f(\\frac{a_k+b_k}{2})',
                                 '\\epsilon_k']],
                               *list(zip(k, a, b, f_a, f_b, med, f_med, error))], headers='firstrow', tablefmt='pipe')))
    return a[-1], b[-1]


delta = [-1, -0.5]
error = 10 ** -7
localization_error = 10 ** -2
a, b = localize_root(f, *delta, localization_error)

t = sp.symbols('t')


def phi(t):
    return t + c * f(t)


def fsym(x):
    return sp.log(t + 2) - t ** 2


c = -0.3
deriv = sp.lambdify(t, sp.diff(t + c * fsym(t), t)) # производная \phi
x0 = (a + b) / 2
delta1 = abs(b - x0)
q = 0.45
abs(x0 - phi(x0)) <= (1 - q) * delta1

# Метод простых итераций
def im(a, b, x0, error):
    k = [0]
    result = [x0]
    eps = [float('inf')]
    while (eps[-1] >= error):
        result.append(phi(result[-1]))
        eps.append(abs(result[-1] - result[-2]))
        k.append(k[-1] + 1)
    return k, result, eps


df = sp.lambdify(t, sp.diff(fsym(t), t))

# Метод Ньютона
def newton(a, b, x0, error):
    k = [0]
    result = [x0]
    eps = [float('inf')]
    while (eps[-1] >= error):
        result.append(result[-1] - f(result[-1]) / df(result[-1]))
        eps.append(abs(result[-1] - result[-2]))
        k.append(k[-1] + 1)
    return k, result, eps



k1, res1, eps1 = im(a, b, x0, error)
k2, res2, eps2 = newton(a, b, x0, error)
print(tabulate([*list(it.zip_longest(k1, res1, eps1, res2, eps2, fillvalue='-'))], tablefmt='pipe', floatfmt='.9f'))
