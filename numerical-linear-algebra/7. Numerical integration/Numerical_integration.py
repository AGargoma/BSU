import numpy as np
import math


def f(x):
    return 1/(np.log(1+x+x**2))

a, b = 2, 3

eps = 10**(-4)


runge_coeffs = {'right_rectangles_method': 1, 'trapezium_method': 1/3,
              'simpsons_method': 1/15}


nodes = {2: [-0.55773502692, 0.55773502692],
           3: [-0.7745966692, 0, 0.7745966692],
           4: [-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116]}
gauss_coeffs = {2: [1, 1],
           3: [5/9, 8/9, 5/9],
           4: [0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451]}


# преобразование [-1,1] -> [a,b]
def transform(x):
    return (a + b) / 2 + (b - a) * x / 2

# погрешность по правилу Рунге
def runge(prev, curr, theta):
    return np.abs(curr - prev) * theta




def right_rectangles_method(n):
    xs = np.linspace(a, b, n+1)
    h = xs[1:] - xs[:-1] # h_i = x_i - x_{i-1}, i = 1,..
    return np.sum(f(xs[1:]) * h) # сумма f(x_i)*h_i


def trapezium_method(n):
    xs = np.linspace(a, b, n+1)
    h = xs[1:] - xs[:-1]
    return np.sum((f(xs[:-1]) + f(xs[1:])) / 2 * h)


def simpsons_method(n):
    xs = np.linspace(a, b, n+1)
    h = xs[1:] - xs[:-1]
    return np.sum((f(xs[:-1]) + 4 * f((xs[:-1] + xs[1:])/2) +
        + f(xs[1:])) * h / 6)


def gauss_formula(n):
    result = 0
    for (xi, ai) in zip(nodes[n], gauss_coeffs[n]):
        result += ai * f(transform(xi))
    return result * (b-a)/2



def calculate(method, method_name):
    theta = runge_coeffs[method_name]
    n = 1
    h = b - a
    curr = method(n)
    error = math.inf
    print(f'& $h = {h:.6f}$ & $I_h={curr:.6f}$ &  \\\\ \\cline{{2-4}} ')
    while error > eps:
        n *= 2
        h /= 2
        prev, curr = curr, method(n)
        error = runge(prev, curr, theta)
        print(f'& $h/{n} = {h:.6f}$ & $I_{{h/{n}}}={curr:.6f}$ & $R_{{h/{n}}}=\
{error:.6f}$ \\\\ \cline{{2-4}} ')

calculate(right_rectangles_method, 'right_rectangles_method')
calculate(trapezium_method, 'trapezium_method')
calculate(simpsons_method, 'simpsons_method')

for i in range(2, 5):
    print(f'{i}:\t{gauss_formula(i):.10f}')
