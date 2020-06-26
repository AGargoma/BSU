import numpy as np
import matplotlib.pyplot as plt
import math
a,b = -2,2
length = abs(b-a)
x = np.linspace(a,b,10000)
# функции
def f_1(x):
    return x**3*math.cos(3*x-1)
def f_2(x):
    return abs(5*math.cos(3*x)+3)
y0 = [np.array(list(map(f_1,x))),np.array(list(map(f_2,x)))]
def newton_pol(n,func,nodes):
    y = np.zeros((n,n+1))
    for j in range(n):
        y[j][0] = func(nodes[j])
    for h in range(n-2,-1,-1):
        k = n-h-1
        for i in range(0,h+1):
            y[i][k] = (y[i+1][k-1]-y[i][k-1])/(nodes[i+k]-nodes[i])
    def function(x):
        w = np.cumprod(x-nodes)
        result = y[0][0]+np.sum(np.dot(w,y[0,1:]))
        return result, func(x)-result
    return function
func_title = ['x^3*cos(3*x-1)','|5*cos(3*x)+3)|']
count = 1
for j,func in enumerate([f_1,f_2]):
    for n in [3,5,7,10,15]:
        # построение равномерно расположенных узлов
        h = length/(n-1)
        nodes = np.zeros(n)
        nodes[0] = a
        for i in range(1,n):
            nodes[i] = nodes[i-1]+h
        vnewtonpol1 = np.vectorize(newton_pol(n,func,nodes))
        y1,y2 = vnewtonpol1(x)
        # построение узлов, расположенных оптимальным образом
        nodes = np.zeros(n)
        for i in range(n):
            nodes[i] = (a+b)/2 + (b-a)/2*math.cos(math.pi*(2*i+1)/(2*n+2))
        vnewtonpol2 = np.vectorize(newton_pol(n,func,nodes))
        y3,y4 = vnewtonpol2(x)
        # построение графиков
        plt.figure(figsize=(10,6))
        plt.plot(x,y0[j],'b',label=func_title[j])
        plt.plot(x,y1,'g',label='1 интерполирующая ф-ия')
        plt.plot(x,y3,'m',label='2 интерполирующая ф-ия')
        plt.title(f'Интерполяция по {n} узлам')
        plt.xlabel('x')
        plt.ylabel('y')        
        plt.legend()
        plt.savefig(f'images/{count}')
        count+=1
        plt.show()
        plt.figure(figsize=(10,6))
        plt.plot(x,y2,label='при равномерно расположенных узлах')
        plt.plot(x,y4,label='при узлах, расположенных оптимальным образом')
        plt.title(f'Ошибка интерполяции в обоих случаях при {n} узлах')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.savefig(f'images/{count}')
        count+=1
        plt.show()
