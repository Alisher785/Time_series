from math import *

def theta(z):
    return 0 if z < 0 else 1

def p(x, k, n1, n2):
    x1 = x[n1 - k : n1]
    x2 = x[n2 - k : n2]
    res = sum([(x1[i] - x2[i]) ** 2 for i in range(k)]) 
    return sqrt(res)

def C(x, N, l, k):
    res = [[theta(l - p(x, k, i, j)) for j in range(N - k, N)] for i in range(N - k, N)] 
    n_res = [sum(res[i]) for i in range(len(res))]
    return((1 / N ** 2) * sum(n_res))

def d(l, C_res):
    return log(C_res) / log(l)
    

dt = 0.01
N = 10000
l = 0.0003
x = [2 * cos(dt * i) for i in range(N)]
y = [1 * sin(dt * i) for i in range(N)]
for k in range(2, 8):
    res_C = C(x, N, l, k)
    print(d(l, res_C))
