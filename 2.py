from math import *

def theta(z):
    return 0 if z < 0 else 1

def p(x, k, n1, n2):
    x1 = x[n1 - k + 1 : n1]
    x2 = x[n2 - k + 1 : n2]
    res = sum([(x1[i] - x2[i]) ** 2 for i in range(k)]) 
    return sqrt(res)

def C(x, N, l, k):
    res = [[Theta(l - p(x, k, i, j)) for j in range(N - k, N)] for i in range(N - k, N)] 
    return((1 / N ** 2) * sum(res))


x = [2 * cos(dt * i) for i in range(N)]
y = [sin(dt * i) for i in range(N)]
