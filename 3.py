import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_graph(x, col):
    plt.plot(x, color = col)
    plt.ylabel('Price')
    plt.show()

def fun_for_n_x(s):
    Sum = 0
    if (1 <= s and s < tau):
        for i in range(s):
            Sum += X_tilda[i][s - i]
        return Sum / s
    if (tau <= s and s < n):
        for i in range(1, tau + 1):
            Sum += X_tilda[i - 1][s - i]
        return Sum / tau
    if (n <= s and s <= N):
        for i in range(N - s + 1):
            Sum += X_tilda[i + s - n - 1][n - i - 1]
        return Sum / (N - s + 1)

if __name__ == '__main__':
    csv_data = pd.read_csv('GAZP.csv', delimiter = ';')    
    N = len(csv_data)
    #date = pd.to_datetime(csv_data['<DATE>'])
    price = csv_data['<LAST>'].values
    tau = int((N + 1) / 2)
    r = 15
    xt = np.array([float(price[i]) for i in range(N)]) 
    X = np.array([xt[i : N - tau + i] for i in range(tau)])
    n = len(X[0])
    C = (1 / n) * X.transpose().dot(X) 
    L, V = np.linalg.eig(C)
    Y = V.T.dot(X)
    
    X_tilda = V[:, :r].dot(Y[:r,])    
    new_X = [fun_for_n_x(i) for i in range(1, N)]

    
    V_tau = V[-1, : r]
    V_s = V[:tau - 1, : r]
    Q = X[-tau + 1 : ]
    res = V_tau.dot(V_s.transpose())
    V_tau_T = 1 - V_tau.dot(V_tau.T)
    pred_X = (res.dot(Q)) / V_tau_T

    plot_graph(price, 'r')
    plot_graph(new_X, 'g')
    plot_graph(pred_X, 'b')