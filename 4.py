import numpy as np
import pandas as pd
import requests
import json
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def LA(data, p, eps = 1e-6):
    Xi = 3 * (p + 1)
    X = data[np.arange(p) + np.arange(len(data) - p)[:, None]]
    omega = np.argpartition(np.sum(np.power(X - data[-p:], 2), axis=1), Xi)[:Xi]
    idx = np.arange(p)[:, None] - np.arange(p) <= 0
    Y = np.hstack((np.ones(Xi)[:, None], (X[omega, :, None] * X[omega, None, :])[:, idx]))
    params = np.linalg.solve(Y.T.dot(Y) + eps * np.eye(Y.shape[1]), Y.T.dot(data[omega + p]))
    return np.sum(params * np.hstack([1, (data[-p:, None] * data[-p:])[idx]]))


def T_steps_prediction(T, X):
    prediction = np.array([])
    for i in range(T):
        pred = LA(X, 2)
        prediction = np.append(prediction, pred)
        X = np.append(X, pred)
    return prediction
  
if __name__ == '__main__':
    start_time = time.time() - 100*60*60
    resource = requests.get("https://poloniex.com/public?command=returnChartData&currencyPair=BTC_ETH&start=%s&end=9999999999&period=1800" % start_time)
    data = json.loads(resource.text)
    
    quotes = {}
    quotes['open']=np.asarray([item['open'] for item in data])
    quotes['close']=np.asarray([item['close'] for item in data])
    quotes['high']=np.asarray([item['high'] for item in data])
    quotes['low']=np.asarray([item['low'] for item in data])
    
    X = quotes['close'] - quotes['open']
    y = [1 if i >= 0 else -1 for i in X]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    scores = {}
    for k in range(2, 12):
        classifier = KNeighborsClassifier(n_neighbors = k)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        scores[k] = accuracy_score(y_test, y_pred)
    prediction = T_steps_prediction(50, X)
    prediction[prediction > 0] = 1 
    prediction[prediction < 0] = -1
    print(prediction)
