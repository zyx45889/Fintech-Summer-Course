import numpy as np
from gen_data import gen_data
from plot import plot
from todo import func
from todo import LR
from todo import SVM

no_iter = 100  # number of iteration
no_train = 80# YOUR CODE HERE # number of training data
no_test = 20# YOUR CODE HERE  # number of testing data
no_data = 100  # number of all data
assert(no_train + no_test == no_data)

cumulative_train_err = 0
cumulative_test_err = 0

for i in range(no_iter):
    print("now iter:",i)
    X, y, w_f = gen_data(no_data)
    P, N = X.shape
    X_train, X_test = X[:, :no_train], X[:, no_train:]
    y_train, y_test = y[:, :no_train], y[:, no_train:]
    w_g = LR(X_train, y_train)
    train_err=0
    for j in range(no_train):
        S = w_g[0]
        for k in range(P):
            S = S + w_g[k+1] * X_train[k][j]
        if S > 0:
            y_predict = 1
        else:
            y_predict = -1
        if y_predict!=y_train[0][j]:
            train_err=train_err+1
    cumulative_train_err += train_err/no_train
    test_err=0
    for j in range(no_test):
        S = w_g[0]
        for k in range(P):
            S = S + w_g[k+1] * X_test[k][j]
        if S > 0:
            y_predict = 1
        else:
            y_predict = -1
        if y_predict!=y_test[0][j]:
            test_err=test_err+1
    cumulative_test_err += test_err/no_test

train_err = cumulative_train_err / no_iter
test_err = cumulative_test_err / no_iter

plot(X, y, w_f, w_g, "Classification")
print("Training error: %s" % train_err)
print("Testing error: %s" % test_err)