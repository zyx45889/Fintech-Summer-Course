import numpy

# You may find below useful for Support Vector Machine
# More details in
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
#from scipy.optimize import minimize

def func(X, y):
    n_epoch=40
    P, N = X.shape
    w = numpy.zeros((P + 1, 1), dtype='float64')
    epoch=0
    rho=0.13
    while(epoch<n_epoch):
        for i in range(N):
            S=w[0]
            for j in range(P):
                S=S+w[j+1]*X[j][i]
            if S>0:
                y_predict=1
            else:
                y_predict=-1
            for j in range(P):
                w[j+1]=w[j+1]+rho*X[j][i]*(y[0][i]-y_predict)
            w[0]=w[0]+rho*(y[0][i]-y_predict)
        epoch=epoch+1
    return w

def LR(XX,yy):
    P, N = XX.shape
    X=numpy.zeros((N,P+1),dtype='float64')
    for i in range(N):
        X[i][0]=1
        for j in range(P):
            X[i][j+1]=XX[j][i]
    y=yy.transpose()
    return numpy.dot(numpy.linalg.inv(numpy.dot(numpy.transpose(X),X)),numpy.dot(numpy.transpose(X),y))

# from sklearn import svm
# def SVM(X,y):
#     P, N = X.shape
#     clf = svm.SVC(kernel='linear')
#     Y=numpy.zeros((N,))
#     for i in range(N):
#         Y[i]=y[0][i]
#     clf.fit(X.transpose(), Y)
#     ww = clf.coef_[0]
#     bb = clf.intercept_[0]
#     w = numpy.zeros((P + 1, 1), dtype='float64')
#     w[0]=bb
#     for i in range(P):
#         w[i+1]=ww[i]
#     return w

from scipy.optimize import minimize
def SVM(XX,yy):
    P, N = XX.shape
    X=numpy.zeros((N,P+1),dtype='float64')
    for i in range(N):
        X[i][0]=1
        for j in range(P):
            X[i][j+1]=XX[j][i]
    # y=yy.transpose()
    w = numpy.zeros((P + 1, 1), dtype='float64')
    y = numpy.zeros((N,))
    for i in range(N):
        y[i] = yy[0][i]

    def fun(w, X, y):
        object_value = 0.5 * numpy.sum(w ** 2)
        return object_value

    def constraint(w, X, y):
        return y * numpy.dot(X,w) - 1

    solver = minimize(fun,
                           w, args=(X, y),
                           constraints=({'type': 'ineq', 'args': (X, y),
                                         'fun': lambda w, X,
                                                       y: constraint(w, X, y)}
                                        ))
    return solver.x

