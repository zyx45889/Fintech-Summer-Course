# ONS Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk and the portfolio returns µk
import numpy
import cvxopt as opt
from cvxopt import blas, solvers
solvers.options['show_progress'] = False
import os
from data_load.stocks import Stocks
from trade.portfolio import Portfolio


span_t = 120


def proj_netwon(A, y):
    n = A.shape[0]
    P=opt.matrix(A)
    Q=opt.matrix(0.0, (n, 1))
    G = opt.matrix(numpy.eye(n))  # negative n x n identity matrix
    h = opt.matrix(y)
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(numpy.sum(y)-1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = solvers.qp(P, Q, G, h, A, b)['x']
    x = numpy.asarray(portfolios)
    x = x.T
    x = x[0]
    return y-x

def ONS_weight_compute(n, context=None):
    if type(context["last_w"])==type(None):
        w = numpy.ones(n)
        w = w / n
        return w
    cum_grad = context["cum_grad"]
    A = context["A"]
    A_inv = context["A_inv"]
    beta = 1
    delta = 0.125
    b = (1+1/beta) * cum_grad
    w = proj_netwon(A, delta * A_inv.dot(b))
    return w


if __name__ == "__main__":
    print("this is ONS Portfolio")