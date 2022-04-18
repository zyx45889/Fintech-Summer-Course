# MV Portfolio
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


def MV_weight_compute(n, context=None):
    returns = numpy.asmatrix(context["R"])
    returns = returns.T
    S = opt.matrix(numpy.cov(returns))
    pbar = opt.matrix(numpy.mean(returns, axis=1))

    G = -opt.matrix(numpy.eye(n))
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    portfolios = solvers.qp( S, -0.0*pbar, G, h, A, b)['x']
    wt=numpy.asarray(portfolios)
    wt=wt.T
    wt=wt[0]
    return wt

if __name__ == "__main__":
    print("this is MV Portfolio")