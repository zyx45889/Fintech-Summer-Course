# EG Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk and the portfolio returns µk
import numpy
import os
from data_load.stocks import Stocks
from trade.portfolio import Portfolio


span_t = 120


def EG_weight_compute(n, context=None):
    if type(context["last_w"])==type(None):
        w = numpy.ones(n)
        w = w / n
        return w
    X=numpy.asarray(context["Rk"])
    W=numpy.asarray(context["last_w"])
    learning_rate = 200
    p=0.0
    for i in range(n):
        p=p+W[i]*X[i]
    w=[]
    z=0.0
    for i in range(n):
        wi=W[i]*numpy.exp(learning_rate*X[i]/p)
        w.append(wi)
        z=z+wi
    w=w/z
    return w


if __name__ == "__main__":
    print("this is EG Portfolio")