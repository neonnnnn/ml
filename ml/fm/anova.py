import numpy as np


def anova(p, x, order, dptable):
    d = p.shape[0]
    dptable[0] = 1
    return _anova(p, x, order, d, dptable)


def _anova(p, x, order, d, a):
    for t in range(1, order+1):
        for j in range(t, d+1):
            a[t, j] = a[t, j-1] + p[j-1]*x[j-1]*a[t-1, j-1] 
    return a[t, j]


def grad_anova(p, x, order, dptable_anova, dptable_grad):
    d = p.shape[0]
    dptable_grad[order-1, d-1] = 1

    return _grad_anova(p, x, order, d, dptable_anova, dptable_grad)


def _grad_anova(p, x, order, d, a, grad):
    for t in range(order, 0, -1):
        for j in range(d-1, t-1, -1):
            grad[t-1, j-1] = grad[t-1, j] + grad[t, j]*p[j]*x[j]

    return np.sum(a[:order, :-1] * grad[:order], axis=0) * x
