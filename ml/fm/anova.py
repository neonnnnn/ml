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

def anova_saving_memory(p, x, order, dptable):
    dptable[0] = 1
    dptable[1:] = 0
    d = p.shape[0]
    return _anova_saving_memory(p, x, order, d, dptable)


def _anova_saving_memory(p, x, order, d, a):
    for j in range(order):
        for t in range(j+1):
            a[j+1-t] += a[j-t] * p[j]*x[j]
    for j in range(order, d):
        for t in range(order):
            a[order-t] += a[order-t-1]*p[j]*x[j]

    return a[order]

def anova_alt(p, x, order, dptable_anova, dptable_poly):
    dptable_anova[:, 0] = 1
    dptable_poly[:, 0] = 1
    return _anova_alt(p, x, order, dptable_anova, dptable_poly)


def _anova_alt(p, x, order, a, poly):
    for m in range(1, order+1):
        temp = 0.
        sign = 1.
        for t in range(1, m+1):
            temp += sign * a[:, m-t]*poly[:, t]
            sign *= -1
        a[:, m] = temp / m

    return a[:, order]


def grad_anova(p, x, order, dptable_anova, dptable_grad):
    d = p.shape[0]
    dptable_grad[order-1, d-1] = 1

    return _grad_anova(p, x, order, d, dptable_anova, dptable_grad)


def _grad_anova(p, x, order, d, a, grad):
    for t in range(order, 0, -1):
        for j in range(d-1, t-1, -1):
            grad[t-1, j-1] = grad[t-1, j] + grad[t, j]*p[j]*x[j]

    return np.sum(a[:order, :-1] * grad[:order], axis=0) * x


def grad_anova_alt(p_js, x_j, order, dptable_anova, dptable_poly, dptable_grad):
    dptable_grad[:, 0] = 0
    dptable_grad[:, 1] = x_j
    return _grad_anova_alt(p_js, x_j, order, dptable_anova, dptable_poly, dptable_grad)


def _grad_anova_alt(p, x, order, a, poly, grad):
    for m in range(2, order+1):
        temp = 0
        sign = 1
        for t in range(1, m+1):
            temp += (grad[:, m-t]*poly[:, t] + a[:, m-t]*t*(p**(t-1))*(x**t)) * sign
            sign *= -1
        grad[:, m] = temp / m

    return grad[:, order]

