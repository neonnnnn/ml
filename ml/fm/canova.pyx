cimport cython
cimport numpy as np
import numpy as np


cpdef double anova(double[:] p, double [:] x, int order, double[:, :] dptable):
    d = p.shape[0]
    dptable[0] = 1.
    return _anova(p, x, order, d, dptable)


cdef double _anova(double[:] p, double[:] x, int order, int d, double[:, :] a):
    cdef int t, j
    for t in range(1, order+1):
        for j in range(t, d+1):
            a[t, j] = a[t, j-1] + p[j-1]*x[j-1]*a[t-1, j-1]
    return a[t, j]


cpdef double anova_saving_memory(double[:] p, double[:] x, int order, double[:] dptable):
    dptable[0] = 1
    dptable[1:] = 0
    d = p.shape[0]
    return _anova_saving_memory(p, x, order, d, dptable)


cdef double _anova_saving_memory(double[:] p, double[:] x, int order, int d, double[:] a):
    cdef int i, j
    for j in range(order):
        for t in range(j+1):
            a[j+1-t] += a[j-t]*p[j]*x[j]
    for j in range(order, d):
        for t in range(order):
            a[order-t] += a[order-t-1]*p[j]*x[j]

    return a[order]


cpdef np.ndarray[double, ndim=1] anova_alt(np.ndarray[double, ndim=1] p,
                                           np.ndarray[double, ndim=2] x,
                                           int order,
                                           np.ndarray[double, ndim=2] dptable_anova,
                                           np.ndarray[double, ndim=2] dptable_poly):
    dptable_anova[:, 0] = 1
    dptable_anova[:, 1:] = 0
    dptable_poly[:, 0] = 1
    return _anova_alt(p, x, order, dptable_anova, dptable_poly)


cdef np.ndarray[double, ndim=1] _anova_alt(np.ndarray[double, ndim=1] p,
                                           np.ndarray[double, ndim=2] x,
                                           int order,
                                           np.ndarray[double, ndim=2] a,
                                           np.ndarray[double, ndim=2] poly):
    cdef int m, t, sign
    sign = 1
    for m in range(1, order+1):
        sign = 1
        for t in range(1, m+1):
            a[:, m] += sign * a[:, m-t]*poly[:, t]
            sign *= -1
        a[:, m] /= m

    return a[:, order]


cpdef np.ndarray[double, ndim=1] grad_anova(np.ndarray[double, ndim=1] p,
                                          np.ndarray[double, ndim=1] x,
                                          int order,
                                          np.ndarray[double, ndim=2] dptable_anova,
                                          np.ndarray[double, ndim=2] dptable_grad):
    d = p.shape[0]
    dptable_grad[order-1, d-1] = 1

    return _grad_anova(p, x, order, d, dptable_anova, dptable_grad)


cdef np.ndarray[double, ndim=1] _grad_anova(np.ndarray[double, ndim=1] p,
                                            np.ndarray[double, ndim=1] x,
                                            int order,
                                            int d,
                                            np.ndarray[double, ndim=2] a,
                                            np.ndarray[double, ndim=2] grad):
    cdef int t,j
    for t in range(order, 0, -1):
        for j in range(d-1, t-1, -1):
            grad[t-1, j-1] = grad[t-1, j] + grad[t, j]*p[j]*x[j]

    return np.sum(a[:order, :-1] * grad[:order], axis=0) * x


cpdef np.ndarray[double, ndim=1] grad_anova_alt(double p_js,
                                                np.ndarray[double, ndim=1] x_j,
                                                int order,
                                                np.ndarray[double, ndim=2] dptable_anova,
                                                np.ndarray[double, ndim=2] dptable_poly,
                                                np.ndarray[double, ndim=2] dptable_grad):
    dptable_grad[:, :] = 0
    dptable_grad[:, 1] = x_j
    return _grad_anova_alt(p_js, x_j, order, dptable_anova, dptable_poly, dptable_grad)


cdef np.ndarray[double, ndim=1] _grad_anova_alt(double p,
                                                np.ndarray[double, ndim=1] x,
                                                int order,
                                                np.ndarray[double, ndim=2] a,
                                                np.ndarray[double, ndim=2] poly,
                                                np.ndarray[double, ndim=2] grad):
    cdef int m, t, sign
    for m in range(2, order+1):
        sign = 1
        for t in range(1, m+1):
            grad[:, m] += (grad[:, m-t]*poly[:, t] + a[:, m-t]*t*(p**(t-1))*(x**t)) * sign
            sign *= -1
        grad[:, m] = temp / m

    return grad[:, order]
