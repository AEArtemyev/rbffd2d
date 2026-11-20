"""Monomials and their Cartesian derivatives.

exps has shape (M, 2) with nonnegative integer powers [a, b] of x**a * y**b.
x and y follow NumPy broadcasting. Results are float64 arrays with shape
broadcast(x, y).shape + (M,); columns follow the order of exps.
"""

import numpy as np

from .stencil import _integer


def _prepare(x, y, exps):
    exps = np.asarray(exps)
    if (exps.ndim != 2 or exps.shape[1] != 2
            or exps.dtype.kind not in "iuf"
            or not np.all(np.isfinite(exps)) or np.any(exps < 0)
            or np.any(exps != np.floor(exps))):
        raise ValueError(
            "exps must have shape (M, 2) and contain nonnegative integers"
        )
    exps = exps.astype(np.float64, copy=False)
    x, y = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    )
    return x[..., None], y[..., None], exps[:, 0], exps[:, 1]


def get_poly(x, y, exps):
    """Return monomial values."""
    x, y, a, b = _prepare(x, y, exps)
    return x**a * y**b


def get_poly_x(x, y, exps):
    """Return the first x derivatives."""
    x, y, a, b = _prepare(x, y, exps)
    result = np.zeros(x.shape[:-1] + a.shape, dtype=np.float64)
    mask = a >= 1
    a, b = a[mask], b[mask]
    result[..., mask] = a * x**(a-1) * y**b
    return result


def get_poly_y(x, y, exps):
    """Return the first y derivatives."""
    x, y, a, b = _prepare(x, y, exps)
    result = np.zeros(x.shape[:-1] + a.shape, dtype=np.float64)
    mask = b >= 1
    a, b = a[mask], b[mask]
    result[..., mask] = b * x**a * y**(b-1)
    return result


def get_poly_xx(x, y, exps):
    """Return the second x derivatives."""
    x, y, a, b = _prepare(x, y, exps)
    result = np.zeros(x.shape[:-1] + a.shape, dtype=np.float64)
    mask = a >= 2
    a, b = a[mask], b[mask]
    result[..., mask] = a * (a-1) * x**(a-2) * y**b
    return result


def get_poly_yy(x, y, exps):
    """Return the second y derivatives."""
    x, y, a, b = _prepare(x, y, exps)
    result = np.zeros(x.shape[:-1] + a.shape, dtype=np.float64)
    mask = b >= 2
    a, b = a[mask], b[mask]
    result[..., mask] = b * (b-1) * x**a * y**(b-2)
    return result


def get_poly_xy(x, y, exps):
    """Return the mixed xy derivatives."""
    x, y, a, b = _prepare(x, y, exps)
    result = np.zeros(x.shape[:-1] + a.shape, dtype=np.float64)
    mask = (a >= 1) & (b >= 1)
    a, b = a[mask], b[mask]
    result[..., mask] = a * b * x**(a-1) * y**(b-1)
    return result


def monomial_exponents(m):
    """Return powers [a, b] ordered by total degree, then increasing a."""
    m = _integer(m, "m", 0)
    return np.array([(a, s-a) for s in range(m+1) for a in range(s+1)], dtype=int)


def falling_factorial(n, k):
    """Return n*(n-1)*...*(n-k+1) for nonnegative integer n and k.

    n may be a scalar or array; the float64 result has the same shape.
    Return zero where n < k, and one everywhere when k is zero.
    """
    k = _integer(k, "k", 0)
    n = np.asarray(n)
    if (n.dtype.kind not in "iuf" or not np.all(np.isfinite(n))
            or np.any(n < 0) or np.any(n != np.floor(n))):
        raise ValueError("n must contain nonnegative integers")
    n = n.astype(np.float64, copy=False)
    result = np.ones(n.shape, dtype=np.float64)
    if k == 0:
        return result
    mask = n >= k
    result[~mask] = 0
    if np.any(mask):
        values = n[mask]
        product = np.ones(values.shape, dtype=np.float64)
        for j in range(k):
            product *= values-j
        result[mask] = product
    return result


def eval_monomial_deriv(exps, x0, y0, ix, iy):
    """Return derivatives of arbitrary-degree monomials at order (ix, iy).

    ix and iy are nonnegative integers. Shapes follow get_poly, with the
    monomial axis last; exps is the first argument.
    Orders (0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1) give the same
    values and shapes as get_poly, get_poly_x, get_poly_y, get_poly_xx,
    get_poly_yy, get_poly_xy, respectively.
    For a polynomial, multiply the result by its coefficient vector:
    eval_monomial_deriv(exps, x0, y0, ix, iy) @ coefficients.
    """
    ix = _integer(ix, "ix", 0)
    iy = _integer(iy, "iy", 0)
    x, y, a, b = _prepare(x0, y0, exps)
    result = np.zeros(x.shape[:-1] + a.shape, dtype=np.float64)
    mask = (a >= ix) & (b >= iy)
    a, b = a[mask], b[mask]
    coefficient = falling_factorial(a, ix) * falling_factorial(b, iy)
    result[..., mask] = coefficient * x**(a-ix) * y**(b-iy)
    return result
