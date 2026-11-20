"""RBF-FD stencil size and polynomial degree selection."""

from math import ceil, comb, isfinite
from numbers import Real


def _integer(value, name, minimum):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not isfinite(value) or value < minimum or value != int(value)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _check_rho(rho):
    if (isinstance(rho, bool) or not isinstance(rho, Real)
            or not isfinite(rho) or rho <= 0):
        raise ValueError("rho must be finite and positive")


def get_rbf_fd_stencil_size(m, d, rho=2.5, mode="stable"):
    """Return (n, Nm), where Nm = comb(m+d, d).

    m is a nonnegative degree; d is a positive dimension.
    mode='min' gives n=Nm; mode='stable' gives n=ceil(rho*Nm).
    rho must be finite and positive.
    """
    m = _integer(m, "m", 0)
    d = _integer(d, "d", 1)
    _check_rho(rho)
    if mode not in ("min", "stable"):
        raise ValueError("mode must be 'min' or 'stable'")
    nm = comb(m+d, d)
    n = nm if mode == "min" else ceil(rho * nm)
    return n, nm


def get_poly_order(n, d, rho=2.5, mode="stable"):
    """Return the largest degree m with comb(m+d, d) <= the point budget.

    n is a positive stencil size; d must be 2 or 3.
    The budget is n in mode='max', or n/rho in mode='stable'.
    rho must be finite and positive. Raise ValueError if the budget is < 1.
    """
    n = _integer(n, "n", 1)
    d = _integer(d, "d", 1)
    if d not in (2, 3):
        raise ValueError("d must be 2 or 3")
    _check_rho(rho)
    if mode not in ("max", "stable"):
        raise ValueError("mode must be 'max' or 'stable'")
    budget = n if mode == "max" else n / rho
    if budget < 1:
        raise ValueError("n/rho must be >= 1 to include a constant polynomial")
    m = 0
    while comb(m+1+d, d) <= budget:
        m += 1
    return m
