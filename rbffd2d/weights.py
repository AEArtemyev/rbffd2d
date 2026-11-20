"""Polynomially augmented RBF-FD weights."""

import numpy as np

from .neighbors import find_neighbors, _points
from .polynomials import (
    monomial_exponents,
    get_poly, get_poly_x, get_poly_y, get_poly_xx, get_poly_yy, get_poly_xy,
)
from .stencil import get_poly_order


POLYNOMIALS = (get_poly, get_poly_x, get_poly_y, get_poly_xx, get_poly_yy, get_poly_xy)
OPERATORS = ("f", "dx", "dy", "dxx", "dyy", "dxy")


def build_weights(nodes, rbf, search_mode="polyOrder", stencil_size=15,
                  poly_order=2, rho=2, *, tol=1e-6):
    """Return (neighbors, weights) for all nodes.

    Each weights[i] has shape (len(neighbors[i]), 6), with columns
    [f, dx, dy, dxx, dyy, dxy]. In mode 'stencilSize', choose the polynomial
    degree from the actual neighbor count; in 'polyOrder', keep poly_order.
    Undefined kernel values raise ValueError; singular systems raise
    numpy.linalg.LinAlgError.
    """
    nodes = _points(nodes, "nodes")
    neighbors = find_neighbors(
        nodes, usage=search_mode, stencil_size=stencil_size,
        poly_order=poly_order, rho=rho, tol=tol,
    )
    weights = []
    poly_data = {}
    for i, ids in enumerate(neighbors):
        degree = get_poly_order(len(ids), 2, rho) if search_mode == "stencilSize" else poly_order
        if degree not in poly_data:
            exps = monomial_exponents(degree)
            poly_rhs = np.column_stack([function(0, 0, exps) for function in POLYNOMIALS])
            poly_data[degree] = exps, poly_rhs
        exps, poly_rhs = poly_data[degree]
        points = nodes[ids] - nodes[i]
        x, y = points.T
        phi = rbf.f(x[:, None]-x, y[:, None]-y)
        p = get_poly(x, y, exps)
        kernel_rhs = np.column_stack([getattr(rbf, name)(-x, -y) for name in OPERATORS])
        matrix = np.block([[phi, p], [p.T, np.zeros((len(exps), len(exps)))]])
        rhs = np.vstack((kernel_rhs, poly_rhs))
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
            raise ValueError(f"nonfinite RBF-FD system at node {i}")
        weights.append(np.linalg.solve(matrix, rhs)[:len(ids)])
    return neighbors, weights
