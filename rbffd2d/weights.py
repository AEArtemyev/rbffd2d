"""Polynomially augmented RBF-FD weights."""

import numpy as np

from .neighbors import find_neighbors, _points, _search_tree
from .polynomials import (
    monomial_exponents,
    get_poly, get_poly_x, get_poly_y, get_poly_xx, get_poly_yy, get_poly_xy,
)
from .stencil import get_poly_order


POLYNOMIALS = (get_poly, get_poly_x, get_poly_y, get_poly_xx, get_poly_yy, get_poly_xy)
OPERATORS = ("f", "dx", "dy", "dxx", "dyy", "dxy")


def _poly_data(degree):
    exps = monomial_exponents(degree)
    rhs = np.column_stack([function(0, 0, exps) for function in POLYNOMIALS])
    return exps, rhs


def _solve_weights(points, rbf, poly_matrix, poly_rhs):
    x, y = points.T
    phi = rbf.f(x[:, None]-x, y[:, None]-y)
    kernel_rhs = np.column_stack([getattr(rbf, name)(-x, -y) for name in OPERATORS])
    if not all(np.all(np.isfinite(a)) for a in (phi, kernel_rhs, poly_matrix, poly_rhs)):
        raise ValueError("nonfinite RBF-FD system")

    # Balance the blocks without changing the weights or regularizing the system.
    rbf_scale = np.max(np.abs(phi))
    poly_scale = np.max(np.abs(poly_matrix), axis=0)
    if np.any(poly_scale == 0):
        raise np.linalg.LinAlgError("zero polynomial block scale")
    # A zero RBF block can still give an invertible augmented system.
    if rbf_scale == 0:
        rbf_scale = 1.0
    scaled_poly = poly_matrix / poly_scale
    nm = poly_matrix.shape[1]
    matrix = np.block([
        [phi / rbf_scale, scaled_poly],
        [scaled_poly.T, np.zeros((nm, nm))],
    ])
    rhs = np.vstack((kernel_rhs / rbf_scale, poly_rhs / poly_scale[:, None]))
    return np.linalg.solve(matrix, rhs)[:len(points)]


def compute_weights(points, col_point, rbf, poly_order=2):
    """Return (N, 6) weights on an explicit stencil at col_point.

    points has shape (N, 2), col_point has shape (2,). Columns are
    [f, dx, dy, dxx, dyy, dxy]. No neighbor search or stencil expansion is
    performed. Nonfinite systems raise ValueError; rank-deficient polynomial
    blocks and singular systems raise numpy.linalg.LinAlgError.
    """
    points = _points(points, "points")
    col_point = np.asarray(col_point, dtype=np.float64)
    if col_point.shape != (2,) or not np.all(np.isfinite(col_point)):
        raise ValueError("col_point must be a finite array with shape (2,)")
    exps, poly_rhs = _poly_data(poly_order)
    points = points - col_point
    poly_matrix = get_poly(*points.T, exps)
    if len(points) < len(exps) or np.linalg.matrix_rank(poly_matrix) < len(exps):
        raise np.linalg.LinAlgError("rank-deficient polynomial block on the given stencil")
    return _solve_weights(points, rbf, poly_matrix, poly_rhs)


def build_weights(nodes, rbf, search_mode="polyOrder", stencil_size=15,
                  poly_order=2, rho=2.5, *, tol=1e-6, search_data=None, tree=None):
    """Return (neighbors, weights) at nodes, searching in search_data.

    Pass tree to reuse a cKDTree instead of supplying search_data. The same
    tree is used for initial search and expansion. Do not modify the tree's
    underlying coordinates. If neither is given or search_data is empty,
    search among nodes. Indices refer to search_data or tree.data.
    Each weights[i] has shape (len(neighbors[i]), 6),
    with columns [f, dx, dy, dxx, dyy, dxy]. In mode 'stencilSize', choose
    the polynomial degree from the initial neighbor count; in 'polyOrder',
    keep poly_order. Expand rank-deficient stencils by complete distance
    shells, keeping the degree fixed. Raise numpy.linalg.LinAlgError if all
    search points are exhausted or the full system is singular. Undefined
    kernel values raise ValueError.
    """
    nodes = _points(nodes, "nodes")
    tree = _search_tree(nodes, search_data, tree)
    search = tree.data
    neighbors = find_neighbors(
        nodes, usage=search_mode, stencil_size=stencil_size,
        poly_order=poly_order, rho=rho, tol=tol, tree=tree,
    )
    weights = []
    poly_data = {}
    for i, ids in enumerate(neighbors):
        degree = get_poly_order(len(ids), 2, rho) if search_mode == "stencilSize" else poly_order
        if degree not in poly_data:
            poly_data[degree] = _poly_data(degree)
        exps, poly_rhs = poly_data[degree]
        while True:
            points = search[ids] - nodes[i]
            poly_matrix = get_poly(*points.T, exps)
            if np.linalg.matrix_rank(poly_matrix) == len(exps):
                break
            if len(ids) == len(search):
                raise np.linalg.LinAlgError(
                    f"rank-deficient polynomial block at query {i}: search points exhausted"
                )
            distance, _ = tree.query(nodes[i], k=[len(ids)+1])
            radius = np.nextafter(distance[0] + tol, np.inf)
            ids = np.asarray(tree.query_ball_point(nodes[i], radius), dtype=int)
            delta = search[ids] - nodes[i]
            distances = np.hypot(delta[:, 0], delta[:, 1])
            ids = ids[np.lexsort((ids, distances))]
        neighbors[i] = ids
        try:
            weights.append(_solve_weights(points, rbf, poly_matrix, poly_rhs))
        except (ValueError, np.linalg.LinAlgError) as error:
            raise type(error)(f"{error} at query {i}") from error
    return neighbors, weights
