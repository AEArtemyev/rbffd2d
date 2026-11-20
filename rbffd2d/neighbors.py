"""Nearest neighbors with complete boundary shells."""

from numbers import Real

import numpy as np
from scipy.spatial import cKDTree

from .stencil import _integer, get_rbf_fd_stencil_size


def _points(data, name):
    points = np.asarray(data, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be a finite array with shape (N, 2)")
    return points


def _search_tree(query, search_data, tree):
    if tree is not None:
        if search_data is not None:
            raise ValueError("pass either search_data or tree, not both")
        if not isinstance(tree, cKDTree):
            raise TypeError("tree must be a scipy.spatial.cKDTree")
        if tree.m != 2:
            raise ValueError("tree must contain two-dimensional points")
        return tree
    search = query if search_data is None or np.size(search_data) == 0 else _points(search_data, "search_data")
    return cKDTree(search)


def find_neighbors(query_data, search_data=None, *, usage="polyOrder",
                   stencil_size=15, poly_order=2, rho=2.5, tol=1e-6, tree=None):
    """Return a list of zero-based neighbor index arrays, sorted by distance.

    Pass tree to reuse a cKDTree instead of supplying search_data. Indices
    then refer to tree.data. Do not modify the tree's underlying coordinates.
    Search among query_data if both are omitted or search_data is empty.
    usage='stencilSize' uses stencil_size; usage='polyOrder' uses the size
    for poly_order and rho. Include all points within the kth distance + tol.
    Query points belonging to the search set are included.
    """
    query = _points(query_data, "query_data")
    if (isinstance(tol, (bool, np.bool_)) or not isinstance(tol, Real)
            or not np.isfinite(tol) or tol < 0):
        raise ValueError("tol must be finite and nonnegative")
    if usage == "stencilSize":
        size = _integer(stencil_size, "stencil_size", 1)
    elif usage == "polyOrder":
        size, _ = get_rbf_fd_stencil_size(poly_order, 2, rho)
    else:
        raise ValueError("usage must be 'stencilSize' or 'polyOrder'")
    tree = _search_tree(query, search_data, tree)
    search = tree.data
    if size > tree.n:
        raise ValueError("stencil size exceeds the number of search points")
    distances, _ = tree.query(query, k=[size])
    radii = np.nextafter(distances[:, 0] + tol, np.inf)
    neighbors = []
    for point, ids in zip(query, tree.query_ball_point(query, radii)):
        ids = np.asarray(ids, dtype=int)
        delta = search[ids] - point
        distance = np.hypot(delta[:, 0], delta[:, 1])
        neighbors.append(ids[np.lexsort((ids, distance))])
    return neighbors
