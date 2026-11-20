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


def _prepare_domain(domain, query, search):
    if domain is None:
        return
    try:
        import shapely
        from shapely.geometry import Polygon
    except ImportError as error:
        raise ImportError("domain requires Shapely; install rbffd2d[geometry]") from error
    if not isinstance(domain, Polygon):
        raise TypeError("domain must be a shapely.Polygon")
    if domain.is_empty or domain.has_z or not domain.is_valid:
        raise ValueError("domain must be a nonempty valid two-dimensional polygon")
    # Preparation is cached on the same polygon without changing its coordinates.
    shapely.prepare(domain)
    for name, points in (("query_data", query), ("search_data", search)):
        inside = shapely.covers(domain, shapely.points(points))
        outside = np.flatnonzero(~inside)
        if outside.size:
            raise ValueError(f"{name}[{outside[0]}] is outside domain")


def _neighbors_in_radius(point, tree, radius, domain):
    ids = np.asarray(tree.query_ball_point(point, radius), dtype=int)
    if domain is not None and ids.size:
        import shapely

        targets = tree.data[ids]
        # A self-neighbor is valid; a zero-length LineString is not.
        same = np.all(targets == point, axis=1)
        visible = same.copy()
        segments = np.stack((np.broadcast_to(point, targets[~same].shape),
                             targets[~same]), axis=1)
        visible[~same] = shapely.covers(domain, shapely.linestrings(segments))
        ids = ids[visible]
    delta = tree.data[ids] - point
    distances = np.hypot(delta[:, 0], delta[:, 1])
    return ids[np.lexsort((ids, distances))]


def _neighbor_ids(point, tree, size, tol, domain):
    """Find a complete shell, or all eligible nodes if size is unreachable."""
    count = min(size, tree.n)
    while True:
        distance, _ = tree.query(point, k=[count])
        radius = np.nextafter(distance[0] + tol, np.inf)
        ids = _neighbors_in_radius(point, tree, radius, domain)
        if domain is None:
            return ids
        if len(ids) >= size:
            # Candidate growth can overshoot the kth visible node. Recompute
            # its radius and fetch its full shell, including tolerance.
            delta = tree.data[ids[size-1]] - point
            radius = np.nextafter(np.hypot(*delta) + tol, np.inf)
            return _neighbors_in_radius(point, tree, radius, domain)
        if count == tree.n:
            return ids
        count = min(tree.n, 2 * count)


def find_neighbors(query_data, search_data=None, *, usage="polyOrder",
                   stencil_size=15, poly_order=2, rho=2.5, tol=1e-6, tree=None,
                   domain=None):
    """Return a list of zero-based neighbor index arrays, sorted by distance.

    Pass tree to reuse a cKDTree instead of supplying search_data. Indices
    then refer to tree.data. Do not modify the tree's underlying coordinates.
    Search among query_data if both are omitted or search_data is empty.
    usage='stencilSize' uses stencil_size; usage='polyOrder' uses the size
    for poly_order and rho. Include all points within the kth distance + tol.
    Query points belonging to the search set are included.
    With domain, pass a valid nonempty 2D shapely.Polygon (requires the
    geometry extra). Only directly visible nodes are eligible; boundary
    contacts and travel along edges are allowed. All query and search points
    must be inside or on the boundary. tol affects distances only, not geometry.
    Raise ValueError if there are too few visible nodes. The polygon is prepared
    in place for reuse without changing its coordinates.
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
    _prepare_domain(domain, query, tree.data)
    if size > tree.n:
        raise ValueError("stencil size exceeds the number of search points")
    if domain is not None:
        neighbors = []
        for i, point in enumerate(query):
            ids = _neighbor_ids(point, tree, size, tol, domain)
            if len(ids) < size:
                raise ValueError(f"query {i}: not enough visible search points "
                                 f"({len(ids)} available, {size} required)")
            neighbors.append(ids)
        return neighbors
    # Keep batched tree queries for the original mode.
    distances, _ = tree.query(query, k=[size])
    radii = np.nextafter(distances[:, 0] + tol, np.inf)
    neighbors = []
    for point, ids in zip(query, tree.query_ball_point(query, radii)):
        ids = np.asarray(ids, dtype=int)
        delta = tree.data[ids] - point
        distance = np.hypot(delta[:, 0], delta[:, 1])
        neighbors.append(ids[np.lexsort((ids, distance))])
    return neighbors
