"""Visibility, complete shells and RBF-FD expansion on polygonal domains."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial import cKDTree

from rbf2d import PHS
from rbffd2d import build_weights, compute_weights, find_neighbors

try:
    import shapely
    from shapely.geometry import Polygon, MultiPolygon
except ImportError:
    shapely = None


class TestOptionalGeometry(unittest.TestCase):
    def test_search_without_shapely(self):
        with patch.dict("sys.modules", {"shapely": None}):
            ids = find_neighbors([[0, 0]], usage="stencilSize", stencil_size=1)
            assert_array_equal(ids[0], [0])
            _, weights = build_weights([[0, 0]], PHS(3), poly_order=0, rho=1)
            assert_array_equal(weights[0], [[1, 0, 0, 0, 0, 0]])
            with self.assertRaisesRegex(ImportError, r"rbffd2d\[geometry\]"):
                find_neighbors([[0, 0]], domain=object())


@unittest.skipIf(shapely is None, "requires the geometry extra")
class TestDomainNeighbors(unittest.TestCase):
    def setUp(self):
        self.square = Polygon([[-3, -3], [3, -3], [3, 3], [-3, 3]])
        self.domain = Polygon(self.square.exterior.coords,
                              holes=[[[-0.5, -0.5], [0.5, -0.5],
                                      [0.5, 0.5], [-0.5, 0.5]]])

    def neighbors(self, query, search, size, **kwargs):
        return find_neighbors(query, search, domain=self.domain,
                              usage="stencilSize", stencil_size=size, **kwargs)

    def test_hidden_nodes_do_not_fill_stencil(self):
        search = [[-1, 0], [1, 0], [-1, 2.5], [-1, -2.5]]
        ids = self.neighbors([[-1, 0]], search, 2, tol=0)[0]
        assert_array_equal(ids, [0, 2, 3])

    def test_equal_distance_visible_shell_and_tolerance(self):
        search = [[-1, 0], [1, 0], [-1, 2], [-1, -2], [-1, 2.0000005]]
        assert_array_equal(self.neighbors([[-1, 0]], search, 2, tol=0)[0], [0, 2, 3])
        assert_array_equal(self.neighbors([[-1, 0]], search, 2)[0], [0, 2, 3, 4])

    def test_convex_domain_matches_existing_search_and_weights(self):
        x, y = np.meshgrid(np.linspace(-3, 3, 7), np.linspace(-3, 3, 7))
        search = np.column_stack((x.ravel(), y.ravel()))
        query = np.array([[-3, -3], [0, 0], [3, 0], [0.17, -0.23]])
        tree = cKDTree(search)
        expected_ids, expected_weights = build_weights(query, PHS(3), tree=tree)
        ids, weights = build_weights(query, PHS(3), tree=tree, domain=self.square)
        for actual, expected, w, expected_w in zip(ids, expected_ids, weights, expected_weights):
            assert_array_equal(actual, expected)
            assert_allclose(w, expected_w, atol=1e-12, rtol=1e-12)
        self.assertTrue(shapely.is_prepared(self.square))
        for actual, expected in zip(find_neighbors(query, tree=tree, domain=self.square),
                                    find_neighbors(query, tree=tree)):
            assert_array_equal(actual, expected)

    def test_boundary_edges_vertex_tangency_and_self(self):
        # From (-1, 0), the segment to (0, 1) grazes the hole corner (-.5, .5).
        search = [[-1, 0], [-0.5, 0], [0, 1], [1, 0]]
        assert_array_equal(self.neighbors([[-1, 0]], search, 3, tol=0)[0], [0, 1, 2])
        # Travel along a hole edge and an exterior edge, including coincident points.
        for query, search in (([[-0.5, -0.5]], [[-0.5, -0.5], [0.5, -0.5]]),
                              ([[-3, -3]], [[-3, -3], [3, -3]])):
            assert_array_equal(self.neighbors(query, search, 2, tol=0)[0], [0, 1])
        ids, weights = build_weights([[-0.5, -0.5]], PHS(3), domain=self.domain,
                                     poly_order=0, rho=1)
        assert_array_equal(ids[0], [0])
        assert_array_equal(weights[0], [[1, 0, 0, 0, 0, 0]])

    def test_nonconvex_exterior_blocks_neighbors(self):
        domain = Polygon([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]])
        search = [[0.5, 2], [2, 0.5], [0.5, 0.5]]
        ids = find_neighbors([[0.5, 2]], search, domain=domain,
                             usage="stencilSize", stencil_size=2, tol=0)[0]
        assert_array_equal(ids, [0, 2])

    def test_adaptive_search_matches_full_visible_search(self):
        rng = np.random.default_rng(71)
        search = rng.uniform(-3, 3, (200, 2))
        search = search[shapely.covers(self.domain, shapely.points(search))]
        query = np.array([[-0.6, 0], [0.6, 0], [0, 2], [-3, -3]])
        tree = cKDTree(search)
        for size in (1, 7, 30):
            for tol in (0, 0.2):
                actual = find_neighbors(query, tree=tree, domain=self.domain,
                                        usage="stencilSize", stencil_size=size, tol=tol)
                for point, ids in zip(query, actual):
                    segments = shapely.linestrings(np.stack((np.broadcast_to(point, search.shape),
                                                             search), axis=1))
                    eligible = np.flatnonzero(shapely.covers(self.domain, segments))
                    distances = np.hypot(*(search[eligible] - point).T)
                    order = np.lexsort((eligible, distances))
                    radius = distances[order[size-1]] + tol
                    expected = eligible[order][distances[order] <= radius]
                    assert_array_equal(ids, expected)

    def test_distance_tolerance_does_not_open_a_thin_obstacle(self):
        domain = Polygon([[-1, -1], [1, -1], [1, 1], [-1, 1]],
                         holes=[[[1e-7, -0.5], [2e-7, -0.5],
                                 [2e-7, 0.5], [1e-7, 0.5]]])
        ids = find_neighbors([[0, 0]], [[3e-7, 0], [0, 1]], domain=domain,
                             usage="stencilSize", stencil_size=1, tol=1)[0]
        assert_array_equal(ids, [1])

    def test_insufficient_visible_nodes_and_empty_queries(self):
        with self.assertRaisesRegex(ValueError, "query 0.*visible"):
            self.neighbors([[-1, 0]], [[-1, 0], [1, 0]], 2)
        with self.assertRaisesRegex(ValueError, "query 0.*visible"):
            build_weights([[-1, 0]], PHS(3), domain=self.domain,
                          search_data=[[-1, 0], [1, 0]], poly_order=0, rho=2)
        self.assertEqual(self.neighbors(np.empty((0, 2)), [[-1, 0]], 1), [])

    def test_invalid_domains_and_outside_points(self):
        invalid = [Polygon(), Polygon([[0, 0], [1, 1], [0, 1], [1, 0]]),
                   Polygon(self.square.exterior.coords,
                           holes=[[[4, 0], [5, 0], [4, 1]]]),
                   Polygon([[0, 0, 1], [1, 0, 1], [0, 1, 1]])]
        for domain in invalid:
            with self.subTest(domain=domain), self.assertRaises(ValueError):
                find_neighbors([[-1, 0]], domain=domain, usage="stencilSize", stencil_size=1)
        for domain in ([], MultiPolygon([self.square])):
            with self.subTest(domain=domain), self.assertRaises(TypeError):
                find_neighbors([[-1, 0]], domain=domain)
        for query, search in (([[0, 0]], [[-1, 0]]),
                              ([[-1, 0]], [[0, 0]]),
                              ([[3.0000001, 0]], [[-1, 0]])):
            with self.subTest(query=query, search=search), self.assertRaisesRegex(ValueError, "outside"):
                self.neighbors(query, search, 1, tol=1)

    def test_rank_expansion_uses_only_visible_nodes_and_complete_shells(self):
        query = [[-1, 0]]
        search = np.array([[-1, 0], [-1.1, 0], [-0.9, 0],
                           [0.6, 0], [-1, 2], [-1, -2], [-1, 2.0000005]])
        tree = cKDTree(search)
        for mode in ("polyOrder", "stencilSize"):
            for tol, expected in ((0, [0, 2, 1, 4, 5]), (1e-6, [0, 2, 1, 4, 5, 6])):
                with self.subTest(mode=mode, tol=tol):
                    ids, weights = build_weights(query, PHS(3), domain=self.domain,
                                                 tree=tree, search_mode=mode, poly_order=1,
                                                 stencil_size=3, rho=1, tol=tol)
                    assert_array_equal(ids[0], expected)
                    assert_allclose(weights[0], compute_weights(search[expected], query[0], PHS(3), 1),
                                    atol=1e-12, rtol=1e-12)
        # Hidden off-axis nodes cannot restore rank when visible points are collinear.
        search = [[-1, 0], [-1.1, 0], [-0.9, 0], [1, 0.2], [1, -0.2]]
        with self.assertRaisesRegex(np.linalg.LinAlgError, "query 0.*exhausted"):
            build_weights(query, PHS(3), domain=self.domain,
                          search_data=search, poly_order=1, rho=1)

    def test_only_center_to_node_visibility_is_required(self):
        # Both sides of the hole are visible from the center above it.
        query = [[0, 2]]
        search = [[-1, 0], [1, 0], [0, 2]]
        ids, weights = build_weights(query, PHS(3), domain=self.domain,
                                     search_data=search, poly_order=1, rho=1)
        assert_array_equal(ids[0], [2, 0, 1])
        assert_allclose(weights[0], compute_weights(np.array(search)[ids[0]], query[0], PHS(3), 1),
                        atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
