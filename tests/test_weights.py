"""Explicit stencils, separate query points, scaling and rank recovery."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial import cKDTree

from rbf2d import PHS, Gauss, RadialPower
from rbffd2d import (
    build_weights, compute_weights, find_neighbors, monomial_exponents,
    get_poly, get_poly_x, get_poly_y, get_poly_xx, get_poly_yy, get_poly_xy,
)


POLYNOMIALS = (get_poly, get_poly_x, get_poly_y, get_poly_xx, get_poly_yy, get_poly_xy)


class CountingTree(cKDTree):
    builds = 0

    def __init__(self, *args, **kwargs):
        type(self).builds += 1
        super().__init__(*args, **kwargs)


class TestExplicitWeights(unittest.TestCase):
    def setUp(self):
        self.points = np.random.default_rng(17).uniform(-1, 1, (24, 2))
        self.query = np.array([0.17, -0.23])

    def test_polynomial_reproduction(self):
        exps = monomial_exponents(2)
        for rbf in (PHS(3), Gauss(beta=2, epsilon=2)):
            weights = compute_weights(self.points, self.query, rbf)
            expected = np.column_stack([f(*self.query, exps) for f in POLYNOMIALS])
            assert_allclose(get_poly(*self.points.T, exps).T @ weights,
                            expected, atol=1e-10, rtol=1e-10)
            translated = compute_weights(self.points + [2, -3], self.query + [2, -3], rbf)
            assert_allclose(translated, weights, atol=1e-9, rtol=1e-9)

    def test_small_stencil_scaling(self):
        baseline = compute_weights(self.points, self.query, PHS(3))
        orders = np.array([0, 1, 1, 2, 2, 2])
        for scale in (1e-4, 1e-6):
            weights = compute_weights(scale*self.points, scale*self.query, PHS(3))
            assert_allclose(weights * scale**orders, baseline, atol=1e-10, rtol=1e-9)

    def test_explicit_stencil_preserves_order(self):
        weights = compute_weights(self.points, self.query, PHS(3))
        permutation = np.random.default_rng(23).permutation(len(self.points))
        permuted = compute_weights(self.points[permutation], self.query, PHS(3))
        assert_allclose(permuted, weights[permutation], atol=1e-10, rtol=1e-9)

    def test_close_nodes_reproduce_polynomials(self):
        points = self.points.copy()
        points[1] = points[0] + [1e-3, -1e-3]
        weights = compute_weights(points, self.query, PHS(3))
        exps = monomial_exponents(2)
        expected = np.column_stack([f(*self.query, exps) for f in POLYNOMIALS])
        assert_allclose(get_poly(*points.T, exps).T @ weights,
                        expected, atol=1e-9, rtol=1e-9)

    def test_invalid_input(self):
        for point in ([0], [[0, 0]], [np.nan, 0], [0, np.inf]):
            with self.subTest(point=point), self.assertRaises(ValueError):
                compute_weights(self.points, point, PHS(3))
        with self.assertRaises(ValueError):
            compute_weights([[np.nan, 0]], [0, 0], PHS(3))
        with self.assertRaises(ValueError):
            compute_weights(self.points, self.query, PHS(3), -1)
        with self.assertRaises(ValueError):
            compute_weights(self.points, self.points[0], RadialPower(1))
        for points in (np.empty((0, 2)), [[0, 0]], np.zeros((15, 2))):
            with self.assertRaises(np.linalg.LinAlgError):
                compute_weights(points, [0, 0], PHS(3))

    def test_singular_system_with_full_rank_polynomials(self):
        points = np.vstack((self.points, self.points[0]))
        with self.assertRaises(np.linalg.LinAlgError):
            compute_weights(points, self.query, PHS(3))

    def test_single_node_constant_polynomial(self):
        expected = [[1, 0, 0, 0, 0, 0]]
        assert_array_equal(compute_weights([[0, 0]], [0, 0], PHS(3), 0), expected)
        neighbors, weights = build_weights([[0, 0]], PHS(3), poly_order=0, rho=1)
        assert_array_equal(neighbors[0], [0])
        assert_array_equal(weights[0], expected)


class TestStencilExpansion(unittest.TestCase):
    def setUp(self):
        # The five closest points are collinear. The next shell restores rank.
        self.search = np.array([
            [0, 0], [0.1, 0], [-0.1, 0], [0.2, 0], [-0.2, 0],
            [0, 1], [0, -1], [2, 0],
        ])
        self.query = np.array([[0, 0]])

    def test_expansion_in_both_modes(self):
        for mode in ("polyOrder", "stencilSize"):
            options = dict(search_mode=mode, stencil_size=3, poly_order=1,
                           rho=1, tol=0, search_data=self.search)
            neighbors, weights = build_weights(self.query, PHS(3), **options)
            # Degree remains one even after the final stencil has seven nodes.
            assert_array_equal(neighbors[0], np.arange(7))
            expected = compute_weights(self.search[:7], self.query[0], PHS(3), 1)
            assert_allclose(weights[0], expected, atol=1e-12, rtol=1e-12)

    def test_expansion_tolerance(self):
        self.search[6] = [0, -1.0000005]
        neighbors, _ = build_weights(self.query, PHS(3), poly_order=1, rho=1,
                                     search_data=self.search)
        assert_array_equal(neighbors[0], np.arange(7))
        neighbors, _ = build_weights(self.query, PHS(3), poly_order=1, rho=1,
                                     tol=0, search_data=self.search)
        assert_array_equal(neighbors[0], np.arange(6))

    def test_quadratic_expansion_through_multiple_shells(self):
        line = np.column_stack((np.linspace(-0.7, 0.7, 15), np.zeros(15)))
        search = np.vstack((line, [[0, 1], [0, -1], [1, 1], [-1, 1]]))
        neighbors, weights = build_weights(self.query, PHS(3), search_data=search)
        self.assertEqual(set(neighbors[0]), set(range(len(search))))
        exps = monomial_exponents(2)
        expected = np.column_stack([f(0, 0, exps) for f in POLYNOMIALS])
        assert_allclose(get_poly(*search[neighbors[0]].T, exps).T @ weights[0],
                        expected, atol=1e-10, rtol=1e-10)

    def test_exhausted_search(self):
        with self.assertRaisesRegex(np.linalg.LinAlgError, "query 0.*exhausted"):
            build_weights(self.query, PHS(3), poly_order=1, rho=1,
                          search_data=self.search[:5])

    def test_separate_query_and_search(self):
        search = np.random.default_rng(11).uniform(-1, 1, (35, 2))
        query = np.array([[0.15, -0.17], [0.38, 0.26]])
        for mode in ("polyOrder", "stencilSize"):
            neighbors, weights = build_weights(query, PHS(3), search_mode=mode,
                                               search_data=search)
            expected_ids = find_neighbors(query, search, usage=mode)
            for point, ids, w, expected in zip(query, neighbors, weights, expected_ids):
                assert_array_equal(ids, expected)
                assert_allclose(w, compute_weights(search[ids], point, PHS(3)),
                                atol=1e-12, rtol=1e-12)
        self.assertEqual(build_weights(np.empty((0, 2)), PHS(3), search_data=search), ([], []))


class TestReusableSearchTree(unittest.TestCase):
    def setUp(self):
        self.search = np.array([
            [0, 0], [0.1, 0], [-0.1, 0], [0.2, 0], [-0.2, 0],
            [0, 1], [0, -1], [2, 0],
        ])
        self.query = np.zeros((2, 2))

    def test_one_tree_for_initial_search_and_all_expansions(self):
        CountingTree.builds = 0
        with patch("rbffd2d.neighbors.cKDTree", CountingTree):
            neighbors, _ = build_weights(self.query, PHS(3), poly_order=1,
                                         rho=1, search_data=self.search)
        self.assertEqual(CountingTree.builds, 1)
        for ids in neighbors:
            assert_array_equal(ids, np.arange(7))

    def test_supplied_tree_reused_across_calls_and_expansions(self):
        tree = CountingTree(self.search, copy_data=True)
        CountingTree.builds = 0
        with patch("rbffd2d.neighbors.cKDTree", CountingTree):
            for mode in ("polyOrder", "stencilSize"):
                ids, weights = build_weights(self.query, PHS(3), search_mode=mode,
                                             poly_order=1, stencil_size=3, rho=1, tree=tree)
                for neighbors, w in zip(ids, weights):
                    assert_array_equal(neighbors, np.arange(7))
                    expected = compute_weights(tree.data[neighbors], [0, 0], PHS(3), 1)
                    assert_allclose(w, expected, atol=1e-12, rtol=1e-12)
            initial = find_neighbors(self.query, tree=tree, usage="stencilSize", stencil_size=3)
            for neighbors in initial:
                assert_array_equal(neighbors, [0, 1, 2])
        self.assertEqual(CountingTree.builds, 0)

    def test_invalid_tree_arguments(self):
        tree = cKDTree(self.search)
        for function in (find_neighbors, build_weights):
            args = (self.query,) if function is find_neighbors else (self.query, PHS(3))
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(ValueError, "either search_data or tree"):
                    function(*args, tree=tree, search_data=self.search)
                with self.assertRaises(TypeError):
                    function(*args, tree=self.search)
                with self.assertRaisesRegex(ValueError, "two-dimensional"):
                    function(*args, tree=cKDTree(np.zeros((20, 3))))


if __name__ == "__main__":
    unittest.main()
