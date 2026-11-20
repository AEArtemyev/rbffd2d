"""Neighbor search and RBF-FD reproduction checks."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from rbf2d import PHS, Gauss, RadialPower
from rbffd2d import (
    find_neighbors, build_weights, monomial_exponents, get_poly_order,
    get_poly, get_poly_x, get_poly_y, get_poly_xx, get_poly_yy, get_poly_xy,
)


POLYNOMIALS = (get_poly, get_poly_x, get_poly_y, get_poly_xx, get_poly_yy, get_poly_xy)
OPERATORS = ("f", "dx", "dy", "dxx", "dyy", "dxy")


class TestNeighbors(unittest.TestCase):
    def test_boundary_shell(self):
        points = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [2, 0]]
        ids = find_neighbors([[0, 0]], points, usage="stencilSize", stencil_size=2, tol=0)[0]
        assert_array_equal(ids, [0, 1, 2, 3, 4])
        self.assertEqual(len(find_neighbors(points, usage="polyOrder", poly_order=0)[0]), 5)
        self.assertEqual(find_neighbors(points, usage="stencilSize", stencil_size=1)[3][0], 3)

    def test_tolerance(self):
        points = [[0, 0], [1, 0], [1.0000005, 0], [1.01, 0]]
        options = dict(usage="stencilSize", stencil_size=2)
        assert_array_equal(find_neighbors([[0, 0]], points, **options)[0], [0, 1, 2])
        assert_array_equal(find_neighbors([[0, 0]], points, tol=0, **options)[0], [0, 1])

    def test_search_matches_distances(self):
        rng = np.random.default_rng(42)
        points = rng.normal(size=(30, 2))
        query = rng.normal(size=(5, 2))
        neighbors = find_neighbors(query, points, usage="stencilSize", stencil_size=7, tol=0)
        for point, ids in zip(query, neighbors):
            distances = np.linalg.norm(points-point, axis=1)
            assert_array_equal(ids, np.argsort(distances)[:7])
        neighbors = find_neighbors(query, points, poly_order=2, tol=0)
        for point, ids in zip(query, neighbors):
            distances = np.linalg.norm(points-point, axis=1)
            assert_array_equal(ids, np.argsort(distances)[:15])
        options = dict(usage="stencilSize", stencil_size=1)
        for data in (None, []):
            result = find_neighbors(points, data, **options)
            assert_array_equal([ids[0] for ids in result], np.arange(30))
        self.assertEqual(find_neighbors(np.empty((0, 2)), points, **options), [])

    def test_invalid_input(self):
        for points in ([1, 2], [[1, 2, 3]], [[np.nan, 0]], [[np.inf, 0]]):
            with self.assertRaises(ValueError):
                find_neighbors(points)
        for options in (dict(usage="unknown"), dict(stencil_size=0), dict(tol=-1),
                        dict(tol=np.nan), dict(tol=True), dict(stencil_size=3)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                find_neighbors([[0, 0], [1, 0]], usage=options.pop("usage", "stencilSize"), **options)


class TestWeights(unittest.TestCase):
    def setUp(self):
        x, y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
        self.nodes = np.column_stack((x.ravel(), y.ravel())) + [2.3, -1.7]

    def test_monomial_exponents(self):
        assert_array_equal(monomial_exponents(0), [[0, 0]])
        assert_array_equal(monomial_exponents(2), [[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]])
        for value in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                monomial_exponents(value)

    def test_polynomial_reproduction(self):
        for mode in ("polyOrder", "stencilSize"):
            ids_list, weights = build_weights(self.nodes, PHS(3), mode, stencil_size=12)
            for i, (ids, w) in enumerate(zip(ids_list, weights)):
                if mode == "polyOrder":
                    self.assertGreaterEqual(len(ids), 15)
                degree = 2 if mode == "polyOrder" else get_poly_order(len(ids), 2, 2.5)
                exps = monomial_exponents(degree)
                values = get_poly(*self.nodes[ids].T, exps)
                expected = np.column_stack([f(*self.nodes[i], exps) for f in POLYNOMIALS])
                self.assertEqual(w.shape, (len(ids), 6))
                assert_allclose(values.T @ w, expected, atol=1e-10, rtol=1e-10)

    def test_augmented_interpolant_reproduction(self):
        exps = monomial_exponents(2)
        for rbf in (PHS(3), Gauss(beta=2, epsilon=2)):
            ids_list, weights = build_weights(self.nodes, rbf)
            for i, (ids, w) in enumerate(zip(ids_list, weights)):
                points = self.nodes[ids]-self.nodes[i]
                p = get_poly(*points.T, exps)
                # RBF coefficients must satisfy the polynomial side conditions.
                _, _, vh = np.linalg.svd(p.T, full_matrices=True)
                coefficients = vh[len(exps)]
                assert_allclose(p.T @ coefficients, 0, atol=1e-14)
                dx = points[:, None, 0]-points[:, 0]
                dy = points[:, None, 1]-points[:, 1]
                values = rbf.f(dx, dy) @ coefficients
                expected = [getattr(rbf, name)(*-points.T) @ coefficients for name in OPERATORS]
                assert_allclose(values @ w, expected, atol=1e-10, rtol=1e-10)

    def test_failures_are_explicit(self):
        with self.assertRaises(ValueError):
            build_weights(self.nodes, RadialPower(1))
        with self.assertRaises(np.linalg.LinAlgError):
            build_weights(np.zeros((15, 2)), PHS(3))


if __name__ == "__main__":
    unittest.main()
