"""Checks for stencil size and polynomial degree selection."""

from math import ceil, comb
import unittest

import numpy as np

from rbffd2d import get_poly_order, get_rbf_fd_stencil_size


class TestStencil(unittest.TestCase):
    def test_stencil_size(self):
        self.assertEqual(get_rbf_fd_stencil_size(2, 2), (15, 6))
        self.assertEqual(get_rbf_fd_stencil_size(2, 3), (25, 10))
        self.assertEqual(get_rbf_fd_stencil_size(0, 2), (3, 1))
        self.assertEqual(get_rbf_fd_stencil_size(2, 2, 1.1), (7, 6))
        self.assertEqual(get_rbf_fd_stencil_size(2, 2, mode="min"), (6, 6))
        self.assertEqual(get_rbf_fd_stencil_size(3, 1, mode="min"), (4, 4))
        self.assertEqual(get_rbf_fd_stencil_size(2, 4, mode="min"), (15, 15))
        self.assertEqual(get_rbf_fd_stencil_size(0, 2, 0.5), (1, 1))

    def test_poly_order(self):
        self.assertEqual(get_poly_order(15, 2), 2)
        self.assertEqual(get_poly_order(14, 2), 1)
        self.assertEqual(get_poly_order(25, 3), 2)
        self.assertEqual(get_poly_order(24, 3), 1)
        self.assertEqual(get_poly_order(6, 2, mode="max"), 2)
        self.assertEqual(get_poly_order(5, 2, mode="max"), 1)
        self.assertEqual(get_poly_order(1, 2, mode="max"), 0)
        self.assertEqual(get_poly_order(1, 2, 0.5), 0)
        self.assertEqual(get_poly_order(3, 2), 0)
        self.assertEqual(get_poly_order(16, 2, 2.5), 2)

    def test_degree_thresholds(self):
        for d in (2, 3):
            for m in range(1, 12):
                nm = comb(m+d, d)
                with self.subTest(d=d, m=m):
                    self.assertEqual(get_poly_order(nm, d, mode="max"), m)
                    self.assertEqual(get_poly_order(nm-1, d, mode="max"), m-1)
                    self.assertEqual(get_poly_order(ceil(2.5*nm), d), m)
                    self.assertEqual(get_poly_order(ceil(2.5*nm)-1, d), m-1)
                    n, count = get_rbf_fd_stencil_size(m, d, 2.5)
                    self.assertEqual(count, nm)
                    self.assertEqual(get_poly_order(n, d, 2.5), m)

    def test_numeric_scalar_types(self):
        self.assertEqual(get_rbf_fd_stencil_size(2.0, np.int64(2)), (15, 6))
        self.assertEqual(get_poly_order(np.int64(15), 2.0, np.float64(2.5)), 2)

    def test_insufficient_budget(self):
        with self.assertRaises(ValueError):
            get_poly_order(1, 2)
        with self.assertRaises(ValueError):
            get_poly_order(3, 3, 4)

    def test_invalid_parameters(self):
        invalid_integers = (-1, 1.5, True, np.bool_(True), np.nan, np.inf, [2], "2", 2j)
        for value in invalid_integers:
            for function, args in (
                (get_rbf_fd_stencil_size, (value, 2)),
                (get_rbf_fd_stencil_size, (2, value)),
                (get_poly_order, (value, 2)),
                (get_poly_order, (12, value)),
            ):
                with self.subTest(function=function.__name__, args=args):
                    with self.assertRaises(ValueError):
                        function(*args)
        for function, args in ((get_rbf_fd_stencil_size, (2, 2)), (get_poly_order, (12, 2))):
            for rho in (0, -1, True, np.bool_(True), np.nan, np.inf, [2], "2", 2j):
                with self.subTest(function=function.__name__, rho=rho):
                    with self.assertRaises(ValueError):
                        function(*args, rho=rho)
            for mode in ("invalid", "", None):
                with self.subTest(function=function.__name__, mode=mode):
                    with self.assertRaises(ValueError):
                        function(*args, mode=mode)
        for d in (0, 1, 4):
            with self.assertRaises(ValueError):
                get_poly_order(12, d)
        with self.assertRaises(ValueError):
            get_poly_order(0, 2)
        with self.assertRaises(ValueError):
            get_rbf_fd_stencil_size(2, 0)


if __name__ == "__main__":
    unittest.main()
