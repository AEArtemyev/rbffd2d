"""Checks for falling factorials and arbitrary monomial derivatives."""

import math
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from rbffd2d import (
    falling_factorial, eval_monomial_deriv, monomial_exponents,
    get_poly, get_poly_x, get_poly_y, get_poly_xx, get_poly_yy, get_poly_xy,
)


class TestMonomialDerivatives(unittest.TestCase):
    def test_factorial(self):
        n = np.arange(12).reshape(3, 4)
        for k in range(15):
            expected = [[math.perm(int(v), k) for v in row] for row in n]
            result = falling_factorial(n, k)
            assert_array_equal(result, expected)
            self.assertEqual(result.dtype, np.float64)
        self.assertEqual(falling_factorial(5, 2).shape, ())
        self.assertEqual(falling_factorial(5, 2), 20)
        self.assertEqual(falling_factorial(2, 3), 0)
        self.assertEqual(falling_factorial([], 2).shape, (0,))

    def test_matches_explicit_functions(self):
        exps = monomial_exponents(6)
        x = np.array([[-1.2], [0], [0.7]])
        y = np.array([0, -0.4, 1.3])
        functions = (get_poly, get_poly_x, get_poly_y, get_poly_xx, get_poly_yy, get_poly_xy)
        orders = ((0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1))
        with np.errstate(all="raise"):
            for f, (ix, iy) in zip(functions, orders):
                assert_allclose(eval_monomial_deriv(exps, x, y, ix, iy), f(x, y, exps))

    def test_high_order(self):
        exps = [[5, 4], [3, 2], [2, 6], [5, 4]]
        # Dxxx Dyy (x**5*y**4) = 720*x**2*y**2.
        assert_array_equal(eval_monomial_deriv(exps, 2, -3, 3, 2), [25920, 12, 0, 25920])
        with np.errstate(all="raise"):
            assert_array_equal(eval_monomial_deriv(exps, 0, 0, 3, 2), [0, 12, 0, 0])
            assert_array_equal(eval_monomial_deriv(exps, [0, 1], 0, 8, 8), np.zeros((2, 4)))
        self.assertEqual(eval_monomial_deriv(np.empty((0, 2)), [1, 2], 0, 3, 4).shape, (2, 0))
        self.assertEqual(eval_monomial_deriv(exps, [], [], 3, 4).shape, (0, 4))

    def test_invalid_arguments(self):
        invalid = (-1, 1.5, True, np.nan, np.inf, [2], "2", 2j)
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    falling_factorial(3, value)
                with self.assertRaises(ValueError):
                    eval_monomial_deriv([[2, 2]], 0, 0, value, 0)
                with self.assertRaises(ValueError):
                    eval_monomial_deriv([[2, 2]], 0, 0, 0, value)
        for value in (-1, 1.5, True, np.nan, np.inf, "2", 2j):
            with self.assertRaises(ValueError):
                falling_factorial(value, 2)


if __name__ == "__main__":
    unittest.main()
