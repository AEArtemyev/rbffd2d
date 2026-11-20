"""Checks for monomial values and derivatives."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from rbffd2d import (
    get_poly, get_poly_x, get_poly_y, get_poly_xx, get_poly_yy, get_poly_xy,
)


FUNCTIONS = (get_poly, get_poly_x, get_poly_y, get_poly_xx, get_poly_yy, get_poly_xy)
EXPS = np.array([[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0], [3, 2]])


class TestPolynomials(unittest.TestCase):
    def test_known_values(self):
        expected = (
            [1, -3, 2, 9, -6, 4, 72],
            [0, 0, 1, 0, -3, 4, 108],
            [0, 1, 0, -6, 2, 0, -48],
            [0, 0, 0, 0, 0, 2, 108],
            [0, 0, 0, 2, 0, 0, 16],
            [0, 0, 0, 0, 1, 0, -72],
        )
        order = [6, 2, 0, 4, 2, 5, 1, 3]
        for function, values in zip(FUNCTIONS, expected):
            with self.subTest(function=function.__name__):
                result = function(2, -3, EXPS)
                self.assertEqual(result.shape, (7,))
                self.assertEqual(result.dtype, np.float64)
                assert_array_equal(result, values)
                assert_array_equal(
                    function(2, -3, EXPS[order].astype(float)),
                    np.array(values)[order],
                )

    def test_origin(self):
        expected = (
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 0],
            [0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
        )
        with np.errstate(all="raise"):
            for function, values in zip(FUNCTIONS, expected):
                with self.subTest(function=function.__name__):
                    assert_array_equal(function(0, 0, EXPS), values)
                    self.assertTrue(np.all(np.isfinite(function([0, 2], [3, 0], EXPS))))

    def test_broadcasting(self):
        x = np.array([[-2], [0], [3]])
        y = np.array([0, -1, 2, 4])
        x, y = np.broadcast_arrays(x, y)
        zero = np.zeros_like(x)
        one = np.ones_like(x)
        expected = (
            [one, y, x, y*y, x*y, x*x],
            [zero, zero, one, zero, y, 2*x],
            [zero, one, zero, 2*y, x, zero],
            [zero, zero, zero, zero, zero, 2*one],
            [zero, zero, zero, 2*one, zero, zero],
            [zero, zero, zero, zero, one, zero],
        )
        for function, values in zip(FUNCTIONS, expected):
            with self.subTest(function=function.__name__):
                result = function(x[:, :1], y[0], EXPS[:6])
                self.assertEqual(result.shape, (3, 4, 6))
                assert_array_equal(result, np.stack(values, axis=-1))
                assert_array_equal(function(0, y[0], EXPS[:6]), result[1])

    def test_derivatives_by_finite_differences(self):
        exps = [[0, 0], [1, 0], [0, 1], [4, 3], [2, 5], [6, 0], [0, 7]]
        x, y, h = np.array([-0.7, 0.3, 1.2]), np.array([0.6, -0.8, 0.4]), 1e-5
        checks = (
            (get_poly_x, get_poly, h, 0),
            (get_poly_y, get_poly, 0, h),
            (get_poly_xx, get_poly_x, h, 0),
            (get_poly_yy, get_poly_y, 0, h),
            (get_poly_xy, get_poly_x, 0, h),
            (get_poly_xy, get_poly_y, h, 0),
        )
        for derivative, function, dx, dy in checks:
            with self.subTest(derivative=derivative.__name__, function=function.__name__):
                expected = (function(x+dx, y+dy, exps) - function(x-dx, y-dy, exps))/(2*h)
                assert_allclose(derivative(x, y, exps), expected, rtol=1e-7, atol=1e-9)

    def test_empty_arrays(self):
        for function in FUNCTIONS:
            with self.subTest(function=function.__name__):
                self.assertEqual(function(0, 0, np.empty((0, 2))).shape, (0,))
                self.assertEqual(function([0, 1], 0, np.empty((0, 2))).shape, (2, 0))
                self.assertEqual(function([], [], EXPS).shape, (0, 7))

    def test_invalid_exponents(self):
        invalid = (
            [], [1, 2], [[1]], [[1, 2, 3]], [[-1, 0]], [[0.5, 0]],
            [[np.nan, 0]], [[np.inf, 0]], [[True, False]], [[1j, 0]], [["1", "0"]],
        )
        for function in FUNCTIONS:
            for exps in invalid:
                with self.subTest(function=function.__name__, exps=exps):
                    with self.assertRaises(ValueError):
                        function(0, 0, exps)

    def test_incompatible_coordinates(self):
        for function in FUNCTIONS:
            with self.subTest(function=function.__name__), self.assertRaises(ValueError):
                function([1, 2], [1, 2, 3], EXPS)


if __name__ == "__main__":
    unittest.main()
