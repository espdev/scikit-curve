# -*- coding: utf-8 -*-

import pytest

from curve import Curve
from curve import make_uniform_interp_grid, interp_methods


@pytest.mark.parametrize('pcount, extrap_size, extrap_units, expected', [
    (3, (0, 0), 'points', [0., 2.82842712, 5.65685425]),
    (5, (0, 0), 'points', [0., 1.41421356, 2.82842712, 4.24264069, 5.65685425]),
    (3, (2, 2), 'points', [-5.65685425, -2.82842712, 0., 2.82842712, 5.65685425, 8.48528137, 11.3137085]),
    (5, (2, 2), 'points', [-2.82842712, -1.41421356, 0., 1.41421356, 2.82842712, 4.24264069, 5.65685425, 7.07106781, 8.48528137]),
    (3, (6, 6), 'length', [-5.65685425, -2.82842712, 0., 2.82842712, 5.65685425, 8.48528137, 11.3137085]),
    (5, (4, 4), 'length', [-2.82842712, -1.41421356, 0., 1.41421356, 2.82842712, 4.24264069, 5.65685425, 7.07106781, 8.48528137]),
])
def test_make_uniform_interp_grid(pcount, extrap_size, extrap_units, expected):
    curve = Curve([(1, 3, 5), (1, 3, 5)])

    grid = make_uniform_interp_grid(
        curve, pcount, extrap_size=extrap_size, extrap_units=extrap_units)

    assert grid == pytest.approx(expected)


@pytest.mark.parametrize('ndmin', [None, 2, 3, 4])
@pytest.mark.parametrize('method', interp_methods())
def test_interp(ndmin, method):
    curve = Curve([[1, 3, 6, 9], [1, 3, 6, 9]], ndmin=ndmin)
    expected = Curve([[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]], ndmin=ndmin)

    assert curve.interpolate(9, method=method) == expected
