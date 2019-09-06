# -*- coding: utf-8 -*-

import pytest

from curve import Curve
from curve import InterpolationUniformGrid, interp_methods


@pytest.mark.parametrize('pcount, extrap_left, extrap_right, interp_units, extrap_units, expected', [
    (3, 0, 0, 'points', 'points', [0., 2.82842712, 5.65685425]),
    (5, 0, 0, 'points', 'points', [0., 1.41421356, 2.82842712, 4.24264069, 5.65685425]),
    (3, 2, 2, 'points', 'points', [-5.65685425, -2.82842712, 0., 2.82842712, 5.65685425, 8.48528137, 11.3137085]),
    (5, 2, 2, 'points', 'points', [-2.82842712, -1.41421356, 0., 1.41421356, 2.82842712, 4.24264069, 5.65685425,
                                   7.07106781, 8.48528137]),
    (3, 6, 6, 'points', 'length', [-5.65685425, -2.82842712, 0., 2.82842712, 5.65685425, 8.48528137, 11.3137085]),
    (5, 4, 4, 'points', 'length', [-2.82842712, -1.41421356, 0., 1.41421356, 2.82842712, 4.24264069, 5.65685425,
                                   7.07106781, 8.48528137]),
])
def test_uniform_interp_grid(pcount, extrap_left, extrap_right, interp_units, extrap_units, expected):
    curve = Curve([(1, 3, 5)] * 2)

    grid = InterpolationUniformGrid(
        pcount_len=pcount,
        extrap_left=extrap_left,
        extrap_right=extrap_right,
        interp_units=interp_units,
        extrap_units=extrap_units
    )

    assert grid(curve) == pytest.approx(expected)


@pytest.mark.parametrize('ndmin', [None, 2, 3, 4])
@pytest.mark.parametrize('method', interp_methods())
def test_interp(ndmin, method):
    curve = Curve([(1, 3, 6, 9)] * 2, ndmin=ndmin)
    expected = Curve([(1, 2, 3, 4, 5, 6, 7, 8, 9)] * 2, ndmin=ndmin)

    assert curve.interpolate(9, method=method) == expected


@pytest.mark.parametrize('method', [
    'linear',
    'cubic',
    'hermite',
    'pchip',
    'spline',
])
def test_extrap(method):
    curve = Curve([(1, 3, 5, 7, 9)] * 2)
    grid = InterpolationUniformGrid(
        pcount_len=9,
        extrap_left=3,
        extrap_right=3,
        extrap_units='points'
    )

    expected = [(-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)] * 2

    assert curve.interpolate(grid, method) == Curve(expected)
