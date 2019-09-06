# -*- coding: utf-8 -*-

import pytest

from curve import Curve
from curve import UniformInterpolationGrid, UniformExtrapolationGrid, interp_methods


@pytest.mark.parametrize('fill, interp_kind, before, after, extrap_kind, expected', [
    (3, 'point', 0, 0, 'point', [0., 2.82842712, 5.65685425]),
    (5, 'point', 0, 0, 'point', [0., 1.41421356, 2.82842712, 4.24264069, 5.65685425]),
    (3, 'point', 2, 2, 'point', [-5.65685425, -2.82842712, 0., 2.82842712, 5.65685425, 8.48528137, 11.3137085]),
    (5, 'point', 2, 2, 'point', [-2.82842712, -1.41421356, 0., 1.41421356, 2.82842712, 4.24264069, 5.65685425,
                                 7.07106781, 8.48528137]),
    (3, 'point', 6, 6, 'length', [-5.65685425, -2.82842712, 0., 2.82842712, 5.65685425, 8.48528137, 11.3137085]),
    (5, 'point', 4, 4, 'length', [-2.82842712, -1.41421356, 0., 1.41421356, 2.82842712, 4.24264069, 5.65685425,
                                  7.07106781, 8.48528137]),
])
def test_uniform_interp_grid(fill, interp_kind, before, after, extrap_kind, expected):
    interp_grid = UniformInterpolationGrid(
        fill=fill,
        kind=interp_kind,
    )

    extrap_grid = UniformExtrapolationGrid(
        interp_grid,
        before=before,
        after=after,
        kind=extrap_kind,
    )

    curve = Curve([(1, 3, 5)] * 2)

    assert extrap_grid(curve) == pytest.approx(expected)


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
    grid = UniformExtrapolationGrid(
        UniformInterpolationGrid(9, kind='point'),
        before=3, after=3, kind='point')

    expected = [(-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)] * 2

    assert curve.interpolate(grid, method) == Curve(expected)
