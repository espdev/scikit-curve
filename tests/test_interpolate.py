# -*- coding: utf-8 -*-

import pytest
import numpy as np

from curve import Curve
from curve import UniformInterpolationGrid, UniformExtrapolationGrid, interp_methods


@pytest.mark.parametrize('fill, interp_kind, extrap, extrap_kind, expected', [
    (3, 'point', 0, 'point', [0., 2.82842712, 5.65685425]),
    (5, 'point', 0, 'point', [0., 1.41421356, 2.82842712, 4.24264069, 5.65685425]),
    (3, 'point', 2, 'point', [-5.65685425, -2.82842712, 0., 2.82842712, 5.65685425, 8.48528137, 11.3137085]),
    (5, 'point', 2, 'point', [-2.82842712, -1.41421356, 0., 1.41421356, 2.82842712, 4.24264069, 5.65685425,
                              7.07106781, 8.48528137]),
    (3, 'point', 6, 'length', [-5.65685425, -2.82842712, 0., 2.82842712, 5.65685425, 8.48528137, 11.3137085]),
    (5, 'point', 4, 'length', [-4.24264069, -2.82842712, -1.41421356, 0., 1.41421356, 2.82842712, 4.24264069,
                               5.65685425, 7.07106781, 8.48528137, 9.89949494]),
])
@pytest.mark.parametrize('method', ['linear', 'cubic', 'hermite', 'pchip'])
def test_uniform_interp_grid(fill, interp_kind, extrap, extrap_kind, method, expected):
    interp_grid = UniformInterpolationGrid(
        fill=fill,
        kind=interp_kind,
    )

    extrap_grid = UniformExtrapolationGrid(
        interp_grid,
        before=extrap,
        after=extrap,
        kind=extrap_kind,
    )

    curve = Curve([(1, 3, 5)] * 2)

    curve_i = curve.interpolate(interp_grid, method=method)
    curve_e = curve.interpolate(extrap_grid, method=method)

    if extrap_kind == 'length':
        extrap_pcount = round(extrap / curve_i.chordlen.mean()) * 2
    else:
        extrap_pcount = extrap * 2
    extrap_arclen = curve_i.chordlen.mean() * extrap_pcount

    assert extrap_grid(curve) == pytest.approx(expected)
    assert curve_e.size == pytest.approx(curve_i.size + extrap_pcount)
    assert curve_e.arclen == pytest.approx(curve_i.arclen + extrap_arclen)


@pytest.mark.parametrize('ndmin', [None, 2, 3, 4])
@pytest.mark.parametrize('method', interp_methods())
def test_interp(ndmin, method):
    curve = Curve([(1, 3, 6, 9)] * 2, ndmin=ndmin)
    expected = Curve([(1, 2, 3, 4, 5, 6, 7, 8, 9)] * 2, ndmin=ndmin)

    assert np.allclose(expected.chordlen, expected.chordlen[0])
    assert curve.interpolate(9, method=method) == expected


@pytest.mark.parametrize('method', [
    'linear',
    'cubic',
    'hermite',
    'pchip',
    'spline',
])
def test_extrap_kind_point(method):
    curve = Curve([(1, 3, 5, 7, 9)] * 2)
    grid = UniformExtrapolationGrid(
        UniformInterpolationGrid(9, kind='point'),
        before=3, after=3, kind='point')

    expected = [(-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)] * 2

    assert curve.interpolate(grid, method) == Curve(expected)


@pytest.mark.parametrize('method', [
    'linear',
    'cubic',
    'hermite',
    'pchip',
    'spline',
])
@pytest.mark.parametrize('denom', [2, 3, 4, 5, 6])
@pytest.mark.parametrize('exlen', [2, 3, 4, 5, 6])
def test_extrap_kind_length(method, denom, exlen):
    curve = Curve([(1, 3, 5, 7, 9)] * 2)

    chordlen = curve.chordlen.mean() / denom
    extraplen = chordlen * exlen

    interp_grid = UniformInterpolationGrid(fill=chordlen, kind='length')
    extrap_grid = UniformExtrapolationGrid(interp_grid, before=extraplen, after=extraplen, kind='length')

    curve_i = curve.interpolate(interp_grid, method=method)
    curve_e = curve.interpolate(extrap_grid, method=method)

    assert curve_e.arclen == pytest.approx(curve_i.arclen + extraplen * 2)
