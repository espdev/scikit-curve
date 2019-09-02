# -*- coding: utf-8 -*-

import pytest

import numpy as np

from curve import Curve, Axis


def test_nonsingular():
    curve = Curve([(np.inf, np.inf, 1, 2, 2.0000000001, 3, np.nan, np.nan, 4, 4.00000000001, 20),
                   (np.inf, 0, 5, 6, 6.0000000001, 7, 10, np.nan, 8, 8.000000000001, np.nan)])
    assert curve.nonsingular() == Curve([(1, 2, 3, 4), (5, 6, 7, 8)])


def test_chordlen():
    n = 1000
    data = np.arange(n)
    curve = Curve([data] * 3, copy=False)

    expected = [1.7320508075688772] * (n - 1)

    assert curve.chordlen == pytest.approx(expected)


def test_arclen():
    n = 1000
    data = np.arange(n)
    curve = Curve([data] * 2, copy=False)

    expected = 1.4142135623730951 * (n - 1)

    assert curve.arclen == pytest.approx(expected)


def test_arclen_2():
    n = 10
    theta = np.linspace(0, 2 * np.pi, n)
    curve = Curve([np.cos(theta), np.sin(theta)])

    assert curve.arclen == pytest.approx(6.156362579862037)


def test_natural_parametrization():
    n = 5
    data = np.arange(n)
    curve = Curve([data] * 2, copy=False)

    expected = np.cumsum([0.0] + [1.4142135623730951] * (n - 1))

    assert curve.t == pytest.approx(expected)


def test_curvature_circle_2d():
    """The curvature of a circle with radius R is equal to 1/R
    """
    t = np.linspace(0.0, 2*np.pi, 100)
    r = 10.0
    x = np.cos(t) * r
    y = np.sin(t) * r

    curve = Curve([x, y])
    expected = np.ones_like(t) * (1 / r)

    assert curve.curvature == pytest.approx(expected, abs=0.0005)


def test_curvature_circle_3d():
    t = np.linspace(0.0, 2*np.pi, 200)
    r = 10.0
    x = np.cos(t) * r
    y = np.sin(t) * r
    z = np.ones_like(t)

    theta = 0.73

    rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)],
    ])

    ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])

    curve = Curve([x, y, z])

    for i, p in enumerate(curve):
        data = rx @ p.data
        data = ry @ data

        curve[i] = data

    expected = np.ones_like(t) * (1 / r)
    assert curve.curvature == pytest.approx(expected, abs=0.0005)


def test_coorientplane_2d():
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    curve_r = curve.reverse()

    assert curve.coorientplane() == curve_r.coorientplane()


@pytest.mark.parametrize('axis1, axis2', [
    (Axis.X, Axis.Y),
    (Axis.X, Axis.Z),
    (Axis.Y, Axis.Z),
])
def test_coorientplane_3d(axis1, axis2):
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)])
    curve_r = curve.reverse()

    assert curve.coorientplane(axis1, axis2) == curve_r.coorientplane(axis1, axis2)


@pytest.mark.parametrize('curve_data, expected_value', [
    ([(1, 2, 3, 4)], 1.0),
    ([(1, 2, 3, 4), (1, 2, 3, 4)], 0.70710678),
    ([(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)], 0.57735027),
    ([(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)], 0.5),
])
def test_tangent_nd(curve_data, expected_value):
    curve = Curve(curve_data)

    expected = np.ones_like(curve.data) * expected_value
    assert curve.tangent == pytest.approx(expected)


def test_tangent_error():
    curve = Curve([(1, 2, 3, 3, 3, 4), (1, 2, 3, 3, 3, 4)])

    with pytest.raises(ValueError):
        curve.tangent  # noqa
