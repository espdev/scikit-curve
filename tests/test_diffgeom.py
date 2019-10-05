# -*- coding: utf-8 -*-

import pytest

import numpy as np

from curve import Curve, Axis, DifferentialGeometryWarning
from curve import dot1d


def test_nonsingular():
    curve = Curve([(np.inf, np.inf, 1, 2, 2.0000000001, 3, np.nan, np.nan, 4, 4.00000000001, 20),
                   (np.inf, 0, 5, 6, 6.0000000001, 7, 10, np.nan, 8, 8.000000000001, np.nan)])
    assert curve.nonsingular() == Curve([(1, 2, 3, 4), (5, 6, 7, 8)])


def test_chordlen():
    n = 1000
    data = np.arange(n)
    curve = Curve([data] * 3)

    expected = [1.7320508075688772] * (n - 1)

    assert curve.chordlen == pytest.approx(expected)


def test_arclen():
    n = 1000
    data = np.arange(n)
    curve = Curve([data] * 2)

    expected = 1.4142135623730951 * (n - 1)

    assert curve.arclen == pytest.approx(expected)


def test_arclen_2():
    n = 10
    theta = np.linspace(0, 2 * np.pi, n)
    curve = Curve([np.cos(theta), np.sin(theta)])

    assert curve.arclen == pytest.approx(6.156362579862037)


def test_cumarclen():
    n = 5
    data = np.arange(n)
    curve = Curve([data] * 2)

    expected = np.cumsum([0.0] + [1.4142135623730951] * (n - 1))

    assert curve.cumarclen == pytest.approx(expected)


def test_curvature_circle_2d(circle_curve_2d):
    """The curvature of a circle with radius R is equal to 1/R
    """
    r = 10
    curve = circle_curve_2d(n=100, r=r)
    expected = np.ones(curve.size) * (1 / r)

    assert curve.curvature == pytest.approx(expected, abs=0.0005)


def test_curvature_circle_3d(circle_curve_3d):
    r = 10
    curve = circle_curve_3d(r=10)

    expected = np.ones(curve.size) * (1 / r)
    assert curve.curvature == pytest.approx(expected, abs=0.0005)


def test_torsion_circle_2d(circle_curve_2d):
    curve = circle_curve_2d()
    assert np.allclose(curve.torsion, 0.0)


def test_torsion_circle_3d(circle_curve_3d):
    curve = circle_curve_3d()
    assert np.allclose(curve.torsion, 0.0)


def test_torsion_curve_3d(curve_3d):
    curve = curve_3d()
    assert not np.allclose(curve.torsion, 0.0)


def test_tangent_2d():
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])

    expected = np.ones_like(curve.data)
    assert curve.tangent == pytest.approx(expected)


def test_tangent_3d():
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)])

    expected = np.ones_like(curve.data)
    assert curve.tangent == pytest.approx(expected)


def test_normal_2d():
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])

    expected = np.zeros_like(curve.data)
    assert curve.normal == pytest.approx(expected)


def test_normal_3d():
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)])

    expected = np.zeros_like(curve.data)
    assert curve.normal == pytest.approx(expected)


@pytest.mark.parametrize('curve_data, expected_value', [
    ([(1, 2, 3, 4), (1, 2, 3, 4)], 0.70710678),
    ([(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)], 0.57735027),
    ([(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)], 0.5),
    ([(1, 2), (1, 2)], 0.70710678),
])
def test_frenet1_nd(curve_data, expected_value):
    curve = Curve(curve_data)

    expected = np.ones_like(curve.data) * expected_value
    assert curve.frenet1 == pytest.approx(expected)


def test_frenet1_warn():
    curve = Curve([(1, 2, 3, 3, 3, 4), (1, 2, 3, 3, 3, 4)])

    with pytest.warns(DifferentialGeometryWarning):
        print(curve.frenet1)


@pytest.mark.parametrize('curve_data', [
    [(1, 2, 3, 4), (1, 2, 3, 4)],
    [(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)],
    [(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)],
    [(1, 2), (1, 2)],
])
def test_frenet2_warn(curve_data):
    curve = Curve(curve_data)

    with pytest.warns(DifferentialGeometryWarning):
        print(curve.frenet2)


def test_circle_2d_frenet12_dot_product_zero(circle_curve_2d):
    curve = circle_curve_2d()
    assert np.allclose(dot1d(curve.frenet1, curve.frenet2), 0.0)


def test_circle_3d_frenet12_dot_product_zero(circle_curve_3d):
    curve = circle_curve_3d()
    assert np.allclose(dot1d(curve.frenet1, curve.frenet2), 0.0)


def test_frenet_vectors_orthogonal_3d(curve_3d):
    curve = curve_3d()

    assert np.allclose(dot1d(curve.frenet1, curve.frenet2), 0.0)
    assert np.allclose(dot1d(curve.frenet2, curve.frenet3), 0.0)
    assert np.allclose(dot1d(curve.frenet1, curve.frenet3), 0.0)


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
