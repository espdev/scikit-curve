# -*- coding: utf-8 -*-

import pytest

import numpy as np

from curve import Curve


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


def test_curvature_2d():
    """The curvature of a circle with radius R is equal to 1/R
    """
    t = np.linspace(0.0, 2*np.pi, 100)
    r = 10.0
    x = np.cos(t) * r
    y = np.sin(t) * r

    curve = Curve([x, y])
    expected = np.ones_like(t) * (1 / r)

    assert curve.curvature == pytest.approx(expected, abs=0.0005)
