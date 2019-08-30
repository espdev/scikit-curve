# -*- coding: utf-8 -*-

import pytest

import numpy as np

from curve import Curve
from curve.diffgeom import (
    seglength,
    arclength,
    remove_singularity,
    natural_parametrization,
)


def test_seg_length():
    n = 1000
    data = np.arange(n)
    curve = Curve([data] * 3, copy=False)

    expected = [1.7320508075688772] * (n - 1)

    assert seglength(curve) == pytest.approx(expected)


def test_arc_length():
    n = 1000
    data = np.arange(n)
    curve = Curve([data] * 2, copy=False)

    expected = 1.4142135623730951 * (n - 1)

    assert arclength(curve) == pytest.approx(expected)


def test_arc_length_2():
    n = 10
    theta = np.linspace(0, 2 * np.pi, n)
    curve = Curve([np.cos(theta), np.sin(theta)])

    assert arclength(curve) == pytest.approx(6.156362579862037)


def test_natural_parametrization():
    n = 5
    data = np.arange(n)
    curve = Curve([data] * 2, copy=False)

    expected = np.cumsum([0.0] + [1.4142135623730951] * (n - 1))

    assert natural_parametrization(curve) == pytest.approx(expected)


def test_remove_singularity():
    curve = Curve([(np.inf, np.inf, 1, 2, 2.0000000001, 3, np.nan, np.nan, 4, 4.00000000001, 20),
                   (np.inf, 0, 5, 6, 6.0000000001, 7, 10, np.nan, 8, 8.000000000001, np.nan)])
    assert remove_singularity(curve) == Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
