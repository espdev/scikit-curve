# -*- coding: utf-8 -*-

import pytest

import numpy as np

from curve import Curve
from curve.diffgeom import seg_length, arc_length, natural_parametrization


def test_seg_length():
    n = 1000
    data = np.arange(n)
    curve = Curve([data] * 3, copy=False)

    expected = [1.7320508075688772] * (n - 1)

    assert seg_length(curve) == pytest.approx(expected)


def test_arc_length():
    n = 1000
    data = np.arange(n)
    curve = Curve([data] * 2, copy=False)

    expected = 1.4142135623730951 * (n - 1)

    assert arc_length(curve) == pytest.approx(expected)


def test_natural_parametrization():
    n = 5
    data = np.arange(n)
    curve = Curve([data] * 2, copy=False)

    expected = np.cumsum([0.0] + [1.4142135623730951] * (n - 1))

    assert natural_parametrization(curve) == pytest.approx(expected)
