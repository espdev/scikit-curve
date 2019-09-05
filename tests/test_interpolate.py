# -*- coding: utf-8 -*-

import pytest

from curve import Curve
from curve import interp_methods


@pytest.mark.parametrize('ndmin', [None, 2, 3, 4])
@pytest.mark.parametrize('method', interp_methods())
def test_interp(ndmin, method):
    curve = Curve([[1, 3, 6, 9], [1, 3, 6, 9]], ndmin=ndmin)
    expected = Curve([[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]], ndmin=ndmin)

    assert curve.interpolate(9, method=method) == expected
