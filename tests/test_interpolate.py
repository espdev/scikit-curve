# -*- coding: utf-8 -*-

import pytest

from curve import Curve
from curve import interp_methods


@pytest.mark.parametrize('ndmin', [None, 2, 3, 4])
@pytest.mark.parametrize('method', interp_methods())
def test_interp(ndmin, method):
    curve = Curve([[1, 3, 6], [1, 3, 6]], ndmin=ndmin)
    expected = Curve([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]], ndmin=ndmin)

    assert curve.interpolate(6, method=method) == expected
