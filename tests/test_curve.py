# -*- coding: utf-8 -*-

import pytest
import numpy as np

from curve import Curve


@pytest.mark.parametrize('data, size, dim, dtype', [
    (([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), 5, 2, np.float64),
    (([1, 2], [1, 2], [1, 2]), 2, 3, np.float32),
    (([1], [1], [1], [1]), 1, 4, np.int32),
    (([1, 2]), 2, 1, np.float),
    ((), 0, 1, np.int32),
])
def test_create_curve(data, size, dim, dtype):
    """Tests creating the instance of 'Curve' class
    """
    curve = Curve(data, dtype=dtype)

    assert len(curve) == size
    assert curve.size == size
    assert curve.dim == dim
    assert curve.dtype == dtype
