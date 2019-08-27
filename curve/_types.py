# -*- coding: utf-8 -*-

import typing as t
import numpy as np


PointDataType = t.Union[
    t.Sequence[float],
    np.ndarray,
    'Point',
]

CurveDataType = t.Union[
    t.Sequence[t.Sequence[float]],
    t.Sequence[np.ndarray],
    np.ndarray,
    'Curve',
]
