# -*- coding: utf-8 -*-

import typing as t
import numpy as np


NumberType = t.Union[int, float]

PointDataType = t.Union[
    t.Sequence[NumberType],
    np.ndarray,
    'Point',
]

CurveDataType = t.Union[
    t.Sequence[t.Sequence[NumberType]],
    t.Sequence[np.ndarray],
    np.ndarray,
    'Curve',
]
