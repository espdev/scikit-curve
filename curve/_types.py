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
    t.Sequence['Point'],
    np.ndarray,
    'Curve',
]

DataType = t.Union[
    t.Type[int],
    t.Type[float],
    np.dtype,
]
