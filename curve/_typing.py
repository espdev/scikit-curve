# -*- coding: utf-8 -*-

import typing as ty
import numpy as np

if ty.TYPE_CHECKING:
    from curve._base import Point, CurvePoint, Curve


Numeric = ty.Union[int, float, np.number]
NumericSequence = ty.Sequence[Numeric]

PointData = ty.Union[
    NumericSequence,
    np.ndarray,
    'Point',
    'CurvePoint',
]

CurveData = ty.Union[
    ty.Sequence[NumericSequence],
    ty.Sequence[np.ndarray],
    ty.Sequence['Point'],
    np.ndarray,
    'Curve',
]

DType = ty.Optional[
    ty.Union[
        ty.Type[int],
        ty.Type[float],
        np.dtype,
    ]
]

Indexer = ty.Union[
    int,
    slice,
    ty.Sequence[int],
    np.array,
]

PointCurveUnion = ty.Union[
    'Point',
    'CurvePoint',
    'Curve',
]
