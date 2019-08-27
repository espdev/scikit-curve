# -*- coding: utf-8 -*-

import typing as t
import numpy as np


CurveDataType = t.Union[
    t.Tuple[t.Sequence[float], ...],
    t.Tuple[np.ndarray, ...],
    np.ndarray,
]
