# -*- coding: utf-8 -*-

import numpy as np
from ._types import CurveDataType


class Curve:
    """A geometric curve representation
    """

    def __init__(self, curve_data: CurveDataType, dtype=np.float64) -> None:
        """Constructs the curve
        """
        self._data = np.array(curve_data, ndmin=2, dtype=dtype).T  # type: np.ndarray

    @property
    def data(self) -> np.ndarray:
        """Returns the curve data as numpy array
        """
        return self._data

    @property
    def size(self) -> int:
        """Returns the number of data points in the curve
        """
        return self._data.shape[0]

    @property
    def dim(self) -> int:
        """Returns the curve dimension
        """
        return self._data.shape[1]

    @property
    def dtype(self):
        """Returns the data type of the curve data
        """
        return self._data.dtype

    def __len__(self) -> int:
        """Returns the number of data points in the curve
        """
        return self.size
