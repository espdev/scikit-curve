# -*- coding: utf-8 -*-

from collections import abc
import typing as t

import numpy as np

from ._types import PointDataType, CurveDataType


DEFAULT_DTYPE = np.float64


class Point(abc.Sequence):
    """A n-dimensional geometric point representation
    """

    __slots__ = ('_data', )

    def __init__(self, point_data: PointDataType, dtype=None) -> None:
        """Constructs the point
        """
        if isinstance(point_data, Point):
            point_data = point_data.data
            dtype = dtype or point_data.dtype

        if dtype is None:
            dtype = DEFAULT_DTYPE

        data = np.array(point_data, dtype=dtype)

        if data.ndim > 1:
            raise ValueError('Invalid point data: {}\nThe point data must be a vector'.format(point_data))

        self._data = data

    def __len__(self) -> int:
        """Returns the point dimension
        """
        return self._data.size

    def __getitem__(self, item: int):
        return self._data[item]

    def __copy__(self):
        return Point(self._data)

    def __eq__(self, other: 'Point') -> bool:
        if not isinstance(other, Point):
            return NotImplemented

        if self.ndim != other.ndim:
            return False
        return np.allclose(self._data, other.data)

    def __repr__(self):
        return 'Point({}, dtype={})'.format(self._data.tolist(), self._data.dtype)

    @property
    def data(self) -> np.array:
        """Returns the point data as numpy array
        """
        return self._data

    @property
    def dtype(self):
        """Returns the data type of the point data
        """
        return self._data.dtype

    @property
    def ndim(self) -> int:
        """Returns the point dimension
        """
        return self._data.size


class Curve(abc.Sequence):
    """A n-dimensional geometric curve representation
    """

    __slots__ = ('_data', )

    def __init__(self, curve_data: CurveDataType, dtype=None) -> None:
        """Constructs the curve
        """

        is_transpose = True

        if isinstance(curve_data, Curve):
            curve_data = curve_data.data
            dtype = dtype or curve_data.dtype

        if isinstance(curve_data, np.ndarray):
            if curve_data.ndim != 2:
                raise ValueError('If the curve data is ndarray it must be two-dimensional.')
            dtype = dtype or curve_data.dtype
            is_transpose = False

        if dtype is None:
            dtype = DEFAULT_DTYPE

        if not np.issubdtype(dtype, np.number):
            ValueError('dtype must be a numeric type.')

        data = np.array(curve_data, ndmin=2, dtype=dtype)

        if is_transpose:
            data = data.T

        self._data = data  # type: np.ndarray

    def __len__(self) -> int:
        """Returns the number of data points in the curve
        """
        return self._data.shape[0]

    def __getitem__(self, item: t.Union[int, slice]) -> t.Union[Point, 'Curve']:
        data = self._data[item]

        if data.ndim > 1:
            return Curve(data)
        else:
            return Point(data)

    def __reversed__(self) -> 'Curve':
        return Curve(np.flipud(self._data))

    def __contains__(self, other: t.Union[Point, 'Curve']):
        if not isinstance(other, (Point, Curve)):
            return NotImplemented

        if self.ndim != other.ndim:
            return False

        if isinstance(other, Point):
            return np.any(self._is_close(other.data, self._data))
        else:
            self_sz = self.size
            other_sz = other.size

            if self_sz == other_sz:
                return np.allclose(self._data, other.data)

            if self_sz < other_sz:
                return False

            for i in range(self_sz - other_sz + 1):
                self_data = self._data[i:(i + other_sz)]
                if np.allclose(self_data, other.data):
                    return True

            return False

    def __copy__(self) -> 'Curve':
        return Curve(self._data)

    def __eq__(self, other: 'Curve') -> bool:
        if not isinstance(other, Curve):
            return NotImplemented

        if self.ndim != other.ndim:
            return False
        if self.size != other.size:
            return False

        return np.allclose(self._data, other.data)

    def __repr__(self):
        return 'Curve({})'.format(self._data)

    def index(self, point: Point, start: int = None, end: int = None) -> int:
        if start is None and end is None:
            data = self._data
        else:
            data = self._data[slice(start, end)]

        is_close = self._is_close(point.data, data)

        if not np.any(is_close):
            raise ValueError('{} is not in curve'.format(point))

        indices = np.flatnonzero(is_close)

        if start:
            indices += start

        return indices[0]

    def count(self, point: Point) -> int:
        return np.sum(self._is_close(point.data, self._data))

    @property
    def data(self) -> np.ndarray:
        """Returns the curve data as numpy array
        """
        return self._data

    @property
    def dtype(self):
        """Returns the data type of the curve data
        """
        return self._data.dtype

    @property
    def size(self) -> int:
        """Returns the number of data points in the curve
        """
        return self._data.shape[0]

    @property
    def ndim(self) -> int:
        """Returns the curve dimension
        """
        return self._data.shape[1]

    @staticmethod
    def _is_close(point_data, data) -> np.ndarray:
        return np.all(np.isclose(point_data, data), axis=1)
