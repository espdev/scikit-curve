# -*- coding: utf-8 -*-

import collections.abc as abc
import typing as t
import textwrap

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

IndexerType = t.Union[
    int,
    slice,
    t.Sequence[int],
    np.array,
]

PointCurveUnionType = t.Union[
    'Point',
    'Curve',
]

DEFAULT_DTYPE = np.float64


def _cmp(obj1: PointCurveUnionType, obj2: PointCurveUnionType):
    if np.issubdtype(obj1.dtype, np.integer) and np.issubdtype(obj2.dtype, np.integer):
        return np.equal
    else:
        return np.isclose


class Point(abc.Sequence):
    """A n-dimensional geometric point representation

    The class represents n-dimensional geometric point.

    Notes
    -----
    Point object is immutable. All methods which change point data return the copy.

    Parameters
    ----------
    point_data : PointDataType
        The data of n-dimensional point. The data might be represented in the different types:

        * The sequence of numbers ``Sequence[NumberType]``
        * np.ndarray row 1xM where M is point dimension
        * Another Point object. It creates the copy of another point

    dtype : numeric type or numeric numpy.dtype
        The type of point data. The type must be numeric type. For example, `float`, `int`, `np.float32`, ...

        If dtype is not set, by default dtype has value `np.float64`.

    Examples
    --------

    .. code-block:: python

        # 2-D point
        point = Point([1, 2])

    .. code-block:: python

        # 3-D point with float32 dtype
        point = Point(np.array([1, 2, 3]), dtype=np.float32)

    """

    __slots__ = ('_data', )

    def __init__(self, point_data: PointDataType, dtype: t.Optional[DataType] = None) -> None:
        """Constructs the point
        """

        if isinstance(point_data, Point):
            point_data = point_data.data
            dtype = dtype or point_data.dtype

        if dtype is None:
            dtype = DEFAULT_DTYPE

        if not np.issubdtype(dtype, np.number):
            ValueError('dtype must be a numeric type.')

        data = np.array(point_data, dtype=np.dtype(dtype))

        if data.ndim > 1:
            raise ValueError('Invalid point data: {}\nThe point data must be a vector'.format(point_data))

        self._data = data
        self._data.flags.writeable = False

    def __repr__(self):
        return 'Point({}, ndim={}, dtype={})'.format(
            self._data, self.ndim, self._data.dtype)

    def __len__(self) -> int:
        """Returns the point dimension

        Returns
        -------
        ndim : int
            The point dimension

        """

        return self._data.size

    def __getitem__(self, index: int) -> np.number:
        """Returns coord of the point for given index

        Parameters
        ----------
        index : int
            The index of the coord. Must be an integer.

        Returns
        -------
        coord : dtype
            The coord value for given index

        Raises
        ------
        ValueError : if given index is slice

        """

        if not isinstance(index, int):
            raise ValueError('Index must be an integer')

        return self._data[index]

    def __eq__(self, other: 'Point') -> bool:
        """Returns True if other point is equal to the point

        Parameters
        ----------
        other : Point
            Other point object

        Returns
        -------
        flag : bool
            True if other point is equal to the point

        """

        if not isinstance(other, Point):
            return NotImplemented

        if self.ndim != other.ndim:
            return False

        cmp = _cmp(self, other)
        return np.all(cmp(self._data, other.data))

    def __reversed__(self) -> 'Point':
        """Returns reversed the point with reversed coords

        Returns
        -------
        curve : Curve
            Reversed Point instance

        """

        return Point(np.flip(self._data))

    def __copy__(self) -> 'Point':
        return Point(self)

    def __deepcopy__(self, memodict: t.Optional[dict] = None) -> 'Point':
        return Point(self)

    @property
    def data(self) -> np.array:
        """Returns the point data as numpy array

        Returns
        -------
        data : np.ndarray
            Returns the point data as numpy array 1xM where M is point dimension.

        """

        return self._data

    @property
    def dtype(self):
        """Returns the data type of the point data

        Returns
        -------
        dtype : numpy.dtype
            Numpy data type for data of the point

        """

        return self._data.dtype

    @property
    def ndim(self) -> int:
        """Returns the point dimension

        Returns
        -------
        ndim : int
            The dimension of the point

        """
        return self._data.size


class Curve(abc.Sequence):
    """A n-dimensional geometric curve representation

    The class represents n-dimensional geometric curve.

    Notes
    -----
    Curve object is immutable. All methods which change curve data and size return the copy.

    Parameters
    ----------
    curve_data : CurveDataType
        The data of n-dimensional curve. The data might be represented in the different types:

        * The sequence of the vectors with coordinates for each dimension.
          ``Sequence[Sequence[NumberType]]`` or ``Sequence[numpy.ndarray]``
        * The data is represented as np.ndarray NxM where N is number of points and M is curve dimension
        * Another Curve object. It creates the copy of another curve

    dtype : numeric type or numeric numpy.dtype
        The type of curve data. The type must be numeric type. For example, `float`, `int`, `np.float32`, ...

        If dtype is not set, by default dtype has value `np.float64`.

    Examples
    --------

    .. code-block:: python

        # 2-D curve with 5 points
        curve = Curve([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    .. code-block:: python

        # 2-D curve with 4 points
        curve = Curve(np.array([(1, 2, 3, 4), (5, 6, 7, 8)]).T,
                      dtype=np.float32)

    """

    __slots__ = ('_data', )

    def __init__(self, curve_data: CurveDataType, dtype: t.Optional[DataType] = None) -> None:
        """Constructs Curve instance
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

        data = np.array(curve_data, ndmin=2, dtype=np.dtype(dtype))

        if is_transpose:
            data = data.T

        self._data = data  # type: np.ndarray
        self._data.flags.writeable = False

    def __repr__(self) -> str:
        arr_repr = '{}'.format(self._data)
        arr_repr = textwrap.indent(arr_repr, ' ' * 6).strip()

        return 'Curve({}, size={}, ndim={}, dtype={})'.format(
            arr_repr, self.size, self.ndim, self.dtype)

    def __len__(self) -> int:
        """Returns the number of data points in the curve

        Returns
        -------
        size : int
            The number of points in the curve

        """

        return self._data.shape[0]

    def __getitem__(self, indexer: IndexerType) -> PointCurveUnionType:
        """Returns the point of curve or sub-curve or all coords fot given dimension

        Parameters
        ----------
        indexer : int, slice, list, np.array, tuple
            Index (int) or list of indexes or slice or tuple for getting the point or sub-slice

        Returns
        -------
        point : Point
            Point for given index
        curve : Curve
            Sub-curve for given slice
        coord_values : np.ndarray
            All values for given axis

        """

        is_return_values = isinstance(indexer, tuple) and isinstance(indexer[1], int)
        data = self._data[indexer]

        if data.ndim > 1:
            return Curve(data)
        else:
            if is_return_values:
                return data
            else:
                return Point(data)

    def __contains__(self, other: PointCurveUnionType):
        """Returns True if the curve contains given point or sub-curve

        Parameters
        ----------
        other : Point, Curve
            The instance of point or other curve

        Returns
        -------
        flag : bool
            True if the curve contains given point or sub-curve

        """

        if not isinstance(other, (Point, Curve)):
            return False

        if self.ndim != other.ndim:
            return False

        cmp = _cmp(self, other)

        if isinstance(other, Point):
            return np.any(self._is_equal(other.data, self._data, cmp))
        else:
            self_sz = self.size
            other_sz = other.size

            if self_sz == other_sz:
                return np.all(cmp(self._data, other.data))

            if self_sz < other_sz:
                return False

            for i in range(self_sz - other_sz + 1):
                self_data = self._data[i:(i + other_sz)]
                if np.all(cmp(self_data, other.data)):
                    return True

            return False

    def __eq__(self, other: 'Curve') -> bool:
        """Returns True if given curve is equal to the curve

        Parameters
        ----------
        other : Curve
            Other curve object

        Returns
        -------
        flag : bool
            True if given curve is equal to the curve

        """

        if not isinstance(other, Curve):
            return NotImplemented

        if self.ndim != other.ndim:
            return False
        if self.size != other.size:
            return False

        cmp = _cmp(self, other)
        return np.all(cmp(self._data, other.data))

    def __reversed__(self) -> 'Curve':
        """Returns reversed copy of the curve

        Returns
        -------
        curve : Curve
            Reversed Curve instance

        """

        return Curve(np.flipud(self._data))

    def __add__(self, other: 'Curve') -> 'Curve':
        """Returns concatenation of the curve and other curve

        Parameters
        ----------
        other : Curve
            Other curve object

        Returns
        -------
        curve : Curve
            Concatenation the curve and other curve

        Raises
        ------
        ValueError : If other dimension is not equal to the curve dimension

        """

        if not isinstance(other, Curve):
            return NotImplemented

        self._check_ndim(other)
        return Curve(np.vstack((self._data, other.data)))

    def __copy__(self) -> 'Curve':
        return Curve(self._data)

    def __deepcopy__(self, memodict: t.Optional[dict] = None) -> 'Curve':
        return Curve(self._data)

    def index(self, point: Point, start: t.Optional[int] = None, end: t.Optional[int] = None) -> int:
        """Returns the first index for given point in the curve

        Parameters
        ----------
        point : Point
            The instance of point for check

        start : int or None
            Finding start index or None to begin from 0

        end : int or None
            Finding end index for find or None to end of curve

        Returns
        -------
        index : int
            Index if given point exists in the curve and in the given ``start:end`` interval

        Raises
        ------
        ValueError : If other dimension is not equal to the curve dimension
        ValueError : If given point does not exist in the curve and given ``start:end`` interval

        """

        self._check_ndim(point)

        if start is None and end is None:
            data = self._data
        else:
            data = self._data[slice(start, end)]

        cmp = _cmp(self, point)
        is_close = self._is_equal(point.data, data, cmp)

        if not np.any(is_close):
            raise ValueError('{} is not in curve and given interval'.format(point))

        indices = np.flatnonzero(is_close)

        if start:
            indices += start

        return indices[0]

    def count(self, point: Point) -> int:
        """Returns the number of inclusions given point in the curve

        Parameters
        ----------
        point : Point
            The point instance for check

        Returns
        -------
        count : int
            The number of inclusions given point in the curve

        Raises
        ------
        ValueError : If other dimension is not equal to the curve dimension

        """

        self._check_ndim(point)
        cmp = _cmp(self, point)
        return int(np.sum(self._is_equal(point.data, self._data, cmp)))

    @property
    def data(self) -> np.ndarray:
        """Returns the curve data as numpy array

        Returns
        -------
        data : np.ndarray
            Returns the curve data as numpy array NxM where N is number of data points and M is dimension.

        Examples
        --------

        .. code-block:: python

            >>> import numpy as np
            >>> import numpy.random as random
            >>> curve = Curve([random.rand(4), random.rand(4)], dtype=np.float32)
            >>> curve.data

            array([[0.33747587, 0.6552685 ],
                   [0.8087338 , 0.7574597 ],
                   [0.4331359 , 0.26547626],
                   [0.6998019 , 0.6348531 ]], dtype=float32)

        """

        return self._data

    @property
    def dtype(self):
        """Returns the data type of the curve data

        Returns
        -------
        dtype : numpy.dtype
            Numpy data type for data of the curve

        """
        return self._data.dtype

    @property
    def size(self) -> int:
        """Returns the number of data points in the curve

        Returns
        -------
        size : int
            The number of point in the curve

        """
        return self._data.shape[0]

    @property
    def ndim(self) -> int:
        """Returns the curve dimension

        Returns
        -------
        ndim : int
            The dimension of the curve

        """
        return self._data.shape[1]

    @classmethod
    def from_points(cls, points: t.Sequence[Point], dtype: DataType = None) -> 'Curve':
        """Creates Curve object from sequence of Point objects

        Parameters
        ----------
        points : t.Sequence[Point]
            The sequence (can be iterator) of Point objects
        dtype : numpy.dtype
            The curve data type

        Returns
        -------
        curve : Curve
            The curve object

        Raises
        ------
        ValueError : Invalid input sequence

        Examples
        --------

        .. code-block:: python

            # 3-D curve with 4 points
            curve = Curve.from_points([Point([1, 5, 9]),
                                       Point([2, 6, 10]),
                                       Point([3, 7, 11]),
                                       Point([4, 8, 12])])

        """

        if not all(isinstance(p, Point) for p in points):
            raise ValueError('The sequence must be contain only Points')

        return cls(np.array(list(points)), dtype=dtype)

    def insert(self, index: IndexerType, other: PointCurveUnionType) -> 'Curve':
        """Inserts point or sub-curve to the curve

        Parameters
        ----------
        index : int, slice, list, np.ndarray
            Indexer object to insert data
        other : Point, Curve
            Point or curve object to insert

        Returns
        -------
        curve : Curve
            New curve object with inserted data.

        Raises
        ------
        ValueError : If other dimension is not equal to the curve dimension
        IndexError : If the index is out of bounds

        Examples
        --------

            >>> curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])

            >>> point = Point([10, 20])
            >>> curve.insert(1, point)
            Curve([[ 1.  5.]
                   [10. 20.]
                   [ 2.  6.]
                   [ 3.  7.]
                   [ 4.  8.]], size=5, ndim=2, dtype=float64)

            >>> sub_curve = Curve([(10, 20), (30, 40)])
            >>> curve.insert(-3, sub_curve)
            Curve([[ 1.  5.]
                   [10. 30.]
                   [20. 40.]
                   [ 2.  6.]
                   [ 3.  7.]
                   [ 4.  8.]], size=6, ndim=2, dtype=float64)

            >>> curve.insert([1, 2], sub_curve)
            Curve([[ 1.  5.]
                   [10. 30.]
                   [ 2.  6.]
                   [20. 40.]
                   [ 3.  7.]
                   [ 4.  8.]], size=6, ndim=2, dtype=float64)

        """

        self._check_ndim(other)

        if isinstance(other, Point):
            other_data = np.array(other, ndmin=2)
        elif isinstance(other, Curve):
            other_data = other.data
        else:
            raise TypeError('Inserted object must be Point or Curve instance')

        try:
            return Curve(
                np.insert(self._data, index, other_data, axis=0)
            )
        except IndexError as err:
            raise IndexError(
                'Index {} is out of bounds for curve size {}'.format(
                    index, self.size)) from err

    def append(self, other: PointCurveUnionType):
        """Appends point or curve data to the end of the curve

        Parameters
        ----------
        other : Point, Curve
            Point or curve object to insert

        Returns
        -------
        curve : Curve
            New curve object with appended data.

        Raises
        ------
        ValueError : If other dimension is not equal to the curve dimension

        Examples
        --------

        .. code-block:: python

            >>> curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
            >>> point = Point([10, 20])
            >>> curve.append(point)
            Curve([[ 1.  5.]
                   [ 2.  6.]
                   [ 3.  7.]
                   [ 4.  8.]
                   [10. 20.]], size=5, ndim=2, dtype=float64)
        """

        return self.insert(self.size, other)

    def delete(self, index: IndexerType) -> 'Curve':
        """Returns a new curve object with deleted point or sub-curve

        Parameters
        ----------
        index : int, slice, list, np.array
            An integer index, list of indexes or slice object for deleting points or sub-curve respectively

        Returns
        -------
        curve : Curve
            A new curve object with deleted point or sub-curve

        Raises
        ------
        IndexError : If the index is out of bounds

        Examples
        --------

        .. code-block:: python

            >>> curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
            >>> curve.delete(1)
            Curve([[ 1.  5.]
                   [ 3.  7.]
                   [ 4.  8.]], size=5, ndim=2, dtype=float64)

        .. code-block:: python

            >>> curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
            >>> curve.delete(slice(None, 2))
            Curve([[ 3.  7.]
                   [ 4.  8.]], size=5, ndim=2, dtype=float64)

        """

        try:
            return Curve(
                np.delete(self._data, index, axis=0)
            )
        except IndexError as err:
            raise IndexError(
                'Index {} is out of bounds for curve size {}'.format(
                    index, self.size)) from err

    @staticmethod
    def _is_equal(other_data, data, cmp) -> np.ndarray:
        return np.all(cmp(other_data, data), axis=1)

    def _check_ndim(self, other: PointCurveUnionType):
        if self.ndim != other.ndim:
            raise ValueError('The dimensions of the curve and other object do not match')
