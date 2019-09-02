# -*- coding: utf-8 -*-

import collections.abc as abc
import typing as _t
import textwrap
import enum
import weakref

import numpy as np
from cached_property import cached_property

from .distance import MetricType, get_metric
from . import _diffgeom

NumberType = _t.Union[int, float, np.number]

PointDataType = _t.Union[
    _t.Sequence[NumberType],
    np.ndarray,
    'Point',
]

CurveDataType = _t.Union[
    _t.Sequence[_t.Sequence[NumberType]],
    _t.Sequence[np.ndarray],
    _t.Sequence['Point'],
    np.ndarray,
    'Curve',
]

DataType = _t.Union[
    _t.Type[int],
    _t.Type[float],
    np.dtype,
]

IndexerType = _t.Union[
    int,
    slice,
    _t.Sequence[int],
    np.array,
]

PointCurveUnionType = _t.Union[
    'Point',
    'Curve',
]

DEFAULT_DTYPE = np.float64


def _cmp(obj1: PointCurveUnionType, obj2: PointCurveUnionType):
    if np.issubdtype(obj1.dtype, np.integer) and np.issubdtype(obj2.dtype, np.integer):
        return np.equal
    else:
        return np.isclose


class Axis(enum.IntEnum):
    """The enumeration represents three basic axes: X, Y and Z
    """

    X = 0
    """Abscissa (X) axis"""
    Y = 1
    """Ordinate (Y) axis"""
    Z = 2
    """Applicate (Z) axis"""


class Point(abc.Sequence):
    """A n-dimensional geometric point representation

    The class represents n-dimensional geometric point.

    Notes
    -----
    The data in point object can be mutable but the dimension cannot be changed.

    Parameters
    ----------
    point_data : PointDataType
        The data of n-dimensional point. The data might be represented in the different types:

        * The sequence of numbers ``Sequence[NumberType]``
        * np.ndarray row 1xN where N is point dimension
        * Another Point object. It can create the view or the copy of the data of another point

    dtype : numeric type or numeric numpy.dtype
        The type of point data. The type must be numeric type. For example, `float`, `int`, `np.float32`, ...

        If dtype is not set, by default dtype has value `np.float64`.

    copy : bool
        If this flag is True the copy will be created. If it is False the copy will not be created if possible.
        If it is possible not create a copy, dtype argument will be ignored.

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

    def __init__(self, point_data: PointDataType, dtype: _t.Optional[DataType] = None, copy: bool = True) -> None:
        """Constructs the point
        """

        is_copy = True

        if isinstance(point_data, Point):
            point_data = point_data.data

        if isinstance(point_data, np.ndarray):
            is_copy = copy

        if dtype is None:
            dtype = DEFAULT_DTYPE

        if not np.issubdtype(dtype, np.number):
            ValueError('dtype must be a numeric type.')

        if is_copy:
            data = np.array(point_data, dtype=np.dtype(dtype))
        else:
            data = point_data

        if data.ndim > 1:
            raise ValueError('Invalid point data: {}\nThe point data must be a vector'.format(point_data))

        self._data = data

    def __repr__(self):
        return '{}({}, ndim={}, dtype={})'.format(
            type(self).__name__, self._data, self.ndim, self._data.dtype)

    def __len__(self) -> int:
        """Returns the point dimension

        Returns
        -------
        ndim : int
            The point dimension

        """

        return self._data.size

    def __getitem__(self, index: int) -> _t.Union['Point', np.number]:
        """Returns coord of the point for given index

        Parameters
        ----------
        index : int, slice, list, np.array
            The index of the coord or slice or list of indices

        Returns
        -------
        coord : np.number
            The coord value for given index
        point : Point
            The point with smaller dimension for given slice

        Raises
        ------
        TypeError : Invalid index type
        IndexError : The index is out of dimensions

        """

        data = self._data[index]

        if data.size > 1:
            return Point(data)
        else:
            return data

    def __setitem__(self, indexer: IndexerType, value: _t.Union['Point', np.ndarray]) -> None:
        """Sets point or value for given axis

        Parameters
        ----------
        indexer : int, slice, list, np.array, tuple
            Index (int) or list of indexes or slice or tuple for setting the point or sub-slice
        value : Point, scalar, np.ndarray
            Value for setting

        Raises
        ------
        TypeError : Invalid index type
        IndexError : The index out of bounds point dimensions

        """

        if isinstance(value, Point):
            value = value.data

        self._data[indexer] = value

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

    def __matmul__(self, other: 'Point'):
        """Dot product of two points

        Parameters
        ----------
        other : Point
            Other point

        Returns
        -------
        dot : int, float, np.number
            Dot product of two points

        """
        if not isinstance(other, Point):
            return NotImplemented

        return np.dot(self._data, other.data)

    def __copy__(self) -> 'Point':
        return self.__deepcopy__()

    def __deepcopy__(self, memodict: _t.Optional[dict] = None) -> 'Point':
        return Point(self)

    @property
    def data(self) -> np.array:
        """Returns the point data as numpy array

        Returns
        -------
        data : np.ndarray
            Returns the point data as numpy array 1xN where N is point dimension.

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

    def distance(self, other: 'Point', metric: MetricType = 'euclidean', **kwargs) -> NumberType:
        """Compute distance from this point to other point by given metric

        Parameters
        ----------
        other : point
            Other point
        metric : str, callable
            Distance metric. By default ``euclidean`` from ``scipy.spatial.distance`` will be use
        **kwargs : any
            Additional arguments for given metric

        Returns
        -------
        distance : np.number
            Distance between two points

        Raises
        ------
        NameError : Unknown metric name
        TypeError : Metric is not callable

        """

        if isinstance(metric, str):
            metric = get_metric(metric, **kwargs)

        if not callable(metric):
            raise TypeError('Metric must be str or callable')

        return metric(self._data, other.data)


class CurvePoint(Point):
    """The class represents nd-point that is a nd-curve point

    This class is the view wrapper for a curve point data. This class should not used directly.
    It is used in Curve class.

    The class provides additional data and parameters of curve point. For example ``curvature`` value in the point.

    Parameters
    ----------
    point_data : np.ndarray
        Numpy array view for a curve point data

    curve : Curve
        Curve object

    index : int
        The point index in a curve

    """

    __slots__ = Point.__slots__ + ('_curve', '_idx')

    def __init__(self, point_data: np.ndarray, curve: 'Curve', index):
        super().__init__(point_data, copy=False)

        self._curve = weakref.ref(curve)
        self._idx = index

    def __repr__(self):
        return '{}({}, index={}, valid={})'.format(
            type(self).__name__, self._data, self.idx, self.isvalid)

    def __copy__(self) -> 'CurvePoint':
        return self.__deepcopy__()

    def __deepcopy__(self, memodict: _t.Optional[dict] = None) -> 'CurvePoint':
        if not self.isvalid:
            raise RuntimeError('Cannot create the copy of the invalid point')
        return CurvePoint(self, self.curve, self.idx)

    @property
    def isvalid(self) -> bool:
        """Returns True if the curve instance has not been deleted

        Returns
        -------
        flag : bool
            True if the point is valid and the curve instance has not been deleted

        """

        return self.curve is not None

    @property
    def curve(self) -> _t.Optional['Curve']:
        """Returns ref to the curve object or None if the curve instance has been deleted

        Returns
        -------
        curve : Curve
            Curve object for this point

        """

        return self._curve()

    @property
    def idx(self) -> _t.Optional[int]:
        """Returns the point index in the curve

        Returns
        -------
        index : int
            The point index in the curve or None if the curve instance has been deleted.

        """

        if self.isvalid:
            return self._idx

    @property
    def curvature(self) -> float:
        """Returns the curve curvature value for this point

        Returns
        -------
        k : float
            The curve curvature in this point or NaN if the point not valid.

        """

        if self.isvalid:
            return self.curve.curvature[self.idx]
        return np.nan

    def subcurve(self, other_point: 'CurvePoint', inclusive: bool = True) -> np.ndarray:
        """Returns a sub-curve from the point to other curve point for the same curve

        Parameters
        ----------
        other_point : CurvePoint
            Other point in the same curve
        inclusive : bool
            If this flag is True, other point will be include to a sub-curve.

        Returns
        -------
        curve : Curve
            A sub-curve from the point to other curve point. This sub-curve is a view.

        Raises
        ------
        TypeError : Other point is not an instance of "CurvePoint" class
        RuntimeError : The points are not valid. The curve instance has been deleted
        ValueError : Other point belongs to another curve

        """

        if not isinstance(other_point, CurvePoint):
            raise TypeError('Other point must be an instance of "CurvePoint" class')

        if not self.isvalid:
            raise RuntimeError('The curve instance has been deleted')

        if self.curve is not other_point.curve:
            raise ValueError('Other point belongs to another curve')

        end = 1 if inclusive else 0
        return self.curve[self.idx:other_point.idx+end]


class Curve(abc.Sequence):
    """A n-dimensional geometric curve representation

    The class represents n-dimensional geometric curve.

    Notes
    -----
    Curve objects are mutable with the limitations.
    Some methods that can change curve size or dimension return the copy of curve.
    Also deleting data via ``__delitem__`` is not allowed.

    Parameters
    ----------
    curve_data : CurveDataType
        The data of n-dimensional curve. The data might be represented in the different types:

        * The sequence of the vectors with coordinates for each dimension.
          ``Sequence[Sequence[NumberType]]`` or ``Sequence[numpy.ndarray]``
        * The data is represented as np.ndarray MxN where M is number of points and N is curve dimension
        * Another Curve object. It creates the copy of another curve

    dtype : numeric type or numeric numpy.dtype
        The type of curve data. The type must be numeric type. For example, `float`, `int`, `np.float32`, ...

        If dtype is not set, by default dtype has value `np.float64`.

    copy : bool
        If this flag is True the copy will be created. If it is False the copy will not be created if possible.
        If it is possible not create a copy, dtype argument will be ignored.

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

    def __init__(self, curve_data: CurveDataType, dtype: _t.Optional[DataType] = None, copy: bool = True) -> None:
        """Constructs Curve instance
        """

        is_transpose = True
        is_copy = True

        if isinstance(curve_data, Curve):
            curve_data = curve_data.data
            dtype = dtype or curve_data.dtype

        if isinstance(curve_data, np.ndarray):
            if curve_data.ndim != 2:
                raise ValueError('If the curve data is ndarray it must be two-dimensional.')
            dtype = dtype or curve_data.dtype
            is_transpose = False
            is_copy = copy

        if dtype is None:
            dtype = DEFAULT_DTYPE

        if not np.issubdtype(dtype, np.number):
            ValueError('dtype must be a numeric type.')

        if is_copy:
            data = np.array(curve_data, ndmin=2, dtype=np.dtype(dtype))
        else:
            data = curve_data

        if is_transpose:
            data = data.T

        self._data = data  # type: np.ndarray

    def __repr__(self) -> str:
        name = type(self).__name__

        arr_repr = '{}'.format(self._data)
        arr_repr = textwrap.indent(arr_repr, ' ' * (len(name) + 1)).strip()

        return '{}({}, size={}, ndim={}, dtype={})'.format(
            name, arr_repr, self.size, self.ndim, self.dtype)

    def __len__(self) -> int:
        """Returns the number of data points in the curve

        Returns
        -------
        size : int
            The number of points in the curve

        """

        return self._data.shape[0]

    def __bool__(self) -> bool:
        """Returns True if the curve has size >= 1 and dimension >= 1

        Returns
        -------
        flag : bool
            True if the curve has size >= 1 and dimension >= 1

        """

        return self.size != 0 and self.ndim != 0

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
            return Curve(data, copy=False)
        else:
            if is_return_values:
                return data
            else:
                if indexer < 0:
                    indexer = self.size + indexer
                return CurvePoint(data, self, index=indexer)

    def __setitem__(self, indexer: IndexerType, value: _t.Union[PointCurveUnionType, np.ndarray]) -> None:
        """Sets point or sub-curve or values for given axis

        Parameters
        ----------
        indexer : int, slice, list, np.array, tuple
            Index (int) or list of indexes or slice or tuple for setting the point or sub-slice
        value : Point, Curve, np.ndarray

        Raises
        ------
        TypeError : Invalid index type
        IndexError : The index out of bounds curve size or dimensions

        """

        if isinstance(value, (Point, Curve)):
            value = value.data

        self._data[indexer] = value
        self._invalidate_cache()

    def __delitem__(self, key):
        raise ValueError('Deleting curve data is not allowed')

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
        return self.__deepcopy__()

    def __deepcopy__(self, memodict: _t.Optional[dict] = None) -> 'Curve':
        return Curve(self._data)

    def index(self, point: Point, start: _t.Optional[int] = None, end: _t.Optional[int] = None) -> int:
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
            Returns the curve data as numpy array MxN where M is number of data points and N is dimension.

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

    @property
    def is1d(self) -> bool:
        """Returns True if the curve is 1-d

        The curve is 1-dimensional curve: :math:`y = f(x)`.

        Returns
        -------
        flag : bool
            True if the curve is plane

        """

        return self.ndim == 1

    @property
    def is2d(self) -> bool:
        """Returns True if the curve is plane

        The plane curve is 2-dimensional curve (curve on plane).

        Returns
        -------
        flag : bool
            True if the curve is plane

        """

        return self.ndim == 2

    @property
    def is3d(self) -> bool:
        """Returns True if a curve is 3-dimensional

        The spatial curve is 3-dimensional (curve in tri-dimensional space).

        Returns
        -------
        flag : bool
            True if a curve is 3-space

        """

        return self.ndim == 3

    @property
    def isspatial(self) -> bool:
        """Returns True if the curve is spatial

        The spatial curve is 3 or higher dimensional curve.

        Returns
        -------
        flag : bool
            True if the curve is spatial

        """

        return self.ndim >= 3

    @cached_property
    def t(self) -> np.ndarray:
        """Returns the natural parameter vector for the curve

        Parametrization of a curve by the length of its arc.

        Returns
        -------
        t : np.ndarray
            Natural parameter vector

        """

        return _diffgeom.natural_parametrization(self, chord_lengths=self.chordlen)

    @cached_property
    def chordlen(self) -> np.ndarray:
        """Computes length for each chord (segment) of the curve

        Returns
        -------
        lengths : np.ndarray
            The array with lengths for each the curve chord

        """

        return _diffgeom.chordlen(self)

    @cached_property
    def arclen(self) -> float:
        """Computes the length of the curve arc

        Returns
        -------
        length : float
            The curve arc length

        """

        return _diffgeom.arclen(self)

    @cached_property
    def curvature(self) -> np.ndarray:
        r"""Computes curvature for each point of the n-dimensional curve

        The curvature of a plane curve or a space curve in three dimensions (and higher) is the magnitude of the
        acceleration of a particle moving with unit speed along a curve.

        Curvature formula for 1-dimensional curve :math:`y = f(x)`:

        .. math::

            k = \frac{y''}{(1 + y'^2)^\frac{3}{2}}

        and for 2-dimensional (a plane) curve :math:`\gamma(t) = (x(t), y(t))`:

        .. math::

            k = \frac{y''x' - x''y'}{(x'^2 + y'^2)^\frac{3}{2}}

        and for 3-dimensional curve :math:`\gamma(t) = (x(t), y(t), z(t))`:

        .. math::

            k = \frac{||\gamma' \times \gamma''||}{||\gamma'||^3}

        and for n-dimensional curve:

        .. math::

            k = \frac{\sqrt{||\gamma'||^2||\gamma''||^2 - (\gamma' \cdot \gamma'')^2}}{||\gamma'||^3}

        Notes
        -----
        Curvature values at the ends of the curve can be calculated less accurately.

        Returns
        -------
        k : np.ndarray
            Array of the curvature values for each curve point

        """

        return _diffgeom.curvature(self)

    @classmethod
    def from_points(cls, points: _t.Sequence[Point], dtype: DataType = None) -> 'Curve':
        """Creates Curve object from sequence of Point objects

        Parameters
        ----------
        points : Sequence[Point]
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

    def reverse(self) -> None:
        """Reverses the curve in-place

        Notes
        -----
        This method changes this curve data in-place.

        For getting reversed copy can be used ``reversed`` function::

            curve_rev = reversed(curve)

        """

        self._data[:] = np.flipud(self._data)
        self._invalidate_cache()

    def insert(self, index: IndexerType, other: PointCurveUnionType) -> 'Curve':
        """Inserts point or sub-curve to the curve and returns new curve

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
        """Appends point or curve data to the end of the curve and returns new curve

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
                   [ 4.  8.]], size=3, ndim=2, dtype=float64)

        .. code-block:: python

            >>> curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
            >>> curve.delete(slice(None, 2))
            Curve([[ 3.  7.]
                   [ 4.  8.]], size=2, ndim=2, dtype=float64)

        """

        try:
            return Curve(
                np.delete(self._data, index, axis=0)
            )
        except IndexError as err:
            raise IndexError(
                'Index {} is out of bounds for curve size {}'.format(
                    index, self.size)) from err

    def values(self, axis: _t.Union[int, Axis]) -> np.ndarray:
        """Returns the vector with all values for given axis

        Notes
        -----

        This method is equivalent to use::

            values = curve[:, axis]

        Parameters
        ----------
        axis : int, Axis
            The axis for getting values

        Returns
        -------
        values : np.ndarray
            The vector with all values for given axis

        Examples
        --------

        .. code-block:: python

            >>> curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)])
            >>> curve.values(Axis.X)
            [1. 2. 3. 4.]

            >>> curve.values(-1)
            [ 9. 10. 11. 12.]

        """

        if not isinstance(axis, int):
            raise ValueError('Axis must be an integer')

        if axis >= self.ndim:
            raise ValueError(
                'The axis {} is out of the curve dimensions {}'.format(axis, self.ndim))

        return self._data[:, axis]

    def insert_dim(self, axis: int, values: _t.Union[np.ndarray, _t.Sequence[NumberType], None] = None) -> 'Curve':
        """Insert new dimension to the curve and returns new curve

        Parameters
        ----------
        axis : int
            The axis to insert new dimension
        values : np.ndarray, sequence, None
            If it is not None it will be used as values for inserted dimension.
            If it is not set, will be inserted zeros vector

        Returns
        -------
        curve : Curve
            Curve object with inserted dimension

        Raises
        ------
        IndexError : if index is out of the curve dimensions
        ValueError : if could not broadcast input array to the curve size

        Examples
        --------

        .. code-block:: python

            >>> curve = Curve([(1, 2, 3, 4), (9, 10, 11, 12)])
            >>> curve.insert_dim(1, [5, 6, 7, 8])
            Curve([[ 1.  5.  9.]
                   [ 2.  6. 10.]
                   [ 3.  7. 11.]
                   [ 4.  8. 12.]], size=4, ndim=3, dtype=float64)

        """

        if values is None:
            values = np.zeros(self.size, dtype=self.dtype)

        try:
            return Curve(
                np.insert(self._data, axis, values, axis=1)
            )
        except IndexError as err:
            raise IndexError(
                'Axis {} is out of bounds for curve dimensions {}'.format(axis, self.ndim)) from err

    def append_dim(self, values: _t.Union[np.ndarray, _t.Sequence[NumberType], None] = None) -> 'Curve':
        """Appends new dimension to the end of curve and returns new curve

        Parameters
        ----------
        values : np.ndarray, sequence, None
            If it is not None it will be used as values for inserted dimension.
            If it is not set, will be inserted zeros vector

        Returns
        -------
        curve : Curve
            Curve object with appended dimension

        Raises
        ------
        ValueError : if could not broadcast input array to the curve size

        Examples
        --------

        .. code-block:: python

            >>> curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
            >>> curve.append_dim([9, 10, 11, 12])
            Curve([[ 1.  5.  9.]
                   [ 2.  6. 10.]
                   [ 3.  7. 11.]
                   [ 4.  8. 12.]], size=4, ndim=3, dtype=float64)

        """

        return self.insert_dim(self.ndim, values)

    def delete_dim(self, axis: IndexerType) -> 'Curve':
        """Returns a new curve object with deleted dimension(s)

        Parameters
        ----------
        axis : int, slice, list, np.arrau
            Axis (int) or list of axis or slice for deleting dimension(s)

        Returns
        -------
        curve : Curve
            Curve object with appended dimension

        Raises
        ------

        Examples
        --------

        .. code-block:: python

            >>> curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)])
            >>> curve.delete_dim(-1)
            Curve([[ 1.  5.]
                   [ 2.  6.]
                   [ 3.  7.]
                   [ 4.  8.]], size=4, ndim=2, dtype=float64)

        """

        try:
            return Curve(
                np.delete(self._data, axis, axis=1)
            )
        except IndexError as err:
            raise IndexError(
                'Axis {} is out of bounds for curve dimensions {}'.format(axis, self.ndim)) from err

    def unique(self) -> 'Curve':
        """Returns curve with unique points

        The method deletes duplicate points from the curve and return new curve
        with only unique points.

        Returns
        -------
        curve : Curve
            Curve object with unique points

        """

        # FIXME: unique is slow (O(Nlog(N)). Moreover, we are forced to use
        #  additional sorting indices array to preserve order. This is not good way...
        data, index = np.unique(self._data, axis=0, return_index=True)
        return Curve(data[np.argsort(index)])

    def drop(self, isa: _t.Callable) -> 'Curve':
        """Drops points from the curve by given values checker

        Parameters
        ----------
        isa : callable
            Checker callable with support of vectorization. ``numpy.isnan`` for example.

            The checker callable can return a boolean vector with size equal to the curve size
            or indices vector or boolean MxN array where M is the curve size and N is the curve dimension.

        Returns
        -------
        curve : Curve
            New curve object without dropped points

        Examples
        --------

        .. code-block:: python

            >>> curve = Curve([(1, 2, np.nan, 3, 2, 4), (5, 6, 1, 7, np.inf, 8)])
            >>> curve.drop(lambda x: np.isnan(x) | np.isinf(x))
            Curve([[ 1.  5.]
                   [ 2.  6.]
                   [ 3.  7.]
                   [ 4.  8.]], size=4, ndim=2, dtype=float64)

        Raises
        ------
        TypeError : Invalid ``isa`` checker argument
        ValueError : Invalid ``isa``  checker return type
        IndexError : Cannot indexing curve data with indices from ``isa`` checker

        """

        if not callable(isa):
            raise TypeError('isa argument must be a callable object')

        indices = np.asarray(isa(self._data))

        if indices.ndim > 1:
            if indices.dtype != np.bool:
                raise ValueError('drop indices MxN array must be boolean type.')
            indices = np.any(indices, axis=1)

        if indices.dtype != np.bool:
            return Curve(self.delete(indices))
        else:
            return Curve(self._data[~indices])

    def nonsingular(self):
        """Removes singularities in the curve

        The method removes NaN, Inf and the close points from curve to avoid segments with zero-closed lengths.
        These points/segments of an exceptional set where a curve fails to be well-behaved in some
        particular way, such as differentiability for example.

        Returns
        -------
        curve : Curve
            The curve without singularities.

        """

        return _diffgeom.nonsingular(self, chord_lengths=self.chordlen)

    @staticmethod
    def _is_equal(other_data, data, cmp) -> np.ndarray:
        return np.all(cmp(other_data, data), axis=1)

    def _check_ndim(self, other: PointCurveUnionType):
        if self.ndim != other.ndim:
            raise ValueError('The dimensions of the curve and other object do not match')

    def _invalidate_cache(self):
        cls = type(self)
        cached_props = [attr for attr in self.__dict__
                        if isinstance(getattr(cls, attr, None), cached_property)]

        for prop in cached_props:
            self.__dict__.pop(prop, None)
