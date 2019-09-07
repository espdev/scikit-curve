# -*- coding: utf-8 -*-

"""
The module provides the base classes and data types Point, CurvePoint and Curve

"""

import collections.abc as abc
import typing as _t
import textwrap
import enum
import weakref
import warnings

import numpy as np

from decorator import decorator
from cached_property import cached_property

from curve._distance import MetricType, get_metric
from curve._numeric import allequal
from curve import _diffgeom
from curve import _interpolate


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
    'CurvePoint',
    'Curve',
]

InplaceRetType = _t.Optional['Curve']

DEFAULT_DTYPE = np.float64


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
        with np.printoptions(suppress=True, precision=4):
            data_str = '{}'.format(self._data)

        return '{}({}, ndim={}, dtype={})'.format(
            type(self).__name__, data_str, self.ndim, self._data.dtype)

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

    def __delitem__(self, key):
        raise TypeError(
            "'{}' object doesn't support item deletion".format(type(self).__name__))

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

        return bool(allequal(self.data, other.data))

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


@decorator
def _potentially_invalid(func, raise_exc=False, *args, **kwargs):
    obj, *args = args
    if not obj:
        message = 'The curve point is not valid because the curve object has been deleted.'

        if raise_exc:
            raise RuntimeError(message)
        else:
            warnings.warn(message, RuntimeWarning, stacklevel=3)
        return

    return func(obj, *args, **kwargs)


class CurvePoint(Point):
    """The class represents nd-point that is a n-dimensional curve point

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
        with np.printoptions(suppress=True, precision=4):
            data_str = '{}'.format(self._data)

        return '{}({}, index={}, valid={})'.format(
            type(self).__name__, data_str, self.idx, bool(self))

    @_potentially_invalid(raise_exc=True)
    def __setitem__(self, indexer: IndexerType, value: _t.Union['Point', np.ndarray]) -> None:
        self.curve[self.idx, indexer] = value

    def __copy__(self) -> 'CurvePoint':
        return self.__deepcopy__()

    def __deepcopy__(self, memodict: _t.Optional[dict] = None) -> 'CurvePoint':
        if not self:
            raise RuntimeError('Cannot create the copy of the invalid point')
        return CurvePoint(self.data, self.curve, self.idx)

    def __bool__(self) -> bool:
        """Returns True if the curve instance has not been deleted

        Returns
        -------
        flag : bool
            True if the point is valid (the curve instance has not been deleted)

        """
        return self.curve is not None

    @property
    def curve(self) -> _t.Optional['Curve']:
        """Returns ref to the curve object or None if the curve instance has been deleted

        Returns
        -------
        curve : Curve
            Curve object for this point

        See Also
        --------
        Curve

        """

        return self._curve()

    @property
    @_potentially_invalid
    def idx(self) -> _t.Optional[int]:
        """Returns the point index in the curve

        Returns
        -------
        index : int
            The point index in the curve or None if the curve instance has been deleted.

        """

        return self._idx

    @property
    @_potentially_invalid
    def firstderiv(self) -> _t.Optional[np.ndarray]:
        """Returns first-order derivative at this curve point

        Returns
        -------
        fder : np.ndarray
             The 1xN array of first-order derivative at this curve point

        See Also
        --------
        Curve.firstderiv

        """

        return self.curve.firstderiv[self.idx]

    @property
    @_potentially_invalid
    def secondderiv(self) -> _t.Optional[np.ndarray]:
        """Returns second-order derivative at this curve point

        Returns
        -------
        sder : np.ndarray
             The 1xN array of second-order derivative at this curve point

        See Also
        --------
        Curve.secondderiv

        """

        return self.curve.secondderiv[self.idx]

    @property
    @_potentially_invalid
    def thirdderiv(self) -> _t.Optional[np.ndarray]:
        """Returns third-order derivative at this curve point

        Returns
        -------
        tder : np.ndarray
             The 1xN array of third-order derivative at this curve point

        See Also
        --------
        Curve.thirdderiv

        """

        return self.curve.thirdderiv[self.idx]

    @property
    @_potentially_invalid
    def tangent(self) -> _t.Optional[np.ndarray]:
        """Returns tangent vector for the curve point

        Notes
        -----
        This is alias for :func:`CurvePoint.firstderiv` property.

        Returns
        -------
        tangent : np.ndarray
            The 1xN array of tangent vector for the curve point

        See Also
        --------
        firstderiv
        Curve.tangent
        frenet1

        """

        return self.curve.tangent[self.idx]

    @property
    @_potentially_invalid
    def normal(self) -> _t.Optional[np.ndarray]:
        """Returns normal vector at the curve point

        Returns
        -------
        normal : np.ndarray
            The 1xN array of normal vector at the curve point

        See Also
        --------
        tangent
        Curve.normal
        frenet2

        """

        return self.curve.normal[self.idx]

    @property
    @_potentially_invalid
    def binormal(self) -> _t.Optional[np.ndarray]:
        """Returns binormal vector at the curve point

        Returns
        -------
        binormal : np.ndarray
            The 1xN array of binormal vector at the curve point

        See Also
        --------
        Curve.binormal
        normal
        frenet3

        """

        return self.curve.binormal[self.idx]

    @property
    @_potentially_invalid
    def speed(self) -> _t.Optional[float]:
        """Returns the speed in the point

        Returns
        -------
        speed : float
            The speed value in the point

        See Also
        --------
        Curve.speed

        """

        return self.curve.speed[self.idx]

    @property
    @_potentially_invalid
    def frenet1(self) -> _t.Optional[np.ndarray]:
        """Returns the first Frenet vector (unit tangent vector) at the point

        Returns
        -------
        e1 : np.ndarray
            The first Frenet vector (unit tangent)

        See Also
        --------
        Curve.frenet1

        """

        return self.curve.frenet1[self.idx]

    @property
    @_potentially_invalid
    def frenet2(self) -> _t.Optional[np.ndarray]:
        """Returns the second Frenet vector (unit normal vector) at the point

        Returns
        -------
        e2 : np.ndarray
            The second Frenet vector (unit normal vector)

        See Also
        --------
        Curve.frenet2

        """

        return self.curve.frenet2[self.idx]

    @property
    @_potentially_invalid
    def frenet3(self) -> _t.Optional[np.ndarray]:
        """Returns the third Frenet vector (unit binormal vector) at the point

        Returns
        -------
        e3 : np.ndarray
            The third Frenet vector (unit binormal vector)

        See Also
        --------
        Curve.frenet3

        """

        return self.curve.frenet3[self.idx]

    @property
    @_potentially_invalid
    def curvature(self) -> _t.Optional[float]:
        """Returns the curvature value at this point of the curve

        Returns
        -------
        k : float
            The curvature value at this point

        See Also
        --------
        Curve.curvature

        """

        return self.curve.curvature[self.idx]

    @property
    @_potentially_invalid
    def torsion(self) -> _t.Optional[float]:
        """Returns the torsion value at this point of the curve

        Returns
        -------
        tau : float
            The torsion value at this point

        See Also
        --------
        Curve.torsion

        """

        return self.curve.torsion[self.idx]

    @_potentially_invalid(raise_exc=True)
    def subcurve(self, other_point: 'CurvePoint', inclusive: bool = True) -> np.ndarray:
        """Returns a sub-curve view from the point to other point in the same curve

        Parameters
        ----------
        other_point : CurvePoint
            Other point in the same curve
        inclusive : bool
            If this flag is True, other point will be included to a sub-curve.

        Returns
        -------
        curve : Curve
            A sub-curve from the point to other curve point. This sub-curve is the view.

        Raises
        ------
        RuntimeError : The points are not valid. The curve instance has been deleted
        TypeError : Other point is not an instance of "CurvePoint" class
        ValueError : Other point belongs to another curve

        """

        if not isinstance(other_point, CurvePoint):
            raise TypeError('Other point must be an instance of "CurvePoint" class')

        if self.curve is not other_point.curve:
            raise ValueError('Other point belongs to another curve')

        inc = 1 if inclusive else 0
        return self.curve[self.idx:other_point.idx+inc]


class Curve(abc.Sequence):
    """The main class for n-dimensional geometric curve representation

    The class represents n-dimensional geometric curve in the plane or in the Euclidean n-dimensional space
    given by a finity sequence of points.

    Internal data storage of curve points is NumPy MxN array where M is the number of curve points and N is curve
    dimension. In other words, n-dimensional curve data is stored in 2-d array::

        # 2-d curve representation
        Curve([[x1 y1]
               [x2 y2]
               [x3 y3]
               ...
               [xM yM]])

        # 3-d curve representation
        Curve([[x1 y1 z1]
               [x2 y2 z2]
               [x3 y3 z3]
               ...
               [xM yM zM]])

        # N-d curve representation
        Curve([[x1 y1 z1 ... N1]
               [x2 y2 z2 ... N2]
               [x3 y3 z3 ... N3]
               ...
               [xM yM zM ... NM]])

    Notes
    -----
    Curve class implements ``Sequence`` interface but its instances are mutable with the limitations.

    Some methods can change the curve data in-place withous change the data size.
    However, some methods that can change curve size or dimension always return the copy of the curve.
    Also deleting data via ``__delitem__`` is not allowed.

    Parameters
    ----------
    curve_data : CurveDataType
        The data of n-dimensional curve (2 or higher). The data might be represented in the different types:

        * The sequence of the vectors with coordinates for every dimension:
          ``[X, Y, Z, ..., N]`` where X, Y, ... are 1xM arrays.
        * The data is represented as np.ndarray MxN where M is number of points and N is curve dimension.
          N must be at least 2 (a plane curve).
        * Another Curve object. It creates the copy of another curve by default (see ``copy`` argument).

        If the data is not set empty curve will be created with ndmin dimensions
        (2 by default, see ``ndmin`` argument).

    ndmin : int
        The minimum curve dimension. By default it is ``None`` and equal to input data dimension.
        If ``ndmin`` is more than input data dimension, additional dimensions will be added to
        created curve object. All values in additional dimensions are equal to zero.
        If it is set, ``copy`` argument is ignored.

    dtype : numeric type or numeric numpy.dtype
        The type of curve data. The type must be a numeric type. For example, ``float``, ``int``, ``np.float32``, ...

        If ``dtype`` argument is not set, by default dtype of curve data is ``np.float64``.
        If ``dtype`` argument is set, ``copy`` argument is ignored.

    copy : bool
        If this flag is True the copy of the data or curve will be created. If it is False the copy will not be
        created if possible.

    Raises
    ------
    ValueError : If the input data is invalid (1-d array or ndim > 2, for example)
    ValueError : There is not a numeric ``dtype``

    Examples
    --------

    .. code-block:: python

        # 2-D curve with 5 points from list of lists
        curve = Curve([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    .. code-block:: python

        # 2-D curve with 4 points from numpy array
        curve = Curve(np.array([(1, 2, 3, 4), (5, 6, 7, 8)]).T,
                      dtype=np.float32)

    .. code-block:: python

        # 3-D curve with 10 random points
        curve = Curve(np.random.rand(10, 3))

    .. code-block:: python

        # 3-D curve from 2-D data with 5 random points
        curve = Curve(np.random.rand(5, 2), ndmin=3)

    """

    def __init__(self, curve_data: _t.Optional[CurveDataType] = None,
                 ndmin: _t.Optional[int] = None,
                 dtype: _t.Optional[DataType] = None,
                 copy: bool = True) -> None:
        """Constructs Curve instance
        """

        is_transpose = True
        is_copy = True
        is_ndmin = False

        if ndmin is None:
            ndmin = 2
        else:
            is_ndmin = True

        if ndmin < 2:
            raise ValueError('ndmin must be at least of 2')

        if is_ndmin or dtype is not None:
            copy = True

        if isinstance(curve_data, Curve):
            curve_data = curve_data.data

        if isinstance(curve_data, np.ndarray):
            if curve_data.size > 0 and curve_data.ndim != 2:
                raise ValueError('If the curve data is ndarray it must be two-dimensional.')
            dtype = dtype or curve_data.dtype
            is_transpose = False
            is_copy = copy

        if dtype is None:
            dtype = DEFAULT_DTYPE
        dtype = np.dtype(dtype)

        if not np.issubdtype(dtype, np.number):
            ValueError('dtype must be a numeric type.')

        empty_data = np.reshape([], (0, ndmin)).astype(dtype)

        if curve_data is None:
            curve_data = empty_data
            is_transpose = False

        if is_copy:
            data = np.array(curve_data, ndmin=2, dtype=dtype)
        else:
            data = curve_data

        if is_transpose:
            data = data.T

        if data.size == 0:
            data = empty_data

        m, n = data.shape

        if data.size > 0 and n < 2:
            raise ValueError('The input data must be at least 2-dimensinal (a curve in the plane).')

        if is_ndmin and m > 0 and n < ndmin:
            # Change dimension to ndmin
            data = np.hstack([data, np.zeros((m, ndmin - n), dtype=dtype)])

        self._data = data  # type: np.ndarray

    def __repr__(self) -> str:
        name = type(self).__name__

        with np.printoptions(suppress=True, precision=4, edgeitems=4, threshold=10*self.ndim):
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

    def __getitem__(self, indexer: IndexerType) -> _t.Union[PointCurveUnionType, np.ndarray]:
        """Returns the point of curve or sub-curve or all coord values fot given axis

        Parameters
        ----------
        indexer : int, slice, list, np.array, tuple
            Index (int) or list of indexes or slice or tuple for getting the point or sub-slice

        Raises
        ------
        TypeError : Invalid index type
        IndexError : The index out of bounds curve size or dimensions

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
            if data.shape[1] == 1:
                return data.ravel()
            return Curve(data, copy=False)
        else:
            if is_return_values:
                return data
            else:
                if indexer < 0:
                    indexer = self.size + indexer
                return CurvePoint(data, self, index=indexer)

    def __setitem__(self, indexer: IndexerType, value: _t.Union[PointCurveUnionType, np.ndarray]) -> None:
        """Sets a point or a sub-curve or coord values for given axis

        Parameters
        ----------
        indexer : int, slice, list, np.array, tuple
            Index (int) or list of indexes or slice or tuple for setting the point or sub-slice
        value : Point, Curve, np.ndarray
            Value for setting

        Raises
        ------
        TypeError : Invalid index type
        IndexError : The index out of bounds curve size or dimensions

        """

        if isinstance(value, (Point, Curve)):
            value = value.data

        self._data[indexer] = value
        self.invalidate()

    def __delitem__(self, key):
        raise TypeError(
            "'{}' object doesn't support item deletion".format(type(self).__name__))

    def __contains__(self, other: PointCurveUnionType):
        """Returns True if the curve contains given point or sub-curve with the same dimension

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

        if isinstance(other, Point):
            return np.any(allequal(other.data, self._data, axis=1))
        else:
            self_sz = self.size
            other_sz = other.size

            if self_sz == other_sz:
                return allequal(self._data, other.data)

            if self_sz < other_sz:
                return False

            for i in range(self_sz - other_sz + 1):
                self_data = self._data[i:(i + other_sz)]
                if allequal(self_data, other.data):
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

        return bool(allequal(self.data, other.data))

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

        is_close = allequal(point.data, data, axis=1)

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
        return int(np.sum(allequal(point.data, self._data, axis=1)))

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
    def is2d(self) -> bool:
        """Returns True if the curve is 2-dimensional

        The plane curve (2-dimensional) in XY plane

        Returns
        -------
        flag : bool
            True if the curve is 2-dimensional

        """

        return self.ndim == 2

    @property
    def is3d(self) -> bool:
        """Returns True if the curve is 3-dimensional

        The spatial 3-dimensional curve (curve in tri-dimensional Euclidean space).

        Returns
        -------
        flag : bool
            True if the curve is 3-space

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

        Parametrization of the curve by the length of its arc.

        Returns
        -------
        t : np.ndarray
            The 1xM array natural parameter

        See Also
        --------
        chordlen
        arclen
        tn

        """

        return _diffgeom.natural_parametrization(self, chord_lengths=self.chordlen)

    @cached_property
    def tn(self) -> np.ndarray:
        """Returns the normalized natural parameter vector for the curve

        Parametrization of the curve by the length of its arc normalized to 1.0.

        Returns
        -------
        tn : np.ndarray
            The 1xM array normalized natural parameter

        See Also
        --------
        chordlen
        arclen
        t

        """

        return self.t / self.t[-1]

    @cached_property
    def chordlen(self) -> np.ndarray:
        """Returns length of every chord (segment) of the curve

        Returns
        -------
        lengths : np.ndarray
            The 1x(M-1) array with length of every the curve chord

        """

        return _diffgeom.chordlen(self)

    @cached_property
    def arclen(self) -> float:
        """Returns the length of the curve arc

        Returns
        -------
        length : float
            The curve arc length

        See Also
        --------
        chordlen

        """

        return _diffgeom.arclen(self)

    @cached_property
    def firstderiv(self) -> np.ndarray:
        """Returns the first-order derivative at every curve point

        Returns
        -------
        fder : np.ndarray
            The MxN array with first-order derivative at every point of curve

        See Also
        --------
        tangent
        secondderiv
        thirdderiv

        """

        return _diffgeom.gradient(self._data)

    @cached_property
    def secondderiv(self) -> np.ndarray:
        """Returns the second-order derivative at every curve point

        Returns
        -------
        sder : np.ndarray
            The MxN array with second-order derivative at every point of curve

        See Also
        --------
        firstderiv
        thirdderiv

        """

        return _diffgeom.gradient(self.firstderiv)

    @cached_property
    def thirdderiv(self) -> np.ndarray:
        """Returns the third-order derivative at every curve point

        Returns
        -------
        tder : np.ndarray
            The MxN array with third-order derivative at every point of curve

        See Also
        --------
        firstderiv
        secondderiv

        """

        return _diffgeom.gradient(self.secondderiv)

    @cached_property
    def tangent(self) -> np.ndarray:
        """Returns tangent vector at every curve point

        Notes
        -----
        This is alias for :func:`Curve.firstderiv` property.

        Returns
        -------
        tangent : np.ndarray
            The MxN array of tangent vector at every curve point

        See Also
        --------
        firstderiv
        speed

        """

        return self.firstderiv

    @cached_property
    def normal(self) -> np.ndarray:
        r"""Returns the normal vector at every point of curve

        .. math::
            \overline{e_2}(t) = \gamma''(t) - \langle \gamma''(t), e_1(t) \rangle \, e_1(t)

        Notes
        -----
        The normal vector, sometimes called the curvature vector,
        indicates the deviance of the curve from being a straight line.

        Returns
        -------
        normal : np.ndarray
            The MxN array with normal vector at every point of curve

        See Also
        --------
        tangent
        frenet2
        curvature

        """

        return _diffgeom.normal(self)

    @cached_property
    def binormal(self) -> np.ndarray:
        r"""Returns the binormal vector at every point of the curve

        .. math::
            \overline{e_3}(t) = \gamma'''(t) - \langle \gamma'''(t), e_1(t) \rangle \, e_1(t)
            - \langle \gamma'''(t), e_2(t) \rangle \, e_2(t)

        Notes
        -----
        The binormal vector is always orthogonal to the tangent and normal vectors at every point of the curve.

        Returns
        -------
        binormal : np.ndarray
            The MxN array with binormal vector at every point of curve

        See Also
        --------
        tangent
        normal
        frenet3
        torsion

        """

        return _diffgeom.binormal(self)

    @cached_property
    def speed(self) -> np.ndarray:
        """Returns Mx1 array of the speed at the time (at every curve point) as tangent vector's magnitude

        Notes
        -----
        The speed is the tangent (velocity) vector's magnitude (norm).
        In general, speed may be zero at some point if the curve has zero-length segments.

        Returns
        -------
        speed : np.ndarray
            The Mx1 array with speed at every curve point

        See Also
        --------
        tangent

        """

        return _diffgeom.speed(self)

    @cached_property
    def frenet1(self) -> np.ndarray:
        r"""Returns the first Frenet vector (tangent unit vector) at every point of the curve

        .. math::

            e_1(t) = \frac{\gamma'(t)}{||\gamma'(t)||}

        Returns
        -------
        e1 : np.ndarray
            The MxN array of tangent unit vector at every curve point

        Raises
        ------
        ValueError : Cannot compute unit vector if speed is equal to zero (division by zero)

        See Also
        --------
        tangent
        speed
        frenet2
        frenet3

        """

        return _diffgeom.frenet1(self)

    @cached_property
    def frenet2(self) -> np.ndarray:
        r"""Returns the second Frenet vector (normal unit vector) at every point of the curve

        .. math::

            e_2(t) = \frac{e_1'(t)}{||e_1'(t)||}

        Returns
        -------
        e2 : np.ndarray
            The MxN array of normal unit vector at every curve point

        Raises
        ------
        ValueError : Cannot compute unit vector if speed is equal to zero (division by zero)

        See Also
        --------
        normal
        frenet1
        frenet3

        """

        return _diffgeom.frenet2(self)

    @cached_property
    def frenet3(self) -> np.ndarray:
        r"""Returns the third Frenet vector (binormal unit vector) at every point of the curve

        .. math::

            e_3(t) = \frac{\overline{e_3}(t)}{||\overline{e_3}(t)||}

        Returns
        -------
        e2 : np.ndarray
            The MxN array of normal unit vector at every curve point

        Raises
        ------
        ValueError : Cannot compute unit vector if speed is equal to zero (division by zero)

        See Also
        --------
        binormal
        frenet1
        frenet2

        """

        return _diffgeom.frenet3(self)

    @cached_property
    def curvature(self) -> np.ndarray:
        r"""Returns curvature at every point of the curve

        The curvature of a plane curve or a space curve in three dimensions (and higher) is the magnitude of the
        acceleration of a particle moving with unit speed along a curve.

        Curvature formula for 2-dimensional (a plane) curve :math:`\gamma(t) = (x(t), y(t))`:

        .. math::

            k = \frac{y''x' - x''y'}{(x'^2 + y'^2)^\frac{3}{2}}

        and for 3-dimensional curve :math:`\gamma(t) = (x(t), y(t), z(t))`:

        .. math::

            k = \frac{||\gamma' \times \gamma''||}{||\gamma'||^3}

        and for n-dimensional curve in general:

        .. math::

            k = \frac{\sqrt{||\gamma'||^2||\gamma''||^2 - (\gamma' \cdot \gamma'')^2}}{||\gamma'||^3}

        Notes
        -----
        Curvature values at the ends of the curve can be calculated less accurately.

        Returns
        -------
        k : np.ndarray
            The 1xM array of the curvature value at every curve point

        See Also
        --------
        normal
        torsion

        """

        return _diffgeom.curvature(self)

    @cached_property
    def torsion(self) -> np.ndarray:
        r"""Returns torsion at every point of the curve

        The second generalized curvature is called torsion and measures the deviance of the curve
        from being a plane curve. In other words, if the torsion is zero, the curve lies completely
        in the same osculating plane (there is only one osculating plane for every point t).

        It is defined as:

        .. math::

            \tau(t) = \chi_2(t) = \frac{\langle e_2'(t), e_3(t) \rangle}{\| \gamma'(t) \|}

        Returns
        -------
        tau : np.ndarray
            The 1xM array of the torsion value at every curve point

        See Also
        --------
        curvature

        """

        return _diffgeom.torsion(self)

    def reverse(self, inplace: bool = False) -> InplaceRetType:
        """Reverses the curve

        Parameters
        ----------
        inplace : bool
            If it is True, the method reverses the curve in-place and changed this object.

        Returns
        -------
        curve : Curve
            The reversed copy of the curve or None if ``inplace`` is True

        """

        if inplace:
            self._data[:] = np.flipud(self._data)
            self.invalidate()
        else:
            return Curve(np.flipud(self._data))

    def coorientplane(self, axis1: int = 0, axis2: int = 1, inplace: bool = False) -> InplaceRetType:
        """Co-orients the curve to given plane orientation

        Notes
        -----
        By default the curve orients to XY plane orientation.

        Parameters
        ----------
        axis1: int
            First plane axis
        axis2: int
            Second plane axis
        inplace : bool
            If it is True, the method changes this object.

        Returns
        -------
        curve : Curve
            The reversed curve copy or this curve object or None if ``inplace`` is True

        Raises
        ------
        ValueError : Curve has the dimension less than 2
        IndexError : Axis out of the curve dimensions

        """

        is_coorient = _diffgeom.coorientplane(self, axis1=axis1, axis2=axis2)

        if not is_coorient:
            return self.reverse(inplace=inplace)
        else:
            if not inplace:
                return self

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

    def values(self, axis: _t.Union[int, Axis, None] = None) -> _t.Union[np.ndarray, abc.Iterator]:
        """Returns the vector with all values for given axis or the iterator along all axes

        Notes
        -----

        This method is equivalent to use::

            values = curve[:, axis]

        Parameters
        ----------
        axis : int, Axis, None
            The axis for getting values. If it is not set iterator along all axes will be returned

        Returns
        -------
        values : np.ndarray, iterable
            The vector with all values for given axis or iterator along all axes

        Examples
        --------

        .. code-block:: python

            >>> curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)])
            >>> curve.values(Axis.X)
            [1. 2. 3. 4.]

            >>> curve.values(-1)
            [ 9. 10. 11. 12.]

        """

        if axis is not None and not isinstance(axis, int):
            raise ValueError('Axis must be an integer')

        if axis is not None and axis >= self.ndim:
            raise ValueError(
                'The axis {} is out of the curve dimensions {}'.format(axis, self.ndim))

        if axis is not None:
            return self._data[:, axis]
        else:
            return iter(self._data[:, i] for i in range(self.ndim))

    def insertdim(self, axis: int, values: _t.Union[np.ndarray, _t.Sequence[NumberType], None] = None) -> 'Curve':
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
            >>> curve.insertdim(1, [5, 6, 7, 8])
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

    def appenddim(self, values: _t.Union[np.ndarray, _t.Sequence[NumberType], None] = None) -> 'Curve':
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
            >>> curve.appenddim([9, 10, 11, 12])
            Curve([[ 1.  5.  9.]
                   [ 2.  6. 10.]
                   [ 3.  7. 11.]
                   [ 4.  8. 12.]], size=4, ndim=3, dtype=float64)

        """

        return self.insertdim(self.ndim, values)

    def deletedim(self, axis: IndexerType) -> 'Curve':
        """Returns a new curve object with deleted dimension(s)

        Notes
        -----
        If the curve is 2-dimensional this operation is not allowed and raises ``ValueError``.

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
        ValueError : if the curve is 2-dimensional
        IndexError : indexation error

        Examples
        --------

        .. code-block:: python

            >>> curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)])
            >>> curve.deletedim(-1)
            Curve([[ 1.  5.]
                   [ 2.  6.]
                   [ 3.  7.]
                   [ 4.  8.]], size=4, ndim=2, dtype=float64)

        """

        if self.is2d:
            raise ValueError('Cannot delete dimensions from 2-dimensional curve.')

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

    def interpolate(self, pcount_or_grid: _interpolate.InterpGridSpecType, method: str = 'linear', **kwargs) -> 'Curve':
        """Interpolates the curve data

        The method interpolates the curve data by given grid or
        given number of points on uniformly interpolated curve.

        Parameters
        ----------
        pcount_or_grid : np.ndarray, int, UniformInterpolationGrid, UniformExtrapolationGrid
            Interpolation grid or the number of points. In other words, it is parametrization data-vector:
                * If it is ``np.ndarray`` that is interpreted as grid of interpolation.
                  The grid should be 1xM array with increasing ordered values.
                * If it is ``int`` that is interpreted as the number of points in uniformly interpolated curve.
        method : str
            Interpolation method:
                * ``linear`` -- linear interpolation
                * ``cubic`` -- cubic spline interpolation
                * ``hermite`` -- piecewise-cubic interpolation matching values and first derivatives
                * ``akima`` -- Akima interpolation
                * ``pchip`` -- PCHIP 1-d monotonic cubic interpolation
                * ``spline`` -- General k-order weighted spline interpolation

        **kwargs : mapping
            Additional interpolator parameters dependent on interpolation method

        Returns
        -------
        curve : Curve
            Interpolated curve

        Raises
        ------
        ValueError : invalid data or parameters
        InterpolationError : any computation of interpolation errors

        Examples
        --------

        .. code-block:: python

            >>> curve = Curve([[1, 3, 6], [1, 3, 6]])
            >>> curve
            Curve([[1. 1.]
                   [3. 3.]
                   [6. 6.]], size=3, ndim=2, dtype=float64)

            >>> curve.interpolate(t=6)
            Curve([[1. 1.]
                   [2. 2.]
                   [3. 3.]
                   [4. 4.]
                   [5. 5.]
                   [6. 6.]], size=6, ndim=2, dtype=float64)

        .. code-block:: python

            >>> curve = Curve([[1, 3, 6], [1, 3, 6]], ndmin=3)
            >>> curve
            Curve([[1. 1. 0.]
                   [3. 3. 0.]
                   [6. 6. 0.]], size=3, ndim=3, dtype=float64)

            >>> curve.interpolate(t=6)
            Curve([[1. 1. 0.]
                   [2. 2. 0.]
                   [3. 3. 0.]
                   [4. 4. 0.]
                   [5. 5. 0.]
                   [6. 6. 0.]], size=6, ndim=3, dtype=float64)

        """

        return _interpolate.interpolate(self, grid_spec=pcount_or_grid, method=method, **kwargs)

    def invalidate(self):
        """Invalidates the curve parameters cache

        Notes
        -----
        All parameters such as tangent, normal, curvature, etc
        after call this method will be re-calculated.

        """

        cls = type(self)
        cached_props = [attr for attr in self.__dict__
                        if isinstance(getattr(cls, attr, None), cached_property)]

        for prop in cached_props:
            self.__dict__.pop(prop, None)

    def _check_ndim(self, other: PointCurveUnionType):
        if self.ndim != other.ndim:
            raise ValueError('The dimensions of the curve and other object do not match')
