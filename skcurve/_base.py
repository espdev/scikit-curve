# -*- coding: utf-8 -*-

"""
The module provides data types to manipulate n-dimensional geometric curves

The module contains the following basic classes:

    * `Point` -- represents n-dimensional geometric point in Euclidean space
    * `Segment` -- represents n-dimensional geometric segment in Euclidean space
    * `Curve` -- represents a n-dimensional geometric curve in Euclidean space
    * `CurvePoint` -- represents a n-dimensional curve point
    * `CurveSegment` -- represents a n-dimensional curve segment

"""

import collections.abc as abc
import typing as ty
import numbers
import operator
import textwrap
import enum
import warnings

import numpy as np

from cached_property import cached_property

import skcurve._distance as _distance
import skcurve._diffgeom as _diffgeom
import skcurve._geomalg as _geomalg
import skcurve._intersect as _intersect
import skcurve._interpolate as _interpolate
import skcurve._smooth as _smooth
from skcurve._numeric import allequal, F_EPS
from skcurve._utils import as2d


Numeric = ty.Union[numbers.Number, np.number]
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


DEFAULT_DTYPE = np.float64
DATA_FORMAT_PRECISION = 4


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

    The class represents n-dimensional geometric point. `Point` class is immutable.

    Parameters
    ----------
    point_data : PointDataType
        The data of n-dimensional point. The data might be represented in the different types:

        * The sequence of numbers ``Sequence[NumericType]``
        * 1-D np.ndarray 1xN where N is point dimension
        * Another `Point` or `CurvePoint` object.
          It creates the copy of the data of another point.

    dtype : numeric type or numeric `numpy.dtype`
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

    def __init__(self, point_data: PointData, dtype: DType = None) -> None:
        """Constructs the `Point` instance
        """

        if isinstance(point_data, Point):
            point_data = point_data.data

        if dtype is None:
            dtype = DEFAULT_DTYPE

        if not np.issubdtype(dtype, np.number):
            ValueError('dtype must be a numeric type.')

        data = np.array(point_data, dtype=np.dtype(dtype))

        if data.ndim > 1:
            raise ValueError(
                f'Invalid point data: {point_data}\nThe point data must be 1-D array or sequence.')

        self._data = data
        self._data.flags.writeable = False

    def __repr__(self) -> str:
        with np.printoptions(suppress=True, precision=DATA_FORMAT_PRECISION):
            data_str = f'{self._data}'
        return f'{type(self).__name__}({data_str}, ndim={self.ndim}, dtype={self._data.dtype})'

    def __len__(self) -> int:
        """Returns the point dimension

        Returns
        -------
        ndim : int
            The point dimension

        """

        return self._data.size

    def __bool__(self) -> bool:
        return self._data.size > 0

    def __getitem__(self, index: ty.Union[int, slice]) -> ty.Union['Point', np.number]:
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

    def __eq__(self, other: object) -> bool:
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

    def __add__(self, other: ty.Union['Point', Numeric]) -> 'Point':
        return self._op(operator.add, other)

    def __radd__(self, other: ty.Union['Point', Numeric]) -> ty.Optional['Point']:
        return self._op(operator.add, other, right=True)

    def __sub__(self, other: ty.Union['Point', Numeric]) -> 'Point':
        return self._op(operator.sub, other)

    def __rsub__(self, other: ty.Union['Point', Numeric]) -> ty.Optional['Point']:
        return self._op(operator.sub, other, right=True)

    def __mul__(self, other: ty.Union['Point', Numeric]) -> 'Point':
        return self._op(operator.mul, other)

    def __rmul__(self, other: ty.Union['Point', Numeric]) -> ty.Optional['Point']:
        return self._op(operator.mul, other, right=True)

    def __truediv__(self, other: ty.Union['Point', Numeric]) -> 'Point':
        return self._op(operator.truediv, other)

    def __rtruediv__(self, other: ty.Union['Point', Numeric]) -> ty.Optional['Point']:
        return self._op(operator.truediv, other, right=True)

    def __floordiv__(self, other: ty.Union['Point', Numeric]) -> 'Point':
        return self._op(operator.floordiv, other)

    def __rfloordiv__(self, other: ty.Union['Point', Numeric]) -> ty.Optional['Point']:
        return self._op(operator.floordiv, other, right=True)

    def __matmul__(self, other: 'Point') -> Numeric:
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

        return ty.cast(Numeric, np.dot(self._data, other.data))

    def __copy__(self) -> 'Point':
        return self.__deepcopy__()

    def __deepcopy__(self, memodict: ty.Optional[dict] = None) -> 'Point':
        return Point(self)

    @property
    def data(self) -> np.ndarray:
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

    def norm(self) -> float:
        """Returns norm of vector that represented in point object

        Returns
        -------
        norm : float
            Norm of vector

        """

        return np.sqrt(self @ self)

    def distance(self, other: 'Point', metric: _distance.MetricType = 'euclidean', **kwargs) -> Numeric:
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
            metric = _distance.get_metric(metric, **kwargs)

        if not callable(metric):
            raise TypeError('Metric must be str or callable')

        return metric(self._data, other.data)

    def _op(self, op, other: ty.Union['Point', Numeric], right: bool = False) -> ty.Optional['Point']:
        left_data = self._data

        if isinstance(other, Point):
            right_data = other.data
        elif isinstance(other, numbers.Number):
            right_data = other
        else:
            if right:
                return NotImplemented
            else:
                raise TypeError(
                    f"unsupported operand type(s) for '{op.__name__}': "
                    f"'{type(self).__name__}' and '{type(other).__name__}'")

        if right:
            right_data, left_data = left_data, right_data

        return Point(op(left_data, right_data))


class CurvePoint(Point, _diffgeom.CurvePointFunctionMixin):
    """The class represents a n-dimensional curve point

    This class is the view wrapper for a curve point data. This class should not be used directly.
    It is used in Curve class.

    The class provides additional data and parameters of the curve point.

    Parameters
    ----------
    curve : Curve
        Curve object

    index : int
        The point index in a curve

    """

    __slots__ = Point.__slots__ + ('_curve', '_idx')

    def __init__(self, curve: 'Curve', index: int):
        if index < 0:
            index = curve.size + index

        self._curve = curve
        self._idx = index

        super().__init__(curve.data[index])

    def __repr__(self):
        with np.printoptions(suppress=True, precision=DATA_FORMAT_PRECISION):
            data_str = f'{self._data}'
        return f'{type(self).__name__}({data_str}, index={self.idx})'

    def __copy__(self) -> 'CurvePoint':
        return self.__deepcopy__()

    def __deepcopy__(self, memodict: ty.Optional[dict] = None) -> 'CurvePoint':
        return CurvePoint(self.curve, self.idx)

    @property
    def curve(self) -> 'Curve':
        """Returns the reference to the curve object

        Returns
        -------
        curve : Curve
            Curve object for this point

        See Also
        --------
        Curve

        """

        return self._curve

    @property
    def idx(self) -> int:
        """Returns the point index in the curve

        Returns
        -------
        index : int
            The point index in the curve or None if the curve instance has been deleted.

        """

        return self._idx

    @property
    def t(self) -> float:
        """Returns value of ``t`` parametric vector for this point

        Returns
        -------
        tval : numeric
            The value of ``t`` parametric vector for this point

        """

        return self.curve.t[self.idx]

    def subcurve(self, other_point: 'CurvePoint', endpoint: bool = True) -> 'Curve':
        """Returns a sub-curve from the point to other point in the same curve

        Parameters
        ----------
        other_point : CurvePoint
            Other point in the same curve
        endpoint : bool
            If this flag is True, other point will be included to a sub-curve as end point.

        Returns
        -------
        curve : Curve
            A sub-curve from the point to other curve point

        Raises
        ------
        TypeError : Other point is not an instance of "CurvePoint" class
        ValueError : Other point belongs to another curve

        """

        if not isinstance(other_point, CurvePoint):
            raise TypeError('Other point must be an instance of "CurvePoint" class')

        if self.curve is not other_point.curve:
            raise ValueError('Other point belongs to another curve')

        inc = 1 if endpoint else 0
        return self.curve[self.idx:other_point.idx+inc]


class Segment:
    """Represents a segment in n-dimensional Euclidean space

    Parameters
    ----------
    p1 : Point
        Beginning point of segment
    p2 : Point
        Ending point of segment

    """

    __slots__ = ('_p1', '_p2')

    def __init__(self, p1: 'Point', p2: 'Point') -> None:
        if not isinstance(p1, Point) or not isinstance(p2, Point):
            raise TypeError('Invalid type of "p1" or "p2". It must be points.')
        if not p1 or not p2:
            raise ValueError('Points dimension must be greater or equal to 1.')
        if p1.ndim != p2.ndim:
            raise ValueError('"p1" and "p2" points must be the same dimension.')

        self._p1 = p1
        self._p2 = p2

    def __repr__(self) -> str:
        with np.printoptions(suppress=True, precision=DATA_FORMAT_PRECISION):
            p1_data = f'{self._p1.data}'
            p2_data = f'{self._p2.data}'
        return f'{type(self).__name__}(p1={p1_data}, p2={p2_data})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Segment):
            return NotImplemented
        return self.p1 == other.p1 and self.p2 == other.p2

    @property
    def p1(self) -> 'Point':
        """Returns beginning point of the segment

        Returns
        -------
        point : CurvePoint
            Beginning point of the segment
        """

        return self._p1

    @property
    def p2(self) -> 'Point':
        """Returns ending point of the segment

        Returns
        -------
        point : CurvePoint
            Ending point of the segment
        """

        return self._p2

    @property
    def data(self) -> np.ndarray:
        """Returns the segment data as numpy array 2xN

        Returns
        -------
        data : np.ndarray
            The segment data array 2xN
        """

        return np.vstack((self._p1.data, self._p2.data))

    @property
    def ndim(self) -> int:
        """Returns dimension of the segment

        Returns
        -------
        ndim : int
            Dimension of the segment
        """

        return self._p1.ndim

    @property
    def singular(self) -> bool:
        """Returns True if the segment is singular (has zero lenght)

        Returns
        -------
        flag : bool
            Returns True if the segment is singular (has zero lenght)
        """

        return self.p1 == self.p2

    @property
    def seglen(self) -> Numeric:
        """Returns the segment length

        Returns
        -------
        length : float
            Segment length (Euclidean distance between p1 and p2)
        """

        return self._p1.distance(self._p2)

    @property
    def dot(self) -> float:
        """Returns Dot product of the beginning/ending segment points

        Returns
        -------
        dot : float
            The Dot product of the segment points
        """

        return self._p1 @ self._p2

    @property
    def direction(self) -> 'Point':
        """Returns the segment (line) direction vector

        Returns
        -------
        u : Point
            The point object that represents the segment direction
        """

        return self.p2 - self.p1

    def point(self, t: ty.Union[float, ty.Sequence[float], np.ndarray]) -> ty.Union['Point', ty.List['Point']]:
        """Returns the point(s) on the segment for given "t"-parameter value or list of values

        The parametric line equation:

        .. math::

            P(t) = P_1 + t (P_2 - P_1)

        Parameters
        ----------
        t : float
            The parameter value in the range [0, 1] to get point on the segment

        Returns
        -------
        point : Point, List[Points]
            The point(s) on the segment for given "t"
        """

        return _geomalg.segment_point(self, t)

    def t(self, point: ty.Union['Point', ty.Sequence['Point']],
          tol: ty.Optional[float] = None) -> ty.Union[float, np.ndarray]:
        """Returns "t"-parameter value(s) for given point(s) that collinear with the segment

        Parameters
        ----------
        point : Point, Sequence[Point]
            Point or sequence of points that collinear with the segment
        tol : float, None
            Threshold below which SVD values are considered zero

        Returns
        -------
        t : float, np.ndarray
            "t"-parameter value(s) for given points or nan
            if point(s) are not collinear with the segment

        """

        return _geomalg.segment_t(self, point, tol=tol)

    def angle(self, other: 'Segment', ndigits: ty.Optional[int] = None) -> float:
        """Returns the angle between this segment and other segment

        Parameters
        ----------
        other : Segment
            Other segment
        ndigits : int, None
            The number of significant digits

        Returns
        -------
        phi : float
            The angle in radians between this segment and other segment
        """

        if not isinstance(other, Segment):
            raise TypeError('The type of "other" argument must be \'Segment\'.')

        return _geomalg.segments_angle(self, other, ndigits=ndigits)

    def parallel(self, other: 'Segment', tol: float = F_EPS) -> bool:
        """Returns True if the segment and other segment are parallel

        Parameters
        ----------
        other : Segment
            Other segment
        tol : float
            Epsilon. It is a small float number. By default float64 eps

        Returns
        -------
        flag : bool
            True if the segment and other segment are parallel

        See Also
        --------
        collinear
        angle

        """

        return _geomalg.parallel_segments(self, other, tol=tol)

    def collinear(self, other: ty.Union['Segment', 'Point'],
                  tol: ty.Optional[float] = None) -> bool:
        """Returns True if the segment and other segment or point are collinear

        Parameters
        ----------
        other : Segment, Point
            The curve segment or point object
        tol : float, None
            Threshold below which SVD values are considered zero

        Returns
        -------
        flag : bool
            True if the segment and other segment or point are collinear

        See Also
        --------
        parallel
        coplanar

        """

        points = [self.p1, self.p2]

        if isinstance(other, Point):
            if other == self.p1 or other == self.p2:
                return True
            points.append(other)
        elif isinstance(other, Segment):
            if other == self or other.reverse() == self:
                return True
            points.extend([other.p1, other.p2])
        else:
            raise TypeError('"other" argument must be type \'Point\' or \'Segment\'.')

        return _geomalg.collinear_points(points, tol=tol)

    def coplanar(self, other: ty.Union['Segment', 'Point'],
                 tol: ty.Optional[float] = None) -> bool:
        """Returns True if the segment and other segment or point are coplanar

        Parameters
        ----------
        other : Segment, Point
            The curve segment or point object
        tol : float, None
            Threshold below which SVD values are considered zero

        Returns
        -------
        flag : bool
            True if the segment and other segment or point are coplanar

        See Also
        --------
        collinear

        """

        points = [self.p1, self.p2]

        if isinstance(other, Point):
            points.append(other)
        elif isinstance(other, Segment):
            points.extend([other.p1, other.p2])
        else:
            raise TypeError('"other" argument must be type \'Point\' or \'Segment\'')

        return _geomalg.coplanar_points(points, tol=tol)

    def overlap(self, other: 'Segment',
                check_collinear: bool = False) -> ty.Optional['Segment']:
        """Returns overlap segment between the segment and other segment if it exists

        Parameters
        ----------
        other : Segment
            Other segment
        check_collinear : bool
            If the flag is True segments collinearity will be check

        Returns
        -------
        segment : Segment, None
            Overlap segment if it is exist.

        """

        return _geomalg.overlap_segments(self, other, check_collinear)

    def intersect(self, other: 'Segment',
                  method: ty.Optional[ty.Union[str, _intersect.IntersectionMethodBase]] = None,
                  **params) -> _intersect.SegmentsIntersection:
        """Finds the intersection of the segment and other segment

        Parameters
        ----------
        other : Segment
            Other segment
        method : str, None
            The method to determine intersection. By default the following methods are available:
                - ``exact`` -- (default) the exact intersection solving the system of equations
                - ``almost`` -- the almost intersection using the shortest connecting segment.
                  This is usually actual for dimension >= 3.

                The default method is ``exact``.

            if ``method`` is an instance of subclass of ``IntersectionMethodBase`` it will be used directly
            and ``params`` will be ignored.

        params : mapping
            The intersection method parameters

        Returns
        -------
        res : SegmentsIntersection
        """

        return _intersect.intersect(self, other, method=method, **params)

    def distance(self, other: ty.Union['Point', 'Segment']) -> float:
        """Computes the shortest distance between the segment and given point or segment

        Parameters
        ----------
        other : Point, Segment
            The point or segment object

        Returns
        -------
        dist : float
            The shortest distance

        See Also
        --------
        shortest_segment

        """

        return self.shortest_segment(other).seglen

    def shortest_segment(self, other: ty.Union['Point', 'Segment']) -> 'Segment':
        """Returns the shortest segment between the segment and given point or segment

        Parameters
        ----------
        other : Point, Segment
            The point or segment object

        Returns
        -------
        segment : Segment
            Shortest segment

        See Also
        --------
        distance

        """

        if isinstance(other, Point):
            t = _geomalg.segment_to_point(self, other)
            p1 = self.point(t)
            p2 = Point(other)
        elif isinstance(other, Segment):
            t1, t2 = _geomalg.segment_to_segment(self, other)
            p1 = self.point(t1)
            p2 = other.point(t2)
        else:
            raise TypeError('"other" argument must be \'Point\' or \'Segment\' type.')

        return Segment(p1, p2)

    def to_curve(self) -> 'Curve':
        """Returns the copy of segment data as curve object with 2 points

        Returns
        -------
        curve : Curve
            Curve object with 2 points
        """

        return Curve(self.data)

    def reverse(self) -> 'Segment':
        """Returns the reversed segment with swapped beginning and ending points

        Returns
        -------
        segment : Segment
            Reversed segment
        """

        return Segment(self.p2, self.p1)


class CurveSegment(Segment):
    """Represents a curve segment

    Parameters
    ----------
    curve : Curve
        The curve object
    index : int
        The segment index in the curve

    """

    __slots__ = Segment.__slots__ + ('_curve', '_idx')

    def __init__(self, curve: 'Curve', index: int) -> None:
        if index < 0:
            index = (curve.size - 1) + index
        if index >= (curve.size - 1):
            raise ValueError('The index is out of curve size')

        self._curve = curve
        self._idx = index

        p1 = ty.cast(CurvePoint, curve[index])
        p2 = ty.cast(CurvePoint, curve[index + 1])

        super().__init__(p1, p2)

    def __repr__(self) -> str:
        with np.printoptions(suppress=True, precision=DATA_FORMAT_PRECISION):
            p1_data = f'{self._p1.data}'
            p2_data = f'{self._p2.data}'
        return f'{type(self).__name__}(p1={p1_data}, p2={p2_data}, index={self._idx})'

    @property
    def curve(self) -> 'Curve':
        """Returns the curve that contains this segment

        Returns
        -------
        curve : Curve
            The curve that contains this segment
        """

        return self._curve

    @property
    def idx(self) -> int:
        """Returns the segment index in the curve

        Returns
        -------
        idx : int
            The segment index in the curve
        """

        return self._idx

    @property
    def chordlen(self) -> float:
        """Returns the curve chord length for this segment

        This length may be greater or equal to segment length (distance).

        Returns
        -------
        length : float
            The curve chord length for this segment
        """

        return self._curve.chordlen[self._idx]


class Curve(abc.Sequence, _diffgeom.CurvePointFunctionMixin):
    r"""The main class for n-dimensional geometric curve representation

    The class represents n-dimensional geometric curve in the plane
    or in the Euclidean n-dimensional space given by a finity sequence of points.

    Internal data storage of curve points is NumPy MxN array where M is the number
    of curve points and N is curve dimension. In other words, n-dimensional curve data
    is stored in 2-d array::

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
    Curve class implements ``Sequence`` interface and the `Curve` objects are immutable.

    Parameters
    ----------
    curve_data : CurveData
        The data of n-dimensional curve (2 or higher). The data might be represented in the different types:

        * The sequence of the vectors with coordinates for every dimension:
          ``[X, Y, Z, ..., N]`` where X, Y, ... are 1xM arrays.
        * The data is represented as np.ndarray MxN where M is number of points and N is curve dimension.
          N must be at least 2 (a plane curve).
        * Another Curve object. It creates the copy of another curve.

        If "curve_data" is empty, the empty curve will be created with ndmin dimensions
        (2 by default, see ``ndmin`` argument).

    tdata : Optional[Union[NumericSequence, np.ndarray]
        "tdata" defines parametrization vector ``t`` that was used for calculaing
        "curve_data" for parametric curve :math:`\gamma(t) = (x(t), y(t), ..., n(t))`.
        If "curve_data" is a `Curve` object, "tdata" argument will be ignored.

        See also `isparametric` and `t` `Curve` class properties.

    axis : Optional[int]
        "axis" will be used to interpret "curve_data".
        In other words, "axis" is the axis along which located curve data points.
        If "axis" is not set, it will be set to 0 if "curve_data" is numpy array
        or 'Curve' object and 1 in other cases.

        If you set "curve_data" as list of coord-arrays ``[X, Y, ..., N]``
        or from other `Curve`, you should not set "axis" argument.
        This parameter may be useful if you want to set "curve_data" as an array
        with N-dimensional shape.

    ndmin : Optional[int]
        The minimum curve dimension. By default it is ``None`` and equal to input data dimension.
        If ``ndmin`` is more than input data dimension, additional dimensions will be added to
        created curve object. All values in additional dimensions are equal to zero.

    dtype : numeric type or numeric numpy.dtype
        The type of curve data. The type must be a numeric type. For example, ``float``, ``int``, ``np.float32``, ...
        If ``dtype`` argument is not set, by default dtype of curve data is ``np.float64``.

    Raises
    ------
    ValueError : If the input data is invalid
    ValueError : There is not a numeric ``dtype``

    Examples
    --------

    .. code-block:: python

        # 2-D curve with 5 points from list of lists
        curve = Curve([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    .. code-block:: python

        # 2-D curve with 4 points from numpy array
        curve = Curve(np.array([(1, 2, 3, 4), (5, 6, 7, 8)]),
                      axis=1, dtype=np.float32)

    .. code-block:: python

        # 3-D curve with 10 random points
        curve = Curve(np.random.rand(10, 3))

    .. code-block:: python

        # 3-D curve from 2-D data with 5 random points
        curve = Curve(np.random.rand(5, 2), ndmin=3)

    """

    def __init__(self,
                 curve_data: CurveData,
                 tdata: ty.Optional[ty.Union[NumericSequence, np.ndarray]] = None,
                 axis: ty.Optional[int] = None,
                 ndmin: ty.Optional[int] = None,
                 dtype: DType = None) -> None:
        """Constructs Curve instance
        """

        is_ndmin = False
        is_array = isinstance(curve_data, (np.ndarray, Curve))

        if ndmin is None:
            ndmin = 2
        else:
            is_ndmin = True

        if ndmin < 2:
            raise ValueError('ndmin must be at least of 2')

        if isinstance(curve_data, Curve):
            if tdata is not None:
                warnings.warn('Ignoring "tdata" argument because "curve_data" is a "Curve" object.')
            if axis is not None:
                warnings.warn('Ignoring "axis" argument because "curve_data" is a "Curve" object.', RuntimeWarning)

            if curve_data.isparametric:
                tdata = curve_data.t
            else:
                tdata = None

            curve_data = curve_data.data
            axis = 0

        if axis is None:
            axis = 0 if is_array else 1

        if is_array:
            dtype = dtype or curve_data.dtype

        if dtype is None:
            dtype = DEFAULT_DTYPE
        dtype = np.dtype(dtype)

        if not np.issubdtype(dtype, np.number):
            ValueError(f"'dtype' must be a numeric type not {dtype}.")

        data = as2d(curve_data, axis=axis).astype(dtype)

        if data.size == 0:
            data = np.reshape([], (0, ndmin)).astype(dtype)

        m, n = data.shape

        if data.size > 0 and n < 2:
            raise ValueError('The input data must be at least 2-dimensinal (a curve in the plane).')

        if is_ndmin and m > 0 and n < ndmin:
            # Change dimension to ndmin
            data = np.hstack([data, np.zeros((m, ndmin - n), dtype=dtype)])

        if tdata is not None:
            tdata = np.array(tdata, dtype=DEFAULT_DTYPE)
            tdata.flags.writeable = False

            if tdata.ndim != 1:
                raise ValueError("'tdata' must be 1-D array")
            if tdata.size != data.shape[0]:
                raise ValueError("'tdata' size must be equal to the number of curve points.")

        self._data = data  # type: np.ndarray
        self._data.flags.writeable = False

        self._tdata = tdata

    def __repr__(self) -> str:
        name = type(self).__name__

        with np.printoptions(suppress=True, precision=DATA_FORMAT_PRECISION,
                             edgeitems=4, threshold=10*self.ndim):
            arr_repr = f'{self._data}'
            arr_repr = textwrap.indent(arr_repr, ' ' * (len(name) + 1)).strip()
        return f'{name}({arr_repr}, size={self.size}, ndim={self.ndim}, dtype={self.dtype})'

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

    def __getitem__(self, indexer: Indexer) -> ty.Union[CurvePoint, 'Curve']:
        """Returns the point of curve or sub-curve or all coord values fot given axis

        Parameters
        ----------
        indexer : int, slice, sequence, np.array
            Index (int) or list of indexes or slice for getting the curve point or sub-curve

        Raises
        ------
        TypeError : Invalid index type
        IndexError : The index out of bounds curve size or dimensions

        Returns
        -------
        point : CurvePoint
            Point for given integer index
        curve : Curve
            Sub-curve for given slice/indices

        """

        def get_curvepoint(index: int) -> CurvePoint:
            return CurvePoint(curve=self, index=index)

        def get_subcurve(index: ty.Union[slice, NumericSequence, np.ndarray]) -> Curve:
            if self.isparametric:
                tdata = self._tdata[index]
            else:
                tdata = None
            return Curve(self._data[index], tdata=tdata)

        if isinstance(indexer, (int, np.integer)):
            return get_curvepoint(indexer)

        elif isinstance(indexer, slice):
            return get_subcurve(indexer)

        elif isinstance(indexer, (np.ndarray, abc.Sequence)):
            indexer = np.asarray(indexer)
            if indexer.ndim > 1:
                raise IndexError('Indexing array must be 1-d')
            if (not np.issubdtype(indexer.dtype, np.number) and
                    not np.issubdtype(indexer.dtype, np.bool_)):
                raise IndexError('Indexing array must be numeric or boolean')
            return get_subcurve(indexer)
        else:
            raise TypeError(f'Invalid index type {type(indexer)}')

    def __contains__(self, other: object):
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

    def __eq__(self, other: object) -> bool:
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

        if self.isparametric and other.isparametric:
            tdata = np.hstack((self.t, other.t))
        else:
            tdata = None

        return Curve(np.vstack((self._data, other.data)), tdata=tdata)

    def __copy__(self) -> 'Curve':
        return self.__deepcopy__()

    def __deepcopy__(self, memodict: ty.Optional[dict] = None) -> 'Curve':
        return Curve(self)

    def index(self, point: Point, start: ty.Optional[int] = None, end: ty.Optional[int] = None) -> int:
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
            raise ValueError(f'{point} is not in curve and given interval')

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

    @property
    def isplane(self) -> bool:
        """Returns True if the curve is plane

        The plane curve may be spatial, but always is coplanar.

        Returns
        -------
        flag : bool
            If the curve is plane

        See Also
        --------
        isspatial
        torsion

        """

        return _geomalg.coplanar_points(self.data)

    @property
    def isparametric(self) -> bool:
        """Returns True if the curve has "tdata" vector (it is the parametric curve)

        Returns
        -------
        flag : bool
            True if the curve if has "tdata" vector and it is the parametric curve

        See Also
        --------
        t
        cumarclen

        """

        return self._tdata is not None

    @cached_property
    def t(self) -> np.ndarray:
        """Returns parametrization vector for the curve

        Notes
        -----
        This is "tdata" if the curve is parametric and the alias for
        `cumarclen` if the curve is not parametric.

        Returns
        -------
        t : np.ndarray
            The 1xM array parameter vector

        See Also
        --------
        chordlen
        arclen
        cumarclen
        isparametric

        """

        if self.isparametric:
            return self._tdata
        else:
            return self.cumarclen

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
    def segments(self) -> np.ndarray:
        """Returns the numpy array (list) of curve segments

        Returns
        -------
        segments : np.array[CurveSegment]
            The numpy array list of curve segments

        """

        return np.array([CurveSegment(self, idx) for idx in range(self.size - 1)], dtype=np.object)

    def reverse(self) -> 'Curve':
        """Reverses the curve

        Returns
        -------
        curve : Curve
            The reversed copy of the curve

        Notes
        -----
        Is the curve is parametric, reversed curve will also be parametric.

        """

        if self.isparametric:
            tdata = np.flip(self._tdata, axis=0)
        else:
            tdata = None

        return Curve(np.flipud(self._data), tdata=tdata)

    def coorientplane(self, axis1: int = 0, axis2: int = 1) -> 'Curve':
        """Co-orients the curve to given plane orientation

        Parameters
        ----------
        axis1: int
            First plane axis
        axis2: int
            Second plane axis

        Returns
        -------
        curve : Curve
            The co-oriented curve copy

        Raises
        ------
        ValueError : Curve has the dimension less than 2
        IndexError : Axis out of the curve dimensions

        Notes
        -----
        By default the curve orients to XY plane orientation.

        Is the curve is parametric, co-oriented curve will also be parametric.

        """

        is_coorient = _diffgeom.coorientplane(self, axis1=axis1, axis2=axis2)

        if not is_coorient:
            return self.reverse()
        else:
            return Curve(self, tdata=self._tdata)

    def insert(self, index: Indexer, other: PointCurveUnion) -> 'Curve':
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

        Notes
        -----
        The new curve with inserted data will be not parametric.

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
                f'Index {index} is out of bounds for curve size {self.size}') from err

    def append(self, other: PointCurveUnion):
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

        Notes
        -----
        The new curve with appended data will be not parametric.

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

    def delete(self, index: Indexer) -> 'Curve':
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

        Notes
        -----
        Is the curve is parametric, the new curve with deleted data will also be parametric.

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
            if self.isparametric:
                tdata = np.delete(self._tdata, index)
            else:
                tdata = None

            data = np.delete(self._data, index, axis=0)
            return Curve(data, tdata=tdata)

        except IndexError as err:
            raise IndexError(
                f'Index {index} is out of bounds for curve size {self.size}') from err

    def values(self, axis: ty.Union[int, Axis, None] = None) -> ty.Union[np.ndarray, ty.Iterator[np.ndarray]]:
        """Returns the vector with all values for given axis or the iterator along all axes

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

        if axis is not None and not isinstance(axis, (int, np.integer)):
            raise ValueError('Axis must be an integer')

        if axis is not None and axis >= self.ndim:
            raise ValueError(f'The axis {axis} is out of the curve dimensions {self.ndim}')

        if axis is not None:
            return self._data[:, axis]
        else:
            return iter(self._data[:, i] for i in range(self.ndim))

    def insertdim(self, axis: int, values: ty.Union[np.ndarray, NumericSequence, None] = None) -> 'Curve':
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

        Notes
        -----
        The new curve will be not parametric.

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
                f'Axis {axis} is out of bounds for curve dimensions {self.ndim}') from err

    def appenddim(self, values: ty.Union[np.ndarray, NumericSequence, None] = None) -> 'Curve':
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

        Notes
        -----
        The new curve will be not parametric.

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

    def deletedim(self, axis: Indexer) -> 'Curve':
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
        ValueError : if the curve is 2-dimensional
        IndexError : indexation error

        Notes
        -----
        If the curve is 2-dimensional this operation is not allowed and raises ``ValueError``.

        The new curve will be not parametric.

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
                f'Axis {axis} is out of bounds for curve dimensions {self.ndim}') from err

    def unique(self) -> 'Curve':
        """Returns curve with unique points

        The method deletes duplicate points from the curve and return new curve
        with only unique points.

        Returns
        -------
        curve : Curve
            Curve object with unique points

        Notes
        -----
        Is the curve is parametric, the new curve with unique data will also be parametric.

        """

        # FIXME: unique is slow (O(Nlog(N)). Moreover, we are forced to use
        #  additional sorting indices array to preserve order. This is not good way...
        data, index = np.unique(self._data, axis=0, return_index=True)
        s_index = np.sort(index)

        if self.isparametric:
            tdata = self._tdata[s_index]
        else:
            tdata = None

        return Curve(self._data[s_index], tdata=tdata)

    def drop(self, isa: ty.Callable) -> 'Curve':
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

        Raises
        ------
        TypeError : Invalid ``isa`` checker argument
        ValueError : Invalid ``isa``  checker return type
        IndexError : Cannot indexing curve data with indices from ``isa`` checker

        Notes
        -----
        Is the curve is parametric, the new curve with dropped data will also be parametric.

        Examples
        --------

        .. code-block:: python

            >>> curve = Curve([(1, 2, np.nan, 3, 2, 4), (5, 6, 1, 7, np.inf, 8)])
            >>> curve.drop(lambda x: np.isnan(x) | np.isinf(x))
            Curve([[ 1.  5.]
                   [ 2.  6.]
                   [ 3.  7.]
                   [ 4.  8.]], size=4, ndim=2, dtype=float64)

        """

        if not callable(isa):
            raise TypeError('isa argument must be a callable object')

        indices = np.asarray(isa(self._data))

        if indices.ndim > 1:
            if indices.dtype != np.bool:
                raise ValueError('drop indices MxN array must be boolean type.')
            indices = np.any(indices, axis=1)

        if indices.dtype != np.bool:
            return self.delete(indices)
        else:
            if self.isparametric:
                tdata = self._tdata[~indices]
            else:
                tdata = None
            return Curve(self._data[~indices], tdata=tdata)

    def nonsingular(self):
        """Removes singularities in the curve

        The method removes NaN, Inf and the close points from curve to avoid segments with zero-closed lengths.
        These points/segments of an exceptional set where a curve fails to be well-behaved in some
        particular way, such as differentiability for example.

        Returns
        -------
        curve : Curve
            The curve without singularities.

        Notes
        -----
        Is the curve is parametric, the new curve will also be parametric.

        """

        return _diffgeom.nonsingular(self)

    def interpolate(self, grid_spec: _interpolate.InterpGridSpecType, method: str, **kwargs) -> 'Curve':
        """Interpolates the curve data

        The method interpolates the curve data by given grid or
        given number of points on uniformly interpolated curve.

        Parameters
        ----------
        grid_spec : np.ndarray, Sequence[float], int, InterpolationGrid
            Interpolation grid or the number of points. In other words, it is parametrization data-vector:
                * If it is ``np.ndarray`` or sequence that is interpreted as grid of interpolation.
                  The grid should be 1xM array with increasing ordered values.
                * If it is ``int`` that is interpreted as the number of points in uniformly interpolated curve.
                * If it is ``InterpolationGrid`` that is interpreted as interp grid object.
                  The grid data will be computed with using the curve parametrization.
        method : str
            Interpolation methods that available by default:
                * ``linear`` -- linear interpolation
                * ``cubic`` -- cubic spline interpolation
                * ``hermite`` -- piecewise-cubic interpolation matching values and first derivatives
                * ``akima`` -- Akima interpolation
                * ``pchip`` -- PCHIP 1-d monotonic cubic interpolation
                * ``spline`` -- General k-order weighted spline interpolation
                * ``csaps`` -- Smoothing weighted natural cubic spline interpolation/approximation

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

        Notes
        -----
        Is the curve is parametric, the interpolated curve will also be parametric.

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

        return _interpolate.interpolate(self, grid_spec=grid_spec, method=method, **kwargs)

    def smooth(self, method: str, **params) -> 'Curve':
        """Smoothes the curve using the given method and its parameters

        The method smoothes the curve using the given method.
        Returns the smoothed curve with the same number of points and type `np.float64`.

        Parameters
        ----------
        method : str
            Smoothing method
        params : mapping
            The parameters of smoothing method

        Returns
        -------
        curve : Curve
            Smoothed curve with type `numpy.float64`

        Raises
        ------
        ValueError : Input data or parameters have invalid values
        TypeError : Input data or parameters have invalid type
        SmoothingError : Smoothing has failed

        See Also
        --------
        smooth_methods

        Notes
        -----
        If the curve is parametric, the smoothed curve will not be parametric.

        """

        return _smooth.smooth(self, method, **params)

    def intersect(self, other: ty.Optional[ty.Union['Curve', Segment]] = None,
                  method: ty.Optional[ty.Union[str, _intersect.IntersectionMethodBase]] = None,
                  **params) -> ty.List[_intersect.SegmentsIntersection]:
        """Determines the curve intersections with other curve or segment or itself

        Parameters
        ----------
        other : Curve, Segment, None
            Other object to determine intersection or None for itself
        method : str, IntersectionMethodBase, None
            The method to determine intersection. By default the following methods are available:
                - ``exact`` -- (default) the exact intersection solving the system of equations
                - ``almost`` -- the almost intersection using the shortest connecting segment.
                  This is usually actual for dimension >= 3.

                The default method is ``exact``.

            if ``method`` is an instance of subclass of ``IntersectionMethodBase`` it will be used directly
            and ``params`` will be ignored.
        params : mapping
            The intersection method parameters

        Returns
        -------
        intersections : List[SegmentsIntersection]
            The list of intersections which represent as SegmentsIntersection objects

        Raises
        ------
        TypeError : incalid type of input args
        ValueError : invalid data

        """

        if other is None:
            other = self

        return _intersect.intersect(self, other, method=method, **params)

    def _check_ndim(self, other: PointCurveUnion):
        if self.ndim != other.ndim:
            raise ValueError('The dimensions of the curve and other object do not match.')
