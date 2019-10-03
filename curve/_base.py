# -*- coding: utf-8 -*-

"""
The module provides data types to manipulate n-dimensional geometric curves

The module contains the following basic classes:

    * `Point`
    * `CurvePoint`
    * `Curve`

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

from curve._distance import MetricType, get_metric
from curve._utils import as2d
from curve._numeric import allequal
from curve import _diffgeom
from curve._interpolate import InterpGridSpecType, interpolate
from curve._smooth import smooth
from curve._intersect import intersect, SegmentsIntersection


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
                'Invalid point data: {}\nThe point data must be 1-D array or sequence.'.format(point_data))

        self._data = data
        self._data.flags.writeable = False

    def __repr__(self) -> str:
        with np.printoptions(suppress=True, precision=DATA_FORMAT_PRECISION):
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

    def distance(self, other: 'Point', metric: MetricType = 'euclidean', **kwargs) -> Numeric:
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
                raise TypeError("unsupported operand type(s) for '{}': '{}' and '{}'".format(
                    op.__name__, type(self).__name__, type(other).__name__))

        if right:
            right_data, left_data = left_data, right_data

        return Point(op(left_data, right_data))


class CurvePoint(Point):
    """The class represents nd-point that is a n-dimensional curve point

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
            data_str = '{}'.format(self._data)

        return '{}({}, index={})'.format(
            type(self).__name__, data_str, self.idx)

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

    @property
    def cumarclen(self) -> float:
        """Returns value of cumulative arc length for this point

        Returns
        -------
        val : float
            The value of cumulative arc length for this point

        """

        return self.curve.cumarclen[self.idx]

    @property
    def firstderiv(self) -> np.ndarray:
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
    def secondderiv(self) -> np.ndarray:
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
    def thirdderiv(self) -> np.ndarray:
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
    def tangent(self) -> np.ndarray:
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
    def normal(self) -> np.ndarray:
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
    def binormal(self) -> np.ndarray:
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
    def speed(self) -> float:
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
    def frenet1(self) -> np.ndarray:
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
    def frenet2(self) -> np.ndarray:
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
    def frenet3(self) -> np.ndarray:
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
    def curvature(self) -> float:
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
    def torsion(self) -> float:
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


class CurveSegment:
    """Represents a curve segment

    Parameters
    ----------
    curve : Curve
        The curve object
    index : int
        The segment index in the curve

    """

    __slots__ = ('_curve', '_p1', '_p2', '_idx')

    def __init__(self, curve: 'Curve', index: int) -> None:
        if index < 0:
            index = (curve.size - 1) + index
        if index >= (curve.size - 1):
            raise ValueError('The index is out of curve size')

        self._curve = curve
        self._p1 = ty.cast(CurvePoint, curve[index])
        self._p2 = ty.cast(CurvePoint, curve[index + 1])
        self._idx = index

    def __repr__(self) -> str:
        with np.printoptions(suppress=True, precision=DATA_FORMAT_PRECISION):
            p1_data = '{}'.format(self._p1.data)
            p2_data = '{}'.format(self._p2.data)

        return '{}(p1={}, p2={}, len={:.4f}, index={})'.format(
            type(self).__name__, p1_data, p2_data, self.chordlen, self._idx)

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
    def p1(self) -> 'CurvePoint':
        """Returns beginning point of the segment

        Returns
        -------
        point : CurvePoint
            Beginning point of the segment
        """

        return self._p1

    @property
    def p2(self) -> 'CurvePoint':
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
            The segment data
        """

        return np.array([self._p1.data, self._p2.data])

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

    def seglen(self) -> Numeric:
        """Returns the segment length

        Returns
        -------
        length : float
            Segment length (Euclidean distance between p1 and p2)
        """

        return self._p1.distance(self._p2)

    def dot(self) -> float:
        """Returns Dot product of the segment points

        Returns
        -------
        dot : float
            The Dot product of the segment points
        """

        return self._p1 @ self._p2

    def direction(self) -> 'Point':
        """Returns the segment (line) direction vector

        Returns
        -------
        u : Point
            The point object that represents the segment direction

        """

        return self.p2 - self.p1

    def point(self, t: float) -> 'Point':
        """Returns the point on the segment for given "t"-parameter value

        The parametric line equation:

        .. math::

            P(t) = P_1 + t (P_2 - P_1)

        Parameters
        ----------
        t : float
            The parameter value in the range [0, 1] to get point on the segment

        Returns
        -------
        point : Point
            The point on the segment for given "t"

        """

        return self.p1 + self.direction() * t

    def angle(self, other: 'CurveSegment', ndigits: ty.Optional[int] = None) -> float:
        """Returns the angle between this segment and other segment

        Parameters
        ----------
        other : CurveSegment
            Other segment
        ndigits : int, None
            The number of significant digits

        Returns
        -------
        phi : float
            The angle in radians between this segment and other segment

        """

        u1 = self.direction()
        u2 = other.direction()

        cos_phi = (u1 @ u2) / (u1.norm() * u2.norm())

        if ndigits is not None:
            cos_phi = round(cos_phi, ndigits=ndigits)

        # We need to consider floating point errors
        cos_phi = 1.0 if cos_phi > 1.0 else cos_phi
        cos_phi = -1.0 if cos_phi < -1.0 else cos_phi

        return np.arccos(cos_phi)

    def collinear(self, other: ty.Union['CurveSegment', 'Point'],
                  tol: ty.Optional[float] = None) -> bool:
        """Returns True if the segment and other segment or point are collinear

        Parameters
        ----------
        other : CurveSegment, Point
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
        if not isinstance(other, (CurveSegment, Point)):
            raise TypeError('Unsupported type of "other" argument {}'.format(type(other)))

        m = np.vstack((self.data, other.data)).T
        return np.linalg.matrix_rank(m, tol=tol) <= 1

    def parallel(self, other: 'CurveSegment',
                 ndigits: ty.Optional[int] = 8,
                 rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Returns True if the segment and other segment are parallel

        Parameters
        ----------
        other : CurveSegment
            Other segment
        ndigits : int, None
            The number of significant digits
        rtol : float
            Relative tolerance with check angle
        atol : float
            Absolute tolerance with check angle

        Returns
        -------
        flag : bool
            True if the segment and other segment are parallel

        See Also
        --------
        collinear
        angle

        """

        phi = self.angle(other, ndigits=ndigits)
        return np.isclose(phi, [0., np.pi], rtol=rtol, atol=atol).any()

    def coplanar(self, other: ty.Union['CurveSegment', 'Point'],
                 tol: ty.Optional[float] = None) -> bool:
        """Returns True if the segment and other segment or point are coplanar

        Parameters
        ----------
        other : CurveSegment, Point
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

        m = self.data.copy()

        if isinstance(other, Point):
            m -= other.data
        elif isinstance(other, CurveSegment):
            m = np.vstack((m, other.p1.data))
            m -= other.p2.data
        else:
            raise TypeError('"other" argument must be type \'Point\' or \'CurveSegment\'')

        return np.linalg.matrix_rank(m, tol=tol) <= 2

    def intersect(self, other: ty.Union['CurveSegment', 'Curve']) \
            -> ty.Union[None, SegmentsIntersection, ty.List[SegmentsIntersection]]:
        """Determines intersection(s) between the segment and other segment or curve

        Parameters
        ----------
        other : CurveSegment, Curve
            Other curve segment or curve object

        Returns
        -------
        intersections : SegmentsIntersection, List[SegmentsIntersection], None
            Intersection(s) info

        """

        if isinstance(other, CurveSegment):
            intersections = self.to_curve().intersect(other)
            if intersections:
                return SegmentsIntersection(
                    segment1=self,
                    segment2=other,
                    intersect_point=intersections[0].intersect_point,
                )
            else:
                return None
        elif isinstance(other, Curve):
            return [intersection.swap_segments() for intersection in other.intersect(self)]
        else:
            raise TypeError('"other" argument must be "CurveSegment" or "Curve" class.')

    def to_curve(self) -> 'Curve':
        """Returns the copy of segment data as curve object with 2 points

        Returns
        -------
        curve : Curve
            Curve object with 2 points
        """

        return Curve(self.data)


class Curve(abc.Sequence):
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
            ValueError('"dtype" must be a numeric type not {}.'.format(dtype))

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
                raise ValueError('"tdata" must be 1-D array')
            if tdata.size != data.shape[0]:
                raise ValueError('"tdata" size must be equal to the number of curve points.')

        self._data = data  # type: np.ndarray
        self._data.flags.writeable = False

        self._tdata = tdata

    def __repr__(self) -> str:
        name = type(self).__name__

        with np.printoptions(suppress=True, precision=DATA_FORMAT_PRECISION,
                             edgeitems=4, threshold=10*self.ndim):
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
            raise TypeError('Invalid index type {}'.format(type(indexer)))

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
    def cumarclen(self) -> np.ndarray:
        """Returns the cumulative arc length of the curve (natural parametrization)

        Parametrization of the curve by the length of its arc.

        Returns
        -------
        cumarc : np.ndarray
            The 1xM array cumulative arc

        See Also
        --------
        chordlen
        arclen

        """

        return _diffgeom.cumarclen(self)

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

        Notes
        -----
        By default the curve orients to XY plane orientation.

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
                'Index {} is out of bounds for curve size {}'.format(
                    index, self.size)) from err

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
                'Index {} is out of bounds for curve size {}'.format(
                    index, self.size)) from err

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
            raise ValueError(
                'The axis {} is out of the curve dimensions {}'.format(axis, self.ndim))

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
                'Axis {} is out of bounds for curve dimensions {}'.format(axis, self.ndim)) from err

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

        Notes
        -----
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
                'Axis {} is out of bounds for curve dimensions {}'.format(axis, self.ndim)) from err

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

    def interpolate(self, grid_spec: InterpGridSpecType, method: str = 'linear', **kwargs) -> 'Curve':
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

        return interpolate(self, grid_spec=grid_spec, method=method, **kwargs)

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

        return smooth(self, method, **params)

    def intersect(self, other: ty.Optional[ty.Union['Curve', CurveSegment]] = None) -> ty.List[SegmentsIntersection]:
        """Determines the curve intersections with other curve/segment or itself

        Parameters
        ----------
        other : Curve, CurveSegment, None
            Other object to determine intersection or None for itself

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
            curve2 = self
        elif isinstance(other, Curve):
            curve2 = other
        elif isinstance(other, CurveSegment):
            curve2 = other.to_curve()
        else:
            raise TypeError('"other" object must be "Curve" or "CurveSegment" class or None')

        intersect_result = intersect(self, curve2)
        intersections = []

        if not intersect_result:
            return intersections

        for seg1, seg2, point_data in zip(intersect_result.segments1,
                                          intersect_result.segments2,
                                          intersect_result.intersect_points):
            if other is None or isinstance(other, Curve):
                segment2 = CurveSegment(curve2, seg2)
            else:
                segment2 = other

            intersections.append(SegmentsIntersection(
                segment1=CurveSegment(self, seg1),
                segment2=segment2,
                intersect_point=Point(point_data),
            ))

        return intersections

    def _check_ndim(self, other: PointCurveUnion):
        if self.ndim != other.ndim:
            raise ValueError('The dimensions of the curve and other object do not match')
