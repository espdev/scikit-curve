# -*- coding: utf-8 -*-

"""
The module provides routines for determining segments and curves intersections
in n-dimensional Euclidean space.

"""

import typing as ty
import typing_extensions as ty_ext

import abc
import warnings
import enum

import numpy as np

import curve._base
from curve._numeric import F_EPS

if ty.TYPE_CHECKING:
    from curve._base import Point, Segment, Curve  # noqa


_intersect_methods = {}  # type: ty.Dict[str, ty.Type['IntersectionMethodBase']]
_default_intersect_method = None  # type: ty.Optional[str]


class IntersectionWarning(UserWarning):
    """All intersection warnings
    """


class IntersectionError(Exception):
    """All intersection errors
    """


class IntersectionType(enum.Enum):
    """The types of intersection cases
    """

    NONE = 0
    EXACT = 1
    OVERLAP = 2
    ALMOST = 3

    def __call__(self, intersect_data: ty.Optional[ty.Union['Point', 'Segment']] = None) -> 'IntersectionInfo':
        if self == IntersectionType.NONE and intersect_data is not IntersectionType:
            raise ValueError('"intersect_data" must be \'None\' for type {}'.format(self))

        if self == IntersectionType.EXACT and not isinstance(intersect_data, curve._base.Point):
            raise ValueError('"intersect_data" must be \'Point\' for type {}'.format(self))

        if (self in (IntersectionType.OVERLAP, IntersectionType.ALMOST) and
                not isinstance(intersect_data, curve._base.Segment)):
            raise ValueError('"intersect_data" must be \'Segment\' for type {}'.format(self))

        return IntersectionInfo(intersect_data, self)


IntersectionInfo = ty.NamedTuple('IntersectionInfo', [
    ('data', ty.Optional[ty.Union['Point', 'Segment']]),
    ('type', IntersectionType),
])


NOT_INTERSECTED = IntersectionInfo(None, IntersectionType.NONE)  # type: ty_ext.Final[IntersectionInfo]
"""The constant for cases when the intersection does not exist"""


class SegmentsIntersection:
    """The class represents the intersection of two segments

    Parameters
    ----------
    segment1 : Segment
        The first segment object
    segment2 : Segment
        The second segment object
    intersect_info : IntersectionInfo
        The intersection info object

    """

    __slots__ = ('_segment1', '_segment2', '_intersect_info')

    def __init__(self, segment1: 'Segment', segment2: 'Segment',
                 intersect_info: IntersectionInfo) -> None:
        self._segment1 = segment1
        self._segment2 = segment2
        self._intersect_info = intersect_info

    def __repr__(self) -> str:
        return '{}({}, {}, {}, type={})'.format(
            type(self).__name__,
            self.segment1,
            self.segment2,
            self.intersect_point,
            self.intersect_type.name,
        )

    def __bool__(self) -> bool:
        return self.intersect_type != IntersectionType.NONE

    @property
    def segment1(self) -> 'Segment':
        """The first segment

        Returns
        -------
        segment : Segment
            The first segment that intersects the second segment
        """

        return self._segment1

    @property
    def segment2(self) -> 'Segment':
        """The second segment

        Returns
        -------
        segment : Segment
            The second segment that intersects the first segment
        """

        return self._segment2

    @property
    def intersect_info(self) -> IntersectionInfo:
        """Returns the information about intersection

        Returns
        -------
        info : IntersectionInfo
            Intersection info named tuple ``(data, type)`` where:
                - ``data`` is None or Point or Segment
                - ``type`` intersection type (IntersectionType) NONE/EXACT/OVERLAP/ALMOST
        """

        return self._intersect_info

    @property
    def intersect_type(self) -> IntersectionType:
        """Returns the type of intersection

        Returns
        -------
        type : IntersectionType
            Intersection type enum item (NONE/EXACT/OVERLAP/ALMOST)
        """

        return self._intersect_info.type

    @property
    def intersect_point(self) -> ty.Optional['Point']:
        """Returns the intersection point

        Returns
        -------
        point : Point, None
            The intersection point or None if the intersection does not exist

        Notes
        -----

        If the intersection type is OVERLAP or ALMOST will be returned
        ``intersect_segment.point(t=0.5)`` as intersection point.

        See Also
        --------
        intersect_segment
        """

        if not self:
            return None

        if self.intersect_type == IntersectionType.EXACT:
            return self._intersect_info.data
        else:
            return self._intersect_info.data.point(0.5)

    @property
    def intersect_segment(self) -> ty.Optional['Segment']:
        """Returns the segment if the segments are overlapped or almost intersected

        Returns
        -------
        segment : Segment, None
            Overlapping or connecting segment if the segments are overlapped or
            almost intersected. None if the intersection does not exist.

        Notes
        -----

        If the intersection type is EXACT a singular segment will be returned.

        See Also
        --------
        intersect_point
        """

        if not self:
            return None

        if self.intersect_type == IntersectionType.EXACT:
            return curve._base.Segment(self._intersect_info.data, self._intersect_info.data)
        else:
            return self._intersect_info.data


class IntersectionMethodBase(abc.ABC):
    """The base class for all intersection methods
    """

    def __call__(self,
                 obj1: ty.Union['Segment', 'Curve'],
                 obj2: ty.Union['Segment', 'Curve']) -> ty.Union[SegmentsIntersection, ty.List[SegmentsIntersection]]:
        valid_obj_types = (curve._base.Segment, curve._base.Curve)
        if not isinstance(obj1, valid_obj_types) or not isinstance(obj2, valid_obj_types):
            raise TypeError('"obj1" and "obj2" arguments must be \'Segment\' or \'Curve\'')

        if obj1.ndim != obj2.ndim:
            raise ValueError('The dimension of both objects must be equal.')

        if isinstance(obj1, curve._base.Segment) and isinstance(obj2, curve._base.Segment):
            intersect_info = self._intersect_segments(obj1, obj2)
            return SegmentsIntersection(
                segment1=obj1,
                segment2=obj2,
                intersect_info=intersect_info,
            )
        elif isinstance(obj1, curve._base.Curve) and isinstance(obj2, curve._base.Curve):
            return self._intersect_curves(obj1, obj2)
        else:
            # Intersections between the curve and the segment
            obj1_is_segment = isinstance(obj1, curve._base.Segment)
            obj2_is_segment = isinstance(obj2, curve._base.Segment)

            curve1 = ty.cast(curve._base.Segment, obj1).to_curve() if obj1_is_segment else obj1
            curve2 = ty.cast(curve._base.Segment, obj2).to_curve() if obj2_is_segment else obj2

            intersections = self._intersect_curves(curve1, curve2)

            for i, intersection in enumerate(intersections):
                intersections[i] = SegmentsIntersection(
                    segment1=ty.cast(curve._base.Segment, obj1) if obj1_is_segment else intersection.segment1,
                    segment2=ty.cast(curve._base.Segment, obj2) if obj2_is_segment else intersection.segment2,
                    intersect_info=intersection.intersect_info,
                )

            return intersections

    @abc.abstractmethod
    def _intersect_segments(self, segment1: 'Segment', segment2: 'Segment') -> IntersectionInfo:
        pass

    @abc.abstractmethod
    def _intersect_curves(self, curve1: 'Curve', curve2: 'Curve') -> ty.List[SegmentsIntersection]:
        pass


def intersect_methods() -> ty.List[str]:
    """Returns the list of available intersect methods

    Returns
    -------
    methods : List[str]
        The list of available intersect methods

    See Also
    --------
    get_intersect_method
    register_intersect_method

    """

    return list(_intersect_methods.keys())


def get_intersect_method(method: str, **params) -> 'IntersectionMethodBase':
    """Returns the intersection method callable for the given method name

    Parameters
    ----------
    method : str
        Intersection method name
    params : mapping
        The method parameters

    Returns
    -------
    intersect : IntersectionMethodBase
        Intersection method class

    See Also
    --------
    intersect_methods
    register_intersect_method

    Raises
    ------
    NameError : If intersect method is unknown

    """

    if method not in _intersect_methods:
        raise NameError(
            'Unknown method "{}". The following methods are available: {}'.format(
                method, intersect_methods()))

    return _intersect_methods[method](**params)


def default_intersect_method() -> str:
    """Returns default intersect method

    Returns
    -------
    method : str
        Default intersect method
    """

    return _default_intersect_method


def set_default_intersect_method(method: str) -> None:
    """Sets the given intersect method as default

    Parameters
    ----------
    method : str
        Method name

    See Also
    --------
    default_intersect_method
    register_intersect_method
    """

    global _default_intersect_method

    if method not in _intersect_methods:
        raise NameError(
            'Unknown method "{}". The following methods are available: {}'.format(
                method, intersect_methods()))

    _default_intersect_method = method


def register_intersect_method(method: str, default: bool = False):
    """Decorator for registering segment intersection methods

    The decorator can be used for registering new intersection methods.

    The intersection method should be callable and implement `IntersectionMethod` protocol.

    Parameters
    ----------
    method : str
        Method name
    default : bool
        Makes registered method as default

    See Also
    --------
    IntersectionMethod
    intersect_methods
    get_intersect_method

    """

    def decorator(cls: ty.Type[IntersectionMethodBase]):
        if method in _intersect_methods:
            raise ValueError('"{}" intersect method already registered for {}'.format(
                method, _intersect_methods[method]))
        if not issubclass(cls, IntersectionMethodBase):
            raise TypeError("{} is not a subclass of 'IntersectionMethodBase'".format(cls))
        _intersect_methods[method] = cls

        if default:
            set_default_intersect_method(method)

    return decorator


@register_intersect_method(method='exact', default=True)
class ExactIntersectionMethod(IntersectionMethodBase):
    """The method to determine the exact segments and curves intersection

    We should solve the linear system of the following equations:
        x1 + t1 * (x2 - x1) = x3 + t2 * (x4 - x3)
        y1 + t1 * (y2 - y1) = y3 + t2 * (y4 - y3)
        ...
        n1 + t1 * (n2 - n3) = n3 + t2 * (n4 - n3)

    The solution of this system is t1 and t2 parameter values.
    If t1 and t2 in the range [0, 1], the segments are intersect.

    If the coefficient matrix is non-symmetric (for n-dim > 2),
    it requires a solver for over-determined system.

    Parameters
    ----------
    feps : float
        Floating point epsilon. F_EPS by default

    """

    def __init__(self, feps: float = F_EPS):
        self._feps = feps

    def _intersect_segments(self, segment1: 'Segment', segment2: 'Segment') -> IntersectionInfo:
        # Firstly, we should check all corner cases (overlap, parallel, not coplanar, singular...).
        if segment1.collinear(segment2):
            # We return overlap segment because we do not know exactly what point needed in this case.
            overlap_segment = segment1.overlap(segment2)

            if overlap_segment is None:
                return NOT_INTERSECTED
            return IntersectionType.OVERLAP(overlap_segment)

        if segment1.parallel(segment2):
            return NOT_INTERSECTED

        if not segment1.coplanar(segment2):
            return NOT_INTERSECTED

        if segment1.singular or segment2.singular:
            return NOT_INTERSECTED

        # After checking all corner cases we are sure that
        # two segments (or lines) should intersected.

        a = np.stack((segment1.direction.data,
                      -segment2.direction.data), axis=1)
        b = (segment2.p1 - segment1.p1).data

        if segment1.ndim == 2:
            try:
                t = np.linalg.solve(a, b)
            except np.linalg.LinAlgError as err:
                warnings.warn(
                    'Cannot solve system of equations: {}'.format(err), IntersectionWarning)
                return NOT_INTERSECTED
        else:
            t, residuals, *_ = np.linalg.lstsq(a, b, rcond=None)

            if residuals.size > 0 and residuals[0] > self._feps:
                warnings.warn(
                    'The "lstsq" residuals are {} > {}. Computation result might be wrong.'.format(
                        residuals, self._feps), IntersectionWarning)

        if np.all(((t > 0) | np.isclose(t, 0)) &
                  ((t < 1) | np.isclose(t, 1))):
            intersect_point1 = segment1.point(t[0])
            intersect_point2 = segment2.point(t[1])

            if intersect_point1 != intersect_point2:
                distance = intersect_point1.distance(intersect_point2)

                if distance > self._feps:
                    warnings.warn(
                        'Incorrect solution. The points for "t1" and "t2" are different (distance: {}).'.format(
                            distance), IntersectionWarning)
                    return NOT_INTERSECTED

            return IntersectionType.EXACT(intersect_point1)

        return NOT_INTERSECTED

    def _intersect_curves(self, curve1: 'Curve', curve2: 'Curve') -> ty.List[SegmentsIntersection]:
        if curve1.size == 0 or curve2.size == 0:
            return []

        s1, s2 = self._find_segments_bbox_intersection(curve1, curve2)

        if s1.size == 0:
            return []

        intersections = []

        for segment1, segment2 in zip(curve1.segments[s1], curve2.segments[s2]):
            intersect_info = self._intersect_segments(segment1, segment2)
            if intersect_info.type != IntersectionType.NONE:
                intersections.append(SegmentsIntersection(
                    segment1=segment1,
                    segment2=segment2,
                    intersect_info=intersect_info,
                ))

        return intersections

    @staticmethod
    def _find_segments_bbox_intersection(curve1: 'Curve', curve2: 'Curve') \
            -> ty.Tuple[np.ndarray, np.ndarray]:
        """Finds intersections between axis-aligned bounding boxes (AABB) of curves segments

        `Curve` 1 and `Curve` 2 can be different objects or the same objects (self intersection).
        """

        self_intersect = curve2 is curve1

        if curve1.size == 0 or curve2.size == 0:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            )

        # Get beginning and ending points of segments
        p11 = curve1.data[:-1, :]
        p12 = curve1.data[1:, :]

        curve1_pmin = np.minimum(p11, p12)
        curve1_pmax = np.maximum(p11, p12)

        if self_intersect:
            curve2_pmin = curve1_pmin
            curve2_pmax = curve1_pmax
        else:
            p21 = curve2.data[:-1, :]
            p22 = curve2.data[1:, :]

            curve2_pmin = np.minimum(p21, p22)
            curve2_pmax = np.maximum(p21, p22)

        # Find overlapping between all curve1 segment bboxes and curve2 segment
        # bboxes by all coordinates using vectorization
        curve1_pmin_tr = curve1_pmin[np.newaxis].transpose(2, 1, 0)
        curve1_pmax_tr = curve1_pmax[np.newaxis].transpose(2, 1, 0)

        curve2_pmin_tr = curve2_pmin[np.newaxis].transpose(2, 0, 1)
        curve2_pmax_tr = curve2_pmax[np.newaxis].transpose(2, 0, 1)

        is_intersect = (
                ((curve1_pmin_tr < curve2_pmax_tr) | (np.isclose(curve1_pmin_tr, curve2_pmax_tr))) &
                ((curve1_pmax_tr > curve2_pmin_tr) | (np.isclose(curve1_pmax_tr, curve2_pmin_tr)))
        ).all(axis=0)

        if self_intersect:
            # Removing duplicate combinations of segments when self intersection.
            i, j = np.tril_indices(curve1.size - 1, k=0)
            is_intersect[i, j] = False

        s1, s2 = np.nonzero(is_intersect)

        if self_intersect:
            # Removing coincident and adjacent segments
            adjacent = np.abs(s1 - s2) < 2
            s1 = s1[~adjacent]
            s2 = s2[~adjacent]

        return s1, s2


def intersect(obj1: ty.Union['Segment', 'Curve'],
              obj2: ty.Union['Segment', 'Curve'],
              method: ty.Optional[str] = None, **params: ty.Any) -> \
        ty.Union[SegmentsIntersection, ty.List[SegmentsIntersection]]:
    """Finds the intersection between n-dimensional segments and/or curves

    The function finds the intersection of two n-dimensional segments
    using given intersection method.

    Parameters
    ----------
    obj1 : Segment, Curve
        The first segment or curve object
    obj2 : Segment, Curve
        The second segment or curve object
    method : str, None
        The method to determine intersection. By default the following methods are available:
            - ``exact`` -- (default) the exact intersection solving the system of equations
            - ``almost`` -- the almost intersection using the shortest connecting segment.
              This is usually actual for dimension >= 3.

            The default method is ``exact``.
    params : mapping
        The intersection method parameters

    Returns
    -------
    res : Union[SegmentsIntersection, List[SegmentsIntersection]]
        The intersection result

    Raises
    ------
    ValueError : dimensions of the segments are different
    ValueError : Invalid input data or parameters

    """

    if method is None:
        method = _default_intersect_method

    if obj1.ndim != obj2.ndim:
        raise ValueError('The dimension of both objects must be equal.')

    intersect_method = get_intersect_method(method, **params)

    try:
        return intersect_method(obj1, obj2)
    except Exception as err:
        raise IntersectionError("'{}': finding intersection has failed: {}".format(
            method, err)) from err
