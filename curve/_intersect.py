# -*- coding: utf-8 -*-

"""
The module provides routines for determining segments and curves intersections
in n-dimensional Euclidean space.

"""

import collections.abc as abc
import typing as ty
import warnings
import enum

import numpy as np

from curve._numeric import F_EPS

if ty.TYPE_CHECKING:
    from curve._base import Point, Segment, Curve


_INTERSECT_METHODS = {}
_DEFAULT_INTERSECT_METHOD = None  # type: ty.Optional[str]

NotIntersected = None


class IntersectionWarning(UserWarning):
    """All intersection warnings
    """


class IntersectionError(Exception):
    """All intersection errors
    """


class IntersectionType(enum.Enum):
    """The types of intersection cases
    """

    EXACT = 0
    OVERLAP = 1
    ALMOST = 2

    def __call__(self, intersect_data: ty.Union['Point', 'Segment']) -> 'IntersectionInfo':
        from curve._base import Point, Segment

        if ((isinstance(intersect_data, Point) and self != IntersectionType.EXACT) or
                (isinstance(intersect_data, Segment) and self == IntersectionType.EXACT)):
            raise ValueError('Invalid "intersect_data" {} for type {}'.format(
                type(intersect_data), self))

        return IntersectionInfo(intersect_data, self)


IntersectionInfo = ty.NamedTuple('IntersectionInfo', [
    ('data', ty.Union['Point', 'Segment']),
    ('type', IntersectionType),
])


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
        """Returns the type of intersection info

        Returns
        -------
        info : IntersectionInfo
            Intersection info named tuple
        """

        return self._intersect_info

    @property
    def intersect_type(self) -> IntersectionType:
        """Returns the type of intersection

        Returns
        -------
        type : IntersectionType
            Intersection type enum item
        """

        return self._intersect_info.type

    @property
    def intersect_point(self) -> 'Point':
        """Returns the intersection point

        Returns
        -------
        point : Point
            The intersection point

        Notes
        -----

        If the segments are overlapped will be returned
        ``overlap_segment.point(t=0.5)`` as intersection point.
        """

        if self.intersect_type == IntersectionType.EXACT:
            return self._intersect_info.data
        else:
            return self._intersect_info.data.point(0.5)

    @property
    def intersect_segment(self) -> 'Segment':
        """Returns the segment if the segments are overlapped or almost intersected

        Returns
        -------
        segment : Segment
            Overlapping or connecting segment if the segments are overlapped or
            almost intersected. If the intersection is exact a singular segment will be returned.
        """

        from curve._base import Segment

        if self.intersect_type == IntersectionType.EXACT:
            return Segment(self._intersect_info.data, self._intersect_info.data)
        else:
            return self._intersect_info.data


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

    return list(_INTERSECT_METHODS.keys())


def get_intersect_method(method: str) -> abc.Callable:
    """Returns the intersection method callable for the given method name

    Parameters
    ----------
    method : str
        Intersection method name

    Returns
    -------
    intersect : Callable
        Intersection method callable

    See Also
    --------
    intersect_methods
    register_intersect_method

    Raises
    ------
    NameError : If intersect method is unknown

    """

    if method not in _INTERSECT_METHODS:
        raise NameError(
            'Unknown method "{}". The following methods are available: {}'.format(
                method, intersect_methods()))

    return _INTERSECT_METHODS[method]


def default_intersect_method() -> str:
    """Returns default intersect method

    Returns
    -------
    method : str
        Default intersect method
    """
    global _DEFAULT_INTERSECT_METHOD
    return _DEFAULT_INTERSECT_METHOD


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

    global _DEFAULT_INTERSECT_METHOD

    if method not in _INTERSECT_METHODS:
        raise NameError(
            'Unknown method "{}". The following methods are available: {}'.format(
                method, intersect_methods()))

    _DEFAULT_INTERSECT_METHOD = method


def register_intersect_method(method: str, default: bool = False):
    """Decorator for registering segment intersection methods

    Parameters
    ----------
    method : str
        Method name
    default : bool
        Makes given method as default

    See Also
    --------
    intersect_methods
    get_intersect_method

    """

    def decorator(method_callable):
        if method in _INTERSECT_METHODS:
            raise ValueError('"{}" intersect method already registered for {}'.format(
                method, _INTERSECT_METHODS[method]))
        _INTERSECT_METHODS[method] = method_callable

        if default:
            set_default_intersect_method(method)

    return decorator


@register_intersect_method('exact', default=True)
def exact_intersect(segment1: 'Segment', segment2: 'Segment') -> ty.Optional[IntersectionInfo]:
    """Determines the segments intersection exactly

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
    segment1 : Segment
        The first segment
    segment2 : Segment
        The second segment

    Returns
    -------
    intersect_info : IntersectionInfo, NotIntersected
        Intersection info or NotIntersected
    """

    a = np.stack((segment1.direction.data,
                  -segment2.direction.data), axis=1)
    b = (segment2.p1 - segment1.p1).data

    if segment1.ndim == 2:
        try:
            t = np.linalg.solve(a, b)
        except np.linalg.LinAlgError as err:
            warnings.warn(
                'Cannot solve system of equations: {}'.format(err), IntersectionWarning)
            return NotIntersected
    else:
        t, residuals, *_ = np.linalg.lstsq(a, b, rcond=None)

        if residuals.size > 0 and residuals[0] > F_EPS:
            warnings.warn(
                'The "lstsq" residuals are {} > {}. Computation result might be wrong.'.format(
                    residuals, F_EPS), IntersectionWarning)

    if np.all(((t > 0) | np.isclose(t, 0)) &
              ((t < 1) | np.isclose(t, 1))):
        intersect_point1 = segment1.point(t[0])
        intersect_point2 = segment2.point(t[1])

        if intersect_point1 != intersect_point2:
            distance = intersect_point1.distance(intersect_point2)

            if distance > F_EPS:
                warnings.warn(
                    'Incorrect solution. The points for "t1" and "t2" are different (distance: {}).'.format(
                        distance), IntersectionWarning)
                return NotIntersected

        return IntersectionType.EXACT(intersect_point1)

    return NotIntersected


@register_intersect_method('almost')
def almost_intersect(segment1: 'Segment', segment2: 'Segment',
                     almost_tol: float = 1e-5) -> ty.Optional[IntersectionInfo]:
    """Determines the almost intersection of two skewnes segments

    We should compute the shortest connecting segment between the segments in this case.
    We check the length of the shortest segment. If it is smaller a tolerance value we
    consider it as the intersection of the segments.

    Parameters
    ----------
    segment1 : Segment
        The first segment
    segment2 : Segment
        The second segment
    almost_tol : float
        The almost intersection tolerance value for. By default 1e-5.

    Returns
    -------
    intersect_info : IntersectionInfo, NotIntersected
        Intersection info or NotIntersected
    """

    shortest_segment = segment1.shortest_segment(segment2)

    if shortest_segment.seglen < almost_tol:
        return IntersectionType.ALMOST(shortest_segment)

    return NotIntersected


def intersect_segments(segment1: 'Segment', segment2: 'Segment',
                       method: ty.Optional[str] = None, **params) \
        -> ty.Union[NotIntersected, SegmentsIntersection]:
    """Finds exact intersection of two n-dimensional segments

    The function finds exact intersection of two n-dimensional segments
    using linear algebra routines.

    Parameters
    ----------
    segment1 : Segment
        The first segment
    segment2 : Segment
        The second segment
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
    res : NotIntersected, SegmentsIntersection
        The intersection result. It can be:
            - NotIntersected (None): No any intersection of the segments
            - SegmentsIntersection: The intersection of the segments

    Raises
    ------
    ValueError : dimensions of the segments are different
    ValueError : Invalid input data or parameters

    """

    global _DEFAULT_INTERSECT_METHOD

    if method is None:
        method = _DEFAULT_INTERSECT_METHOD

    if segment1.ndim != segment2.ndim:
        raise ValueError('The dimension of the segments must be equal.')

    # Firstly, we should check all corner cases (overlap, parallel, not coplanar, singular...).
    if segment1.collinear(segment2):
        # We return overlap segment because we do not know exactly what point needed in this case.
        overlap_segment = segment1.overlap(segment2)

        if overlap_segment is None:
            return NotIntersected

        return SegmentsIntersection(
            segment1=segment1,
            segment2=segment2,
            intersect_info=IntersectionType.OVERLAP(overlap_segment),
        )

    if segment1.parallel(segment2):
        return NotIntersected

    if method == 'exact' and not segment1.coplanar(segment2):
        return NotIntersected

    if segment1.singular or segment2.singular:
        return NotIntersected

    # After checking all corner cases we are sure that
    # two segments (or lines) should intersected.

    intersect_method = get_intersect_method(method)

    try:
        intersect_info = intersect_method(segment1, segment2, **params)
    except Exception as err:
        raise IntersectionError("'{}': finding intersection has failed: {}".format(
            method, err)) from err

    if intersect_info:
        return SegmentsIntersection(
            segment1=segment1,
            segment2=segment2,
            intersect_info=intersect_info,
        )

    return NotIntersected


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


def intersect_curves(curve1: 'Curve', curve2: 'Curve',
                     method: ty.Optional[str] = None, **params) -> ty.List[SegmentsIntersection]:
    """Finds the intersections between two n-dimensional curves or a curve self intersections

    Parameters
    ----------
    curve1 : Curve
        The first curve object
    curve2 : Curve
        The second curve object. If it is equal to curve1,
        self intersection will be determined.
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
    intersections : List[SegmentsIntersection]
        The list of intersections of curves segments

    Raises
    ------
    ValueError : dimensions of the curves are different

    """

    if curve1.ndim != curve2.ndim:
        raise ValueError('The dimension the curves must be equal.')

    if curve1.size == 0 or curve2.size == 0:
        return []

    s1, s2 = _find_segments_bbox_intersection(curve1, curve2)

    if s1.size == 0:
        return []

    intersections = []

    for segment1, segment2 in zip(curve1.segments[s1], curve2.segments[s2]):
        intersection = intersect_segments(segment1, segment2, method=method, **params)

        if intersection is NotIntersected:
            continue

        intersections.append(intersection)

    return intersections
