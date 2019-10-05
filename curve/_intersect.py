# -*- coding: utf-8 -*-

"""
Curves intersection in n-dimensional Euclidean space

The module provides routines for determining curves intersections in n-dimensional Euclidean space.

The code is inspired by "Fast and Robust Curve Intersections" by Douglas Schwarz
https://www.mathworks.com/matlabcentral/fileexchange/11837-fast-and-robust-curve-intersections

"""

import typing as ty
import warnings

import numpy as np

if ty.TYPE_CHECKING:
    from curve._base import Point, Segment, Curve


NotIntersected = None
F_EPS = np.finfo(np.float64).eps


def intersect_segments(segment1: 'Segment', segment2: 'Segment') \
        -> ty.Union[NotIntersected, 'Point', 'Segment']:
    """Finds exact intersection of two n-dimensional segments

    The function finds exact intersection of two n-dimensional segments
    using linear algebra routines.

    Parameters
    ----------
    segment1 : Segment
        The first segment
    segment2 : Segment
        The second segment

    Returns
    -------
    res : NotIntersected, Point, Segment
        The intersection result. It can be:
            - NotIntersected (None): No any intersection of the segments
            - Point: The intersection point of the segments
            - Segment: The overlap segment in the case of overlapping the collinear segments

    Raises
    ------
    ValueError : dimensions of the segments are different

    """

    if segment1.ndim != segment1.ndim:
        raise ValueError('The dimension of the segments must be equal.')

    # Firstly, we should check all corner cases (overlap, parallel, not coplanar).
    if segment1.collinear(segment2):
        # We return overlap segment because we do not know exactly what point the user needs.
        return segment1.overlap(segment2)

    if segment1.parallel(segment2) or not segment1.coplanar(segment2):
        # In these cases the segments will never intersect
        return NotIntersected

    # We should solve the linear system of the following equations:
    #   x1 + t1 * (x2 - x1) = x3 + t2 * (x4 - x3)
    #   y1 + t1 * (y2 - y1) = y3 + t2 * (y4 - y3)
    #                      ...
    #   n1 + t1 * (n2 - n3) = n3 + t2 * (n4 - n3)
    #
    # The solution of this system is t1 and t2 parameter values.
    # If t1 and t2 in the range [0, 1], the segments are intersect.
    #
    # If the coefficient matrix is non-symmetric (for n-dim > 2),
    # it requires a solver for over-determined system.

    a = np.stack((segment1.direction.data,
                  -segment2.direction.data), axis=1)
    b = (segment2.p1 - segment1.p1).data

    if segment1.ndim == 2:
        t = np.linalg.solve(a, b)
    else:
        t, residuals, *_ = np.linalg.lstsq(a, b, rcond=None)

        if residuals.size > 0 and residuals[0] > F_EPS:
            warnings.warn(
                'The "lstsq" residuals are {} > {}. Computation result might be wrong.'.format(
                    residuals, F_EPS),
                RuntimeWarning
            )

    if np.all(((t > 0) | np.isclose(t, 0)) &
              ((t < 1) | np.isclose(t, 1))):
        return segment1.point(t[0])

    return None


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


class SegmentsIntersection:
    """The class represents the intersection of two segments

    Parameters
    ----------
    segment1 : Segment
        The first segment object
    segment2 : Segment
        The second segment object
    intersection : Point, Segment
        The intersection object:
            - Point if the segments are not overlapped
            - Segment if the segments are overlapped
    """

    __slots__ = ('_segment1', '_segment2', '_intersection')

    def __init__(self,
                 segment1: 'Segment',
                 segment2: 'Segment',
                 intersection: ty.Union['Point', 'Segment']) -> None:
        self._segment1 = segment1
        self._segment2 = segment2
        self._intersection = intersection

    def __repr__(self) -> str:
        return '{}({}, {}, {}, overlap={})'.format(
            type(self).__name__,
            self.segment1,
            self.segment2,
            self.intersect_point,
            self.overlap,
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
        from curve._base import Point

        if isinstance(self._intersection, Point):
            return self._intersection
        else:
            return self.overlap_segment.point(0.5)

    @property
    def overlap_segment(self) -> ty.Optional['Segment']:
        """Returns overlap segment if the segments are overlapped

        Returns
        -------
        overlap : Segment, None
            Segment object if the segments are overlapped or None
        """
        from curve._base import Segment

        if isinstance(self._intersection, Segment):
            return self._intersection
        return None

    @property
    def overlap(self) -> bool:
        """Returns True if the segments are overlapped

        Returns
        -------
        flag : bool
            True if the segments overlapped
        """

        return self.overlap_segment is not None

    def swap_segments(self) -> 'SegmentsIntersection':
        """Returns new intersection object with swapped segments

        Returns
        -------
        intersection: SegmentsIntersection
            `SegmentsIntersection` object with swapped segments
        """

        return SegmentsIntersection(
            segment1=self._segment2,
            segment2=self._segment1,
            intersection=self._intersection,
        )


def intersect_curves(curve1: 'Curve', curve2: 'Curve') -> ty.List[SegmentsIntersection]:
    """Finds the intersections between two n-dimensional curves or a curve self intersections

    Parameters
    ----------
    curve1 : Curve
        The first curve object
    curve2 : Curve
        The second curve object. If it is equal to curve1,
        self intersection will be determined.

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
        intersection = segment1.intersect(segment2)

        if not intersection:
            continue

        intersections.append(SegmentsIntersection(
            segment1=segment1,
            segment2=segment2,
            intersection=intersection,
        ))

    return intersections
