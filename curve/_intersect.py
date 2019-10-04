# -*- coding: utf-8 -*-

"""
Curves intersection in n-dimensional Euclidean space

The module provides routines for determining curves intersections in n-dimensional Euclidean space.

The code is inspired by "Fast and Robust Curve Intersections" by Douglas Schwarz
https://www.mathworks.com/matlabcentral/fileexchange/11837-fast-and-robust-curve-intersections

"""

import typing as ty
import itertools
import warnings

import numpy as np

if ty.TYPE_CHECKING:
    from curve._base import Point, Segment, CurveSegment, Curve


NotIntersected = None

F_EPS = np.finfo(np.float64).eps


SegmentsBBoxIntersectionResult = ty.NamedTuple('SegmentsBBoxIntersectionResult', [
    ('segments1', np.ndarray),
    ('segments2', np.ndarray),
])


SolveSegmentsIntersectionResult = ty.NamedTuple('SolveSegmentsIntersectionResult', [
    ('solution', np.ndarray),
    ('overlap', np.ndarray),
])


DetermineSegmentsIntersectionResult = ty.NamedTuple('DetermineSegmentsIntersectionResult', [
    ('segments1', np.ndarray),
    ('segments2', np.ndarray),
    ('intersect_points', np.ndarray),
])


def intersect_segments(segment1: 'Segment', segment2: 'Segment') \
        -> ty.Union[NotIntersected, 'Point', 'Segment']:
    """Finds intersection of two n-dimensional segments

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

    """

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

    if np.any((t < 0) | (t > 1)):
        return NotIntersected

    return segment1.point(t[0])


def _find_segments_bbox_intersection(curve1: 'Curve', curve2: 'Curve') -> SegmentsBBoxIntersectionResult:
    """Finds intersections between axis-aligned bounding boxes (AABB) of curves segments

    `Curve` 1 and `Curve` 2 can be different objects or the same objects (self intersection).

    """

    self_intersect = curve2 is curve1

    if curve1.size == 0 or curve2.size == 0:
        return SegmentsBBoxIntersectionResult(
            segments1=np.array([], dtype=np.int64),
            segments2=np.array([], dtype=np.int64),
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
        remove = np.flatnonzero(np.abs(s1 - s2) < 2)

        s1 = np.delete(s1, remove)
        s2 = np.delete(s2, remove)

    return SegmentsBBoxIntersectionResult(
        segments1=s1,
        segments2=s2,
    )


def _test_skewness(segment1: 'CurveSegment', segment2: 'CurveSegment', eps: float = F_EPS) -> bool:
    a, b = segment1.p1, segment1.p2
    c, d = segment2.p1, segment2.p2

    m = np.array([
        a - b,
        b - c,
        c - d,
    ])

    return (1 / 6 * np.abs(np.linalg.det(m))) > eps


def _exclude_skew_segments(seg1, seg2, curve1, curve2):
    segments1 = curve1.segments[seg1]
    segments2 = curve2.segments[seg2]

    skew = np.array([_test_skewness(s1, s2) for s1, s2 in zip(segments1, segments2)], dtype=np.bool_)

    return seg1[~skew], seg2[~skew]


def _solve_segments_intersection(
        bbox_intersect: SegmentsBBoxIntersectionResult,
        curve1: 'Curve', curve2: 'Curve') -> SolveSegmentsIntersectionResult:
    """Solves the linear system of equations to determine segments intersections

    So, the equations for two n-dimensional segments intersection problem are

       (x1[1] - x1[0]) * t = x0 - x1[0]
       (x2[1] - x2[0]) * u = x0 - x2[0]
       (y1[1] - y1[0]) * t = y0 - y1[0]
       (y2[1] - y2[0]) * u = y0 - y2[0]
                     ...
       (n1[1] - n1[0]) * t = n0 - n1[0]
       (n2[1] - n2[0]) * u = n0 - n2[0]

    Rearranging and writing in matrix form,
                        A                            t         b
     [x1[1]-x1[0]       0       -1   0 ...  0        [ t      [-x1[0]
           0       x2[1]-x2[0]  -1   0 ...  0    *     u   =   -x2[0]
      y1[1]-y1[0]       0        0  -1 ...  0         x0       -y1[0]
           0       y2[1]-y2[0]   0  -1 ...  0         y0       -y2[0]
                       ...                           ...        ...
      n1[1]-n1[0]       0        0   0 ... -1         n0       -n1[0]
           0       n2[1]-n2[0]   0   0 ... -1]        n0]      -n2[0]]

    where:
      A is MxN matrix:
        M is the number of rows: curve.ndim * 2
        N is the number of columns: M - curve.ndim + 2

    Let's call that A*w = b.  We can solve for w
    using linalg numpy.linalg.lstsq or numpy.linalg.solve for square 2-D case.

    """

    self_intersect = curve2 is curve1

    seg1 = bbox_intersect.segments1
    seg2 = bbox_intersect.segments2

    ndim = curve1.ndim

    if ndim == 3:
        # Exclude all segments that fail skewness test
        seg1, seg2 = _exclude_skew_segments(seg1, seg2, curve1, curve2)

    data1 = curve1.data
    data2 = curve2.data

    data1_diff = np.diff(data1, axis=0)

    if self_intersect:
        data2_diff = data1_diff
    else:
        data2_diff = np.diff(data2, axis=0)

    is_nan = np.isnan(np.sum(data1_diff[seg1, :] + data2_diff[seg2, :], axis=1))

    if np.any(is_nan):
        seg1 = seg1[~is_nan]
        seg2 = seg2[~is_nan]

    n_seg = seg1.size
    n_equ = ndim * 2
    n_unknown = n_equ - ndim + 2

    coeffs = np.zeros((n_equ, n_unknown))
    solution = np.zeros((n_unknown, n_seg))

    # Initialize constant values "-1" for "coeffs" matrix
    coeffs[np.r_[:n_equ], np.r_[2:n_unknown].repeat(2)] = -1

    # Define "values" vectors stack for given curves
    values = np.zeros((n_equ, n_seg))

    for i, vals1, vals2 in zip(itertools.count(step=2), curve1.values(), curve2.values()):
        values[i, :] = -vals1[seg1]
        values[i+1, :] = -vals2[seg2]

    def solve_2d(a, b):
        ovrlp = False

        try:
            x = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            # It seems "coeffs" is a singular matrix
            # We have the parallel, coincident or overlap segments
            x = -1.0 * np.ones_like(b)

            m = np.vstack((
                data1_diff[seg1[i], :],
                data2[seg2[i], :] - data1[seg1[i], :]
            ))

            # Determine if these segments overlap or are just parallel
            try:
                # Reciprocal condition number
                rcond = 1.0 / np.linalg.cond(m, 1)
                ovrlp = rcond < F_EPS
            except np.linalg.LinAlgError:
                pass

        return x, ovrlp

    def solve_ge3d(a, b):
        # FIXME: implement stable and robust intersection algorithm for ndim >= 3
        raise NotImplementedError('Unstable for ndim={}'.format(ndim))
        # eps = 1e-8
        # x, istop, itn, normr, normar, norma, conda, normx = spla.lsmr(a, b)

        # if normr > eps:
        #     x[:2] = -1
        #     ovrlp = False
        # else:
        #     t, u = x[:2]
        #     norm = normr < eps and normar < eps
        #
        #     if t < 0 and u < 0 and norm:
        #         ovrlp = True
        #     elif t > 1 and u > 1 and norm:
        #         ovrlp = True
        #     else:
        #         ovrlp = False

        # tu_out_range = (t < 0 or t > 1) or (u < 0 or t > 1)
        # ovrlp = tu_out_range and (normr < eps and normar < eps)

        # return x, ovrlp

    if ndim == 2:
        solve = solve_2d
    else:
        # We have non-symmetric "coeffs" matrix for n-dim > 2,
        # it requires a solver for over-determined system
        solve = solve_ge3d

    overlap = np.zeros(n_seg, dtype=np.bool)

    for i in range(n_seg):
        coeffs[np.r_[:n_equ:2], 0] = data1_diff[seg1[i], :]
        coeffs[np.r_[1:n_equ:2], 1] = data2_diff[seg2[i], :]

        solution[:, i], overlap[i] = solve(coeffs, values[:, i])

    return SolveSegmentsIntersectionResult(
        solution=solution,
        overlap=overlap,
    )


def _determine_segments_intersection(
        bbox_intersect: SegmentsBBoxIntersectionResult,
        solve_result: SolveSegmentsIntersectionResult,
        curve1: 'Curve', curve2: 'Curve') -> DetermineSegmentsIntersectionResult:
    """Determines segments intersections and intersection points

    Find where t and u are between 0 and 1 and return the
    corresponding intersection points values. Anomalous segment pairs can be
    segment pairs that are colinear (overlap) or the result of segments
    that are degenerate (end points the same). The algorithm will return
    an intersection point that is at the center of the overlapping region.
    Because of the finite precision of floating point arithmetic it is
    difficult to predict when two line segments will be considered to
    overlap exactly or even intersect at an end point.

    """

    seg1 = bbox_intersect.segments1
    seg2 = bbox_intersect.segments2

    solution = solve_result.solution
    overlap = solve_result.overlap

    t = solution[0, :]
    u = solution[1, :]

    # If t and u parameters in the range [0, 1] we have the intersection point on segments
    tu_in_range = (
        ((t > 0.) | (np.isclose(t, 0.))) &
        ((t < 1.) | (np.isclose(t, 1.))) &
        ((u > 0.) | (np.isclose(u, 0.))) &
        ((u < 1.) | (np.isclose(u, 1.)))
    )

    intersect_points = solution[2:, :].T

    if np.any(overlap):
        # Set intersection point to middle of overlapping region
        seg1_overlap = seg1[overlap]
        seg2_overlap = seg2[overlap]

        data1 = curve1.data
        data2 = curve2.data

        data1_p1 = data1[seg1_overlap]
        data1_p2 = data1[seg1_overlap + 1]

        data2_p1 = data2[seg2_overlap]
        data2_p2 = data2[seg2_overlap + 1]

        data_minmax = np.minimum(
            np.maximum(data1_p1, data1_p2),
            np.maximum(data2_p1, data2_p2),
        )

        data_maxmin = np.maximum(
            np.minimum(data1_p1, data1_p2),
            np.minimum(data2_p1, data2_p2),
        )

        # The middle points of overlapping regions
        intersect_points[overlap, :] = (data_minmax + data_maxmin) / 2.0

        is_intersect = tu_in_range | overlap
    else:
        is_intersect = tu_in_range

    return DetermineSegmentsIntersectionResult(
        segments1=seg1[is_intersect],
        segments2=seg2[is_intersect],
        intersect_points=intersect_points[is_intersect, :],
    )


class SegmentsIntersection:
    """The data class represents the intersection of two segments
    """

    def __init__(self,
                 segment1: 'CurveSegment',
                 segment2: 'CurveSegment',
                 intersect_point: 'Point') -> None:
        self._segment1 = segment1
        self._segment2 = segment2
        self._intersect_point = intersect_point

    def __repr__(self) -> str:
        return '{}({}, {}, {})'.format(
            type(self).__name__,
            self.segment1,
            self.segment2,
            self.intersect_point,
        )

    @property
    def segment1(self) -> 'CurveSegment':
        """The first segment

        Returns
        -------
        segment : CurveSegment
            The first segment that intersects the second segment
        """

        return self._segment1

    @property
    def segment2(self) -> 'CurveSegment':
        """The second segment

        Returns
        -------
        segment : CurveSegment
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
        """

        return self._intersect_point

    def swap_segments(self) -> 'SegmentsIntersection':
        """Returns new intersection object with swapped segments

        Returns
        -------
        intersection: SegmentsIntersection
            `SegmentsIntersection` object with swapped segments

        """

        return SegmentsIntersection(
            segment1=self.segment2,
            segment2=self.segment1,
            intersect_point=self.intersect_point,
        )


def intersect(curve1: 'Curve', curve2: 'Curve') \
        -> ty.Optional[DetermineSegmentsIntersectionResult]:
    """Finds the intersections between two n-dimensional curves or self intersections

    Parameters
    ----------
    curve1 : Curve
        The first curve object
    curve2 : Curve
        The second curve object. If it is equal to curve1,
        self intersection will be determined.

    Returns
    -------
    intersections : DetermineSegmentsIntersectionResult, None
        Intersections info

    """

    if curve1.ndim != curve2.ndim:
        raise ValueError('The dimension of both curves must be equal.')

    if curve1.size == 0 or curve2.size == 0:
        return None

    bbox_intersect = _find_segments_bbox_intersection(curve1, curve2)
    solve_result = _solve_segments_intersection(bbox_intersect, curve1, curve2)
    intersections = _determine_segments_intersection(bbox_intersect, solve_result, curve1, curve2)

    return intersections
