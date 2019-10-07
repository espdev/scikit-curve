# -*- coding: utf-8 -*-

"""
The module provides some geometric algorithms over n-dimensional segments and curves

"""

import warnings
import collections.abc as abc
import typing as ty

import numpy as np

from curve._numeric import F_EPS

if ty.TYPE_CHECKING:
    from curve._base import Point, Segment


class GeometryAlgorithmsWarning(UserWarning):
    """All warnings for geometric algorithms
    """


def segment_point(segment: 'Segment',
                  t: ty.Union[float, ty.Sequence[float], np.ndarray]) \
        -> ty.Union['Point', ty.List['Point']]:
    """Returns the point(s) on the segment for given "t"-parameter value or list of values

    The parametric line equation:

    .. math::

        P(t) = P_1 + t (P_2 - P_1)

    Parameters
    ----------
    segment : Segment
        The segment object
    t : float
        The parameter value in the range [0, 1] to get point on the segment

    Returns
    -------
    point : Point, List[Points]
        The point(s) on the segment for given "t"

    """

    if isinstance(t, (abc.Sequence, np.ndarray)):
        t = np.asarray(t)
        if t.ndim > 1:
            raise ValueError('"t" must be a sequence or 1-d numpy array.')

        dt = segment.direction.data * t[np.newaxis].T
        points_data = segment.p1.data + dt

        return [Point(pdata) for pdata in points_data]
    else:
        return segment.p1 + segment.direction * t


def segment_t(segment: 'Segment', point: ty.Union['Point', ty.Sequence['Point']],
              tol: ty.Optional[float] = None) -> ty.Union[float, np.ndarray]:
    """Returns "t"-parameter value(s) for given point(s) that collinear with the segment

    Parameters
    ----------
    segment : Segment
        The segment object
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

    from curve._base import Point

    if isinstance(point, Point):
        if not segment.collinear(point, tol=tol):
            warnings.warn(
                "Given point '{}' is not collinear with the segment '{}'".format(point, segment),
                GeometryAlgorithmsWarning)
            return np.nan
        b = point.data - segment.p1.data
        is_collinear = np.asarray([])
    else:
        is_collinear = np.array([segment.collinear(p, tol=tol) for p in point], dtype=np.bool_)
        b = np.stack([p.data - segment.p1.data for p in point], axis=1)

    a = segment.direction.data[np.newaxis].T

    t, residuals, *_ = np.linalg.lstsq(a, b, rcond=None)

    if residuals.size > 0 and residuals[0] > F_EPS:
        warnings.warn(
            'The "lstsq" residuals are {}. "t" value(s) might be wrong.'.format(residuals),
            GeometryAlgorithmsWarning)

    t = t.squeeze()

    if is_collinear.size == 0:
        return float(t)
    else:
        if not np.all(is_collinear):
            warnings.warn(
                "Some given points are not collinear with the segment", GeometryAlgorithmsWarning)
            t[~is_collinear] = np.nan
        if t.size == 1:
            t = np.array(t, ndmin=1)
        return t


def segments_angle(segment1: 'Segment', segment2: 'Segment',
                   ndigits: ty.Optional[int] = None) -> float:
    """Returns the angle between two segments

    Parameters
    ----------
    segment1 : Segment
        The first segment
    segment2 : Segment
        The second segment
    ndigits : int, None
        The number of significant digits (precision)

    Returns
    -------
    phi : float
        The angle in radians between two segments or NaN if segment(s) are singular

    """

    u1 = segment1.direction
    u2 = segment2.direction

    denominator = u1.norm() * u2.norm()

    if np.isclose(denominator, 0.0):
        warnings.warn(
            'Cannot compute angle between segments. One or both segments degenerate into a point.',
            GeometryAlgorithmsWarning)
        return np.nan

    cos_phi = (u1 @ u2) / denominator

    if ndigits is not None:
        cos_phi = round(cos_phi, ndigits=ndigits)

    # We need to consider floating point errors
    cos_phi = 1.0 if cos_phi > 1.0 else cos_phi
    cos_phi = -1.0 if cos_phi < -1.0 else cos_phi

    return np.arccos(cos_phi)


def parallel_segments(segment1: 'Segment', segment2: 'Segment', tol: float = F_EPS) -> bool:
    """Returns True if two segments are parallel

    Parameters
    ----------
    segment1 : Segment
        The first segment
    segment2 : Segment
        The second segment
    tol : float
        Epsilon. It is a small float number. By default float64 eps

    Returns
    -------
    flag : bool
        True if the segments are parallel

    """

    u = segment1.direction
    v = segment2.direction

    a = u @ u
    b = u @ v
    c = v @ v

    d = a * c - b * b

    return d < tol


def collinear_points(points: ty.Union[ty.List['Point'], np.ndarray],
                     tol: ty.Optional[float] = None) -> bool:
    """Returns True if the three or more distinct points are collinear (all points along a line)

    Notes
    -----

    In n-dimensional space, a set of three or more distinct points are collinear
    if and only if, the matrix of the coordinates of these vectors is of rank 1 or less.

    Parameters
    ----------
    points : List[Point], np.ndarray
        The list of points or MxN array of point coords
    tol : float, None
        Threshold below which SVD values are considered zero

    Returns
    -------
    flag : bool
        True if the points are collinear

    """

    if isinstance(points, np.ndarray):
        points_data = points.T
    else:
        points_data = np.vstack([point.data for point in points]).T

    m = np.unique(points_data, axis=1)

    if m.shape[1] < 3:
        return True

    return np.linalg.matrix_rank(m, tol=tol) <= 1


def coplanar_points(points: ty.Union[ty.List['Point'], np.ndarray],
                    tol: ty.Optional[float] = None) -> bool:
    """Returns True if the poins are coplanar

    Parameters
    ----------
    points : List[Point], np.ndarray
        The list of points or MxN array of point coords
    tol : float, None
        Threshold below which SVD values are considered zero

    Returns
    -------
    flag : bool
        True if the points are coplanar

    """

    if isinstance(points, np.ndarray):
        points_data = np.array(points)
    else:
        points_data = np.vstack([point.data for point in points])

    if points_data.shape[1] < 3:
        # In dimension < 3 the points are coplanar
        return True

    # Relative differences
    points_data -= points_data[-1, :]
    return np.linalg.matrix_rank(points_data[:-1, :], tol=tol) <= 2


def overlap_segments(segment1: 'Segment', segment2: 'Segment',
                     tol: ty.Optional[float] = None) -> ty.Optional['Segment']:
    """Returns overlap segment between two segments if it exists

    Parameters
    ----------
    segment1 : Segment
        The first segment
    segment2 : Segment
        The second segment
    tol : float, None
        Threshold below which SVD values are considered zero

    Returns
    -------
    segment : Segment, None
        Overlap segment if it is exist.

    """

    from curve._base import Point, Segment

    if not segment1.collinear(segment2, tol=tol):
        return None

    p11_data = segment1.p1.data
    p12_data = segment1.p2.data
    p21_data = segment2.p1.data
    p22_data = segment2.p2.data

    data_minmax = np.minimum(
        np.maximum(p11_data, p12_data),
        np.maximum(p21_data, p22_data),
    )

    data_maxmin = np.maximum(
        np.minimum(p11_data, p12_data),
        np.minimum(p21_data, p22_data),
    )

    if np.any(data_maxmin > data_minmax):
        return None

    return Segment(Point(data_maxmin), Point(data_minmax))


def distance_point_to_segment(point: 'Point', segment: 'Segment') -> float:
    """Computes the shortest distance from the point to the segment

    Parameters
    ----------
    point : Point
        The point object
    segment : Segment
        The segment object

    Returns
    -------
    dist : float
        The shortest distance from the point to the segment

    """

    segment_direction = segment.direction
    to_point_direction = point - segment.p1

    c1 = to_point_direction @ segment_direction

    if c1 < 0 or np.isclose(c1, 0):
        return point.distance(segment.p1)

    c2 = segment_direction @ segment_direction

    if c2 < c1 or np.isclose(c2, c1):
        return point.distance(segment.p2)

    b = c1 / c2
    pp = segment.p1 + segment_direction * b

    return point.distance(pp)
