# -*- coding: utf-8 -*-

"""
The module provides some geometric algorithms over n-dimensional segments and curves

"""

import warnings
import collections.abc as abc
import typing as ty

import numpy as np

import curve._base
from curve._numeric import F_EPS

if ty.TYPE_CHECKING:
    from curve._base import Point, Segment  # noqa


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

        return [curve._base.Point(pdata) for pdata in points_data]
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

    if isinstance(point, curve._base.Point):
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

    return curve._base.Segment(curve._base.Point(data_maxmin),
                               curve._base.Point(data_minmax))


def segment_to_point(segment: 'Segment', point: 'Point') -> float:
    """Computes the shortest segment from the segment to the point

    Parameters
    ----------
    segment : Segment
        The segment object
    point : Point
        The point object

    Returns
    -------
    t : float
        The t-parameter to determine the point in the segment

    """

    segment_direction = segment.direction
    to_point_direction = point - segment.p1

    c1 = to_point_direction @ segment_direction

    if c1 < 0 or np.isclose(c1, 0):
        return 0.0

    c2 = segment_direction @ segment_direction

    if c2 < c1 or np.isclose(c2, c1):
        return 1.0

    t = c1 / c2
    return t


def segment_to_segment(segment1: 'Segment', segment2: 'Segment', tol: float = F_EPS) -> ty.Tuple[float, float]:
    """Computes the shortest segment between two segments

    Parameters
    ----------
    segment1 : Segment
        The first segment object
    segment2 : Segment
        The second segment object
    tol : float
        Epsilon. It is a small float number. By default float64 eps

    Returns
    -------
    t1 : float
        The t-parameter in the range [0, 1] for the first segment
    t2 : float
        The t-parameter in the range [0, 1] for the second segment

    References
    ----------
    .. [1] `Distance between lines and segments
            <http://geomalgorithms.com/a07-_distance.html>`_

    """

    u = segment1.direction
    v = segment2.direction
    w = segment1.p1 - segment2.p1

    a = u @ u
    b = u @ v
    c = v @ v
    d = u @ w
    e = v @ w

    dd = a * c - b * b
    sd = td = dd

    # Compute the line parameters of the two closest points
    if dd < tol:
        # the lines are almost parallel
        sn, sd = 0.0, 1.0
        tn, td = e, c
    else:
        sn = b * e - c * d
        tn = a * e - b * d

        if sn < 0.0:
            # sc < 0 => the s=0 edge is visible
            sn = 0.0
            tn, td = e, c
        elif sn > sd:
            # sc > 1  => the s=1 edge is visible
            sn = sd
            tn = e + b
            td = c

    if tn < 0.0:
        # tc < 0 => the t=0 edge is visible
        tn = 0.0
        # recompute sc for this edge
        if -d < 0.0:
            sn = 0.0
        elif -d > a:
            sn = sd
        else:
            sn = -d
            sd = a
    elif tn > td:
        # tc > 1  => the t=1 edge is visible
        tn = td
        # recompute sc for this edge
        if (-d + b) < 0.0:
            sn = 0
        elif (-d + b) > a:
            sn = sd
        else:
            sn = -d + b
            sd = a

    # finally do the division to get sc and tc
    t1 = 0.0 if np.abs(sn) < tol else sn / sd
    t2 = 0.0 if np.abs(tn) < tol else tn / td

    return t1, t2


def segments_to_segments(data1: np.ndarray, data2: np.ndarray, tol: float = F_EPS) \
        -> ty.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes the shortest segments between all segment pairs from two segment sets

    Computes the shortest segments between all segment pairs and returns M1xM2 matrices for t1 and t2 parameters.

    Parameters
    ----------
    data1 : np.ndarray
        The first M1xN data
    data2 : np.ndarray
        The second M2xN data
    tol : float
        Tolerance

    Returns
    -------
    t1 : np.ndarray
        M1xM2 matrix fot t1 parameter
    t2 : np.ndarray
        M1xM2 matrix fot t2 parameter
    p1 : np.ndarray
        NxM1xM2 array of beginning points of shortest segments on the first segments
    p2 : np.ndarray
        NxM1xM2 array of ending points of shortest segments on the second segments

    Notes
    -----

    This function is vectorized version of `segment_to_segment`

    See Also
    --------
    segment_to_segment

    """

    m1 = data1.shape[0] - 1
    m2 = data2.shape[0] - 1

    # Segment direction vectors
    u = np.diff(data1, axis=0)[np.newaxis].transpose(2, 1, 0)
    v = np.diff(data2, axis=0)[np.newaxis].transpose(2, 0, 1)

    p11 = data1[:-1, :][np.newaxis].transpose(2, 1, 0)
    p21 = data2[:-1, :][np.newaxis].transpose(2, 0, 1)

    w = p11 - p21

    # Vectorized computing dot products
    a = np.einsum('ijk,ijk->jk', u, u).repeat(m2, axis=1)
    b = np.einsum('ijk,ikl->jl', u, v)
    c = np.einsum('ijk,ijk->jk', v, v).repeat(m1, axis=0)
    d = np.einsum('ijk,ijl->jl', u, w)
    e = np.einsum('ijk,ilk->lk', v, w)

    dd = a * c - b * b
    sd = dd.copy()
    td = dd.copy()

    sn = np.zeros_like(dd)
    tn = np.zeros_like(dd)

    dd_lt_tol = dd < tol
    not_dd_lt_tol = ~dd_lt_tol

    sd[dd_lt_tol] = 1.0
    tn[dd_lt_tol] = e[dd_lt_tol]
    td[dd_lt_tol] = c[dd_lt_tol]

    be_cd = b * e - c * d
    ae_bd = a * e - b * d

    sn[not_dd_lt_tol] = be_cd[not_dd_lt_tol]
    tn[not_dd_lt_tol] = ae_bd[not_dd_lt_tol]

    sn_lt_zero = (sn < 0) & not_dd_lt_tol

    sn[sn_lt_zero] = 0.0
    tn[sn_lt_zero] = e[sn_lt_zero]
    td[sn_lt_zero] = c[sn_lt_zero]

    sn_gt_sd = (sn > sd) & not_dd_lt_tol

    sn[sn_gt_sd] = sd[sn_gt_sd]
    tn[sn_gt_sd] = e[sn_gt_sd] + b[sn_gt_sd]
    td[sn_gt_sd] = c[sn_gt_sd]

    tn_lt_zero = tn < 0
    tn[tn_lt_zero] = 0.0

    md_lt_zero = (-d < 0) & tn_lt_zero
    md_gt_a = (-d > a) & tn_lt_zero
    md_else = (~md_lt_zero & ~md_gt_a) & tn_lt_zero

    sn[md_lt_zero] = 0.0
    sn[md_gt_a] = sd[md_gt_a]
    sn[md_else] = -d[md_else]
    sd[md_else] = a[md_else]

    tn_gt_td = tn > td
    tn[tn_gt_td] = td[tn_gt_td]

    bmd = b - d

    bmd_lt_zero = (bmd < 0) & tn_gt_td
    bmd_gt_a = (bmd > a) & tn_gt_td
    bmd_else = (~bmd_lt_zero & ~bmd_gt_a) & tn_gt_td

    sn[bmd_lt_zero] = 0.0
    sn[bmd_gt_a] = sd[bmd_gt_a]
    sn[bmd_else] = bmd[bmd_else]
    sd[bmd_else] = a[bmd_else]

    abs_sn_gt_tol = ~(np.abs(sn) < tol)
    abs_tn_gt_tol = ~(np.abs(tn) < tol)

    t1 = np.zeros_like(dd)
    t2 = np.zeros_like(dd)

    t1[abs_sn_gt_tol] = (sn / sd)[abs_sn_gt_tol]
    t2[abs_tn_gt_tol] = (tn / td)[abs_tn_gt_tol]

    p1 = p11 + t1 * u
    p2 = p21 + t2 * v

    return t1, t2, p1, p2
