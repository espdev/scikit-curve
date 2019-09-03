# -*- coding: utf-8 -*-

"""
Differential geometry of curves

"""

import warnings

import typing as t
import numpy as np

from curve._numeric import rowdot

if t.TYPE_CHECKING:
    from curve._base import Curve


DEFAULT_GRAD_EDGE_ORDER = 2


class DifferentialGeometryWarning(UserWarning):
    """Raises when gradient computation problems occurred
    """
    pass


def nonsingular(curve: 'Curve', chord_lengths: t.Optional[np.ndarray] = None):
    """Removes singularities in a curve

    The function removes NaN, Inf and the close points from curve to avoid segments with zero-closed lengths.
    These points/segments of an exceptional set where a curve fails to be well-behaved in some
    particular way, such as differentiability for example.

    Parameters
    ----------
    curve : Curve
        Curve object
    chord_lengths : np.ndarray
        An array with lengths for each curve chord (segment).
        If it is not set it will be computed.

    Returns
    -------
    curve : Curve
        Curve without singularities.

    """

    if chord_lengths is None:
        chord_lengths = chordlen(curve)

    def is_singular(curve_data):
        return (np.any(np.isnan(curve_data) | np.isinf(curve_data), axis=1) |
                np.isclose(np.hstack([1.0, chord_lengths]), 0.0))

    return curve.drop(is_singular)


def chordlen(curve: 'Curve') -> np.ndarray:
    """Computes length for each chord (segment) of a curve

    Notes
    -----
    * ``euclidean`` metric is used to compute segment lengths

    Parameters
    ----------
    curve: Curve
        Curve object

    Returns
    -------
    lengths : np.ndarray
        Numpy vector with lengths for each curve segment

    """

    if not curve:
        return np.ndarray([], dtype=np.float64)

    return np.sqrt(np.sum((np.diff(curve.data, axis=0))**2, axis=1))


def arclen(curve: 'Curve') -> float:
    """Computes the length of a curve arc

    Notes
    -----
    Currently, linear approximation is used to compute arc length

    Parameters
    ----------
    curve: Curve
        Curve object

    Returns
    -------
    length : float
        Curve arc length

    """

    # TODO: Implement numerical integration to increase accuracy

    return float(np.sum(chordlen(curve)))


def natural_parametrization(curve: 'Curve', chord_lengths: t.Optional[np.ndarray] = None) -> np.ndarray:
    """Computes natural parameter vector for given curve

    Parametrization of a curve by the length of its arc.

    Parameters
    ----------
    curve : Curve
        Curve object
    chord_lengths : np.ndarray
        An array with lengths for each curve chord (segment).
        If it is not set it will be computed.

    Returns
    -------
    t : np.ndarray
        Natural parameter vector

    """

    if chord_lengths is None:
        chord_lengths = chordlen(curve)

    return np.hstack((0.0, np.cumsum(chord_lengths)))


def gradient(data: np.ndarray, edge_order: int = DEFAULT_GRAD_EDGE_ORDER) -> np.ndarray:
    """Computes gradient for MxN data array where N is the dimension

    Parameters
    ----------
    data : np.ndarray
        Curve data or MxN curve data derivatives array
    edge_order : {1, 2} int
        Specify how boundaries are treated.
        Gradient is calculated using N-th order accurate differences at the boundaries.
        2 is more precision but it is required more data.

    Returns
    -------
    dr_dt : np.ndarray
        MxN derivatives array

    """

    m_rows = data.shape[0]

    if m_rows == 0:
        return np.array([], dtype=np.float64)

    for edge_order in range(edge_order, 0, -1):
        if m_rows < (edge_order + 1):
            warnings.warn((
                'The number of data points {} too small to calculate a numerical gradient, '
                'at least {} (edge_order + 1) elements are required.'
            ).format(m_rows, edge_order + 1), DifferentialGeometryWarning)
        else:
            break
    else:
        warnings.warn('Cannot calculate a numerical gradient.', DifferentialGeometryWarning)
        return np.zeros((m_rows, 1), dtype=np.float64)

    return np.gradient(data, axis=0, edge_order=edge_order)


def normal(curve: 'Curve') -> np.array:
    """Computes the normal vectors for each point of curve

    Notes
    -----
    The normal vector, sometimes called the curvature vector, indicates the deviance of the curve from
    being a straight line.

    Parameters
    ----------
    curve : Curve
        Curve object

    Returns
    -------
    normal : np.ndarray
        The array MxN with normal vectors for each point of curve

    """

    if not curve:
        return np.array([], ndmin=curve.ndim, dtype=np.float64)

    dot_rdd_e1 = rowdot(curve.secondderiv, curve.frenet1, ndmin=2).T

    return curve.secondderiv - dot_rdd_e1 * curve.frenet1


def speed(curve: 'Curve') -> np.ndarray:
    """Computes the speed at the time (in each curve point)

    Notes
    -----
    The speed is the tangent (velocity) vector's magnitude (norm).
    In general speed may be zero in some point if the curve has zero-length segments.

    Parameters
    ----------
    curve : Curve
        Curve object

    Returns
    -------
    speed : np.ndarray
        The array with speed in each curve point

    """

    return np.linalg.norm(curve.firstderiv, axis=1)


def frenet1(curve: 'Curve') -> np.ndarray:
    """Computes the first Frenet vectors (tangent unit vectors) for each point of a curve

    Parameters
    ----------
    curve : Curve
        Curve object

    Returns
    -------
    e1 : np.ndarray
        The MxN array of tangent unit vectors for each curve points

    Raises
    ------
    ValueError : Cannot compute unit vector if speed is equal to zero (division by zero)

    """

    if not curve:
        return np.array([], ndmin=curve.ndim, dtype=np.float64)

    norm = np.array(curve.speed)

    not_well_defined = np.isclose(norm, 0.0)
    is_not_well_defined = np.any(not_well_defined)

    if is_not_well_defined:
        warnings.warn((
            'Cannot calculate the first Frenet vectors (unit tangent vectors). '
            'The curve has singularity and zero-length segments. '
            'Use "Curve.nonsingular" method to remove singularity from the curve data.'
        ), DifferentialGeometryWarning)

        norm[not_well_defined] = 1.0

    return curve.firstderiv / np.array(norm, ndmin=2).T


def frenet2(curve: 'Curve') -> np.ndarray:
    """Computes the second Frenet vectors (normal unit vectors) for each point of a curve

    Parameters
    ----------
        Curve object

    Returns
    -------
    e2 : np.ndarray
        The MxN array of normal unit vectors for each curve points

    """

    norm = np.linalg.norm(curve.normal, axis=1)
    not_well_defined = np.isclose(norm, 0.0)
    is_not_well_defined = np.any(not_well_defined)

    if is_not_well_defined:
        warnings.warn((
            'Cannot calculate the second Frenet vectors (unit normal vectors). '
            'The curve is straight line and normal vectors are not well defined. '
        ), DifferentialGeometryWarning)

        norm[not_well_defined] = 1.0

    e2 = curve.normal / np.array(norm, ndmin=2).T

    # FIXME: what to do with not well defined the normal vectors?

    return e2


def curvature(curve: 'Curve') -> np.ndarray:
    r"""Computes curvature for each point of a curve

    Parameters
    ----------
    curve : Curve
        Curve object

    Returns
    -------
    k : np.ndarray
        Array of the curvature values for each curve point

    """

    if not curve:
        return np.array([], ndmin=curve.ndim, dtype=np.float64)

    if curve.is1d:
        k = (curve.secondderiv / (1 + curve.firstderiv ** 2) ** 1.5).flatten()
    elif curve.is2d:
        # Compute signed curvature value for a plane curve
        fd_x = curve.firstderiv[:, 0]
        fd_y = curve.firstderiv[:, 1]
        sd_x = curve.secondderiv[:, 0]
        sd_y = curve.secondderiv[:, 1]

        k = (sd_y * fd_x - sd_x * fd_y) / (fd_x * fd_x + fd_y * fd_y) ** 1.5
    else:
        # Compute curvature for 3 or higher dimensional curve
        e1_grad = gradient(curve.frenet1)
        k = np.linalg.norm(e1_grad, axis=1) / np.linalg.norm(curve.firstderiv, axis=1)

    return k


def coorientplane(curve: 'Curve', axis1: int = 0, axis2: int = 1) -> bool:
    """Returns True if a curve co-oriented to a plane

    Notes
    -----
    This method is applicable to 2 or higher dimensional curves.
    By default the method orients a curve to XY plane orientation.

    Parameters
    ----------
    curve : Curve
        Curve object
    axis1: int
        First plane axis
    axis2: int
        Second plane axis

    Returns
    -------
    flag : bool
        True if a curve co-oriented to a plane

    Raises
    ------
    ValueError : Curve has the dimension less than 2
    IndexError : Axis out of dimensions
    """

    if curve.is1d:
        raise ValueError('The curve must be 2 or higher dimensional.')

    if not curve or curve.size == 1:
        return True

    pb = curve[0]
    pe = curve[-1]

    return np.linalg.det([
        [pb[axis1], pe[axis1]],
        [pb[axis2], pe[axis2]],
    ]) > 0
