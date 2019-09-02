# -*- coding: utf-8 -*-

"""
Differential geometry of curves

"""

import typing as t
import numpy as np

if t.TYPE_CHECKING:
    from ._base import Curve

DEFAULT_GRAD_EDGE_ORDER = 2


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
        Curve data or curve data direvatives
    edge_order : {1, 2} int
        Specify how boundaries are treated.
        Gradient is calculated using N-th order accurate differences at the boundaries.
        2 is more precision but it is required more data.

    Returns
    -------
    dr_dt : np.ndarray
        Direvatives data

    """

    if data.shape[0] == 0:
        return np.array([], dtype=np.float64)

    if data.shape[0] < (edge_order + 1):
        return np.zeros((data.shape[0], 1), dtype=np.float64)

    return np.gradient(data, axis=0, edge_order=edge_order)


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
        return np.array([], dtype=np.float64)

    p = 3 / 2

    if curve.is1d:
        k = curve.secondderiv / (1 + curve.firstderiv ** 2) ** p
    else:
        # Compute curvature for 2 or higher dimensional curve
        ssq_dr = np.sum(curve.firstderiv ** 2, axis=1)
        ssq_ddr = np.sum(curve.secondderiv ** 2, axis=1)
        dot_sq_dr_ddr = np.sum(curve.firstderiv * curve.secondderiv, axis=1) ** 2

        k = np.sqrt(ssq_dr * ssq_ddr - dot_sq_dr_ddr) / ssq_dr ** p

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
