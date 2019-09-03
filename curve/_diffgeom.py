# -*- coding: utf-8 -*-

"""
Differential geometry of curves

"""

import warnings

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


class GradientWarning(UserWarning):
    """Raises when gradient computation problems occurred
    """
    pass


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
            ).format(m_rows, edge_order + 1), GradientWarning)
        else:
            break
    else:
        warnings.warn('Cannot calculate a numerical gradient.', GradientWarning)
        return np.zeros((m_rows, 1), dtype=np.float64)

    return np.gradient(data, axis=0, edge_order=edge_order)


def tangent(curve: 'Curve') -> np.ndarray:
    """Computes tangent unit vector (normalized vector) for each point of a n-dimensional curve

    Parameters
    ----------
    curve : Curve
        Curve object

    Returns
    -------
    tangent : np.ndarray
        The array of tangent unit vectors for each curve points

    Raises
    ------
    ValueError : Cannot compute vector norm (division by zero)

    """

    if not curve:
        return np.array([], ndmin=curve.ndim, dtype=np.float64)

    norm = np.sqrt(np.sum(curve.firstderiv ** 2, axis=1))

    if np.any(np.isclose(norm, 0.0)):
        raise ValueError('The curve has singularity and zero-length segments. '
                         'Use "Curve.nonsingular" method to remove singularity.')

    return curve.firstderiv / np.array(norm, ndmin=2).T


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

    p = 3 / 2

    if curve.is1d:
        k = curve.secondderiv / (1 + curve.firstderiv ** 2) ** p
    else:
        # Compute curvature for 2 or higher dimensional curve
        ssq_dr = np.sum(curve.firstderiv ** 2, axis=1)
        ssq_ddr = np.sum(curve.secondderiv ** 2, axis=1)
        dot_sq_dr_ddr = np.einsum('ij,ij->i', curve.firstderiv, curve.secondderiv) ** 2

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
