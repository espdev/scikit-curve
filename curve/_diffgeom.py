# -*- coding: utf-8 -*-

"""
Differential geometry of curves

"""

import typing as t
import numpy as np

if t.TYPE_CHECKING:
    from ._base import Curve


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

    dr = np.gradient(curve.data, axis=0, edge_order=2)
    ddr = np.gradient(dr, axis=0, edge_order=2)

    p = 3 / 2

    if curve.is1d:
        k = ddr / (1 + dr ** 2) ** p
    else:
        # Compute curvature for 2 or higher dimensional curve
        ssq_dr = np.sum(dr ** 2, axis=1)
        ssq_ddr = np.sum(ddr ** 2, axis=1)
        dot_sq_dr_ddr = np.sum(dr * ddr, axis=1) ** 2

        k = np.sqrt(ssq_dr * ssq_ddr - dot_sq_dr_ddr) / ssq_dr ** p

    return k
