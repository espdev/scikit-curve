# -*- coding: utf-8 -*-

"""
Differential geometry of curves

"""

import typing as t
import numpy as np

from ._base import Curve


def is_plane(curve: Curve) -> bool:
    """Returns True if a curve is plane

    The plane curve is 2-dimensional curve (curve on plane).

    Parameters
    ----------
    curve : Curve
        Curve object

    Returns
    -------
    flag : bool
        True if a curve is plane

    """

    return curve.ndim == 2


def is_spatial(curve: Curve) -> bool:
    """Returns True if a curve is spatial

    The spatial curve is 3-dimensional curve (curve in 3-d space).

    Parameters
    ----------
    curve : Curve
        Curve object

    Returns
    -------
    flag : bool
        True if a curve is spatial

    """

    return curve.ndim == 3


def nonsingular(curve: Curve, seglen: t.Optional[np.ndarray] = None):
    """Removes singularities in a curve

    The method removes NaN, Inf and the close points from curve to avoid segments with zero-closed lengths.
    These points/segments of an exceptional set where a curve fails to be well-behaved in some
    particular way, such as differentiability for example.

    Parameters
    ----------
    curve : Curve
        Curve object
    seglen : np.ndarray
        Numpy vector with lengths for each curve segment.
        If it is not set it will be computed.

    Returns
    -------
    curve : Curve
        Curve without singularities.

    """

    if seglen is None:
        seglen = seglength(curve)

    def is_singular(curve_data):
        return (np.any(np.isnan(curve_data) | np.isinf(curve_data), axis=1) |
                np.isclose(np.hstack([1.0, seglen]), 0.0))

    return curve.drop(is_singular)


def seglength(curve: Curve) -> np.ndarray:
    """Computes length for each segment of a curve

    Notes
    -----

    * ``euclidean`` metric is used to compute segment lengths
    * Currently, linear approximation is used to compute arc length

    Parameters
    ----------
    curve: Curve
        Curve object

    Returns
    -------
    seglen : np.ndarray
        Numpy vector with lengths for each curve segment

    """

    if curve.size == 0:
        return np.ndarray([], dtype=float)

    seg_len = np.sqrt(np.sum((np.diff(curve.data, axis=0))**2, axis=1))

    # TODO: Implement numerical integration

    return seg_len


def arclength(curve: Curve) -> float:
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
        Arc length

    """

    return float(np.sum(seglength(curve)))


def natural_parametrization(curve: Curve) -> np.ndarray:
    """Computes natural parameter vector for given curve

    Parametrization of a curve by the length of its arc.

    Parameters
    ----------
    curve : Curve
        Curve object

    Returns
    -------
    t : np.ndarray
        Natural parameter vector

    """

    # TODO: It is required numerical integration in a good way

    return np.hstack((0.0, np.cumsum(seglength(curve))))
