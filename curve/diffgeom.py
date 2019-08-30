# -*- coding: utf-8 -*-

"""
Differential geometry of curves

"""

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


def seg_length(curve: Curve) -> np.ndarray:
    """Computes length for each segment of curve

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
    seg_lenght : np.ndarray
        Numpy vector with lengths for each curve segment

    """

    if curve.size == 0:
        return np.ndarray([], dtype=float)

    p1 = curve.data[:-1]
    p2 = curve.data[1:]

    seg_len = np.sqrt(np.sum((p1 - p2)**2, axis=1))

    # TODO: Implement numerical integration

    return seg_len


def arc_length(curve: Curve) -> float:
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

    return np.sum(seg_length(curve))


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

    return np.hstack((0.0, np.cumsum(seg_length(curve))))
