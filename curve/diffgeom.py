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
    seg_lenght : np.ndarray
        Numpy vector with lengths for each curve segment

    """

    if curve.size == 0:
        return np.ndarray([], dtype=float)

    seg_len = np.sqrt(np.sum((np.diff(curve.data, axis=0))**2, axis=1))

    # TODO: Implement numerical integration

    return seg_len


def remove_singularity(curve: Curve):
    """Removes singularities in a curve

    The method removes NaN, Inf and the close points from curve to avoid segments with zero-closed lengths.
    These points/segments of an exceptional set where a curve fails to be well-behaved in some
    particular way, such as differentiability for example.

    Parameters
    ----------
    curve: Curve
        Curve object

    Returns
    -------
    curve : Curve
        Curve without singularities. This is new curve object if original curve has singularities or
        original curve object if they are not.

    """

    is_close = np.isclose(np.hstack([1.0, seglength(curve)]), 0.0)
    is_nan = np.any(np.isnan(curve.data), axis=1)
    is_inf = np.any(np.isinf(curve.data), axis=1)

    singular = np.flatnonzero(is_close | is_nan | is_inf)

    if singular.size == 0:
        return curve
    else:
        return curve.delete(singular)


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

    return np.sum(seglength(curve))


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
