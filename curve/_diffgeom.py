# -*- coding: utf-8 -*-

"""
Differential geometry of n-dimensional curves

The module provides numerical computing methods for
curve's differential geometry in the plane and the Euclidean n-dimensional space.

Here are some basic routines above the n-dimensional curves:

    * Computing numerical gradient (differentiation) first and higher orders
    * Computing chord and arc length, natural parametrization
    * Computing tangent, normal, binormal, Frenet vectors
    * Computing curvature and torsion

"""

import warnings

import typing as t
import numpy as np

from curve._numeric import rowdot

if t.TYPE_CHECKING:
    from curve._base import Curve


DEFAULT_GRAD_EDGE_ORDER = 2


class DifferentialGeometryWarning(UserWarning):
    """Raises when diffgeom computation problems occurred
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

    sder = curve.secondderiv
    e1 = curve.frenet1

    return sder - rowdot(sder, e1)[:, np.newaxis] * e1


def binormal(curve: 'Curve') -> np.ndarray:
    """Computes binormal vector at every point of a curve

    Parameters
    ----------
    curve : Curve
        Curve object

    Returns
    -------
    binormal : np.ndarray
        The array MxN with binormal vector at every point of curve

    """

    if not curve:
        return np.array([], ndmin=curve.ndim, dtype=np.float64)

    tder = curve.thirdderiv
    e1 = curve.frenet1
    e2 = curve.frenet2

    return tder - rowdot(tder, e1)[:, np.newaxis] * e1 - rowdot(tder, e2)[:, np.newaxis] * e2


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

    return np.linalg.norm(curve.tangent, axis=1)


def _frenet_vector_norm(v: np.ndarray, warn_msg: str) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1)
    not_well_defined = np.isclose(norm, 0.0)
    is_not_well_defined = np.any(not_well_defined)

    if is_not_well_defined:
        warnings.warn(warn_msg, DifferentialGeometryWarning, stacklevel=2)
        norm[not_well_defined] = 1.0

    return norm[:, np.newaxis]


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

    """

    if not curve:
        return np.array([], ndmin=curve.ndim, dtype=np.float64)

    not_well_defined_warn_msg = (
        'Cannot calculate the first Frenet vectors (unit tangent vectors). '
        'The curve has singularity and zero-length segments. '
        'Use "Curve.nonsingular" method to remove singularity from the curve data.'
    )

    norm = _frenet_vector_norm(curve.tangent, not_well_defined_warn_msg)
    return curve.tangent / norm


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

    if not curve:
        return np.array([], ndmin=curve.ndim, dtype=np.float64)

    not_well_defined_warn_msg = (
        'Cannot calculate the second Frenet vectors (unit normal vectors). '
        'The curve is straight line and normal vectors are not well defined. '
    )

    norm = _frenet_vector_norm(curve.normal, not_well_defined_warn_msg)
    e2 = curve.normal / norm

    # FIXME: what to do with not well defined the normal vectors?

    return e2


def frenet3(curve: 'Curve') -> np.ndarray:
    """Computes the third Frenet vector (binormal unit vector) at every point of a curve

    Parameters
    ----------
        Curve object

    Returns
    -------
    e3 : np.ndarray
        The MxN array of binormal unit vector at every curve point

    """

    if not curve:
        return np.array([], ndmin=curve.ndim, dtype=np.float64)

    not_well_defined_warn_msg = (
        'Cannot calculate the third Frenet vectors (unit binormal vectors). '
        'May be the curve is straight line or a plane and binormal vectors are not well defined. '
    )

    norm = _frenet_vector_norm(curve.binormal, not_well_defined_warn_msg)
    e3 = curve.binormal / norm

    # FIXME: what to do with not well defined the binormal vectors?

    return e3


def curvature(curve: 'Curve') -> np.ndarray:
    """Computes the curvature at every point of a curve

    Parameters
    ----------
    curve : Curve
        Curve object

    Returns
    -------
    k : np.ndarray
        The array Mx1 with curvature value at every point of a curve

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
        e1_grad_norm = np.linalg.norm(e1_grad, axis=1)
        tangent_norm = np.linalg.norm(curve.tangent, axis=1)

        k = e1_grad_norm / tangent_norm

    return k


def torsion(curve: 'Curve') -> np.ndarray:
    """Computes the torsion at every point of a curve

    Parameters
    ----------
    curve : Curve
        Curve object

    Returns
    -------
    torsion : np.ndarray
        The array Mx1 with torsion value at every point of a curve

    """

    if not curve:
        return np.array([], ndmin=curve.ndim, dtype=np.float64)

    e2_grad = gradient(curve.frenet2)
    tangent_norm = np.linalg.norm(curve.tangent, axis=1)

    return rowdot(e2_grad, curve.frenet3) / tangent_norm


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
