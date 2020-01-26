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
import functools

import typing as ty
import numpy as np

from cached_property import cached_property

from skcurve._numeric import dot1d
import skcurve._base as _base

if ty.TYPE_CHECKING:
    from skcurve._base import Curve, CurvePoint  # noqa


DEFAULT_GRAD_EDGE_ORDER = 2


class DifferentialGeometryWarning(UserWarning):
    """Raises when diffgeom computation problems occurred
    """


def nonsingular(curve: 'Curve'):
    """Removes singularities in a curve

    The function removes NaN, Inf and the close points from curve to avoid segments with zero-closed lengths.
    These points/segments of an exceptional set where a curve fails to be well-behaved in some
    particular way, such as differentiability for example.

    Parameters
    ----------
    curve : Curve
        Curve object

    Returns
    -------
    curve : Curve
        Curve without singularities.

    """

    chord_lengths = curve.chordlen

    def is_singular(curve_data):
        return (np.any(np.isnan(curve_data) | np.isinf(curve_data), axis=1) |
                np.isclose(np.hstack([1.0, chord_lengths]), 0.0))

    return curve.drop(is_singular)


def chordlen(curve: 'Curve') -> np.ndarray:
    """Computes length for every chord (segment) of a curve

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
        The array 1x(M-1) with lengths for every curve segment

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


def _gradient(data: np.ndarray, edge_order: int = DEFAULT_GRAD_EDGE_ORDER) -> np.ndarray:
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
    grad : np.ndarray
        The array MxN of gradient vectors

    """

    m_rows = data.shape[0]

    if m_rows == 0:
        return np.array([], dtype=np.float64)

    for edge_order in range(edge_order, 0, -1):
        if m_rows < (edge_order + 1):
            warnings.warn(
                f'The number of data points {m_rows} too small to calculate a numerical '
                f'gradient, at least {edge_order + 1} (edge_order + 1) elements are required.',
                DifferentialGeometryWarning)
        else:
            break
    else:
        warnings.warn('Cannot calculate a numerical gradient.', DifferentialGeometryWarning)
        return np.zeros((m_rows, 1), dtype=np.float64)

    return np.gradient(data, axis=0, edge_order=edge_order)


def _frenet_vector_norm(v: np.ndarray, warn_msg: str) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1)
    not_well_defined = np.isclose(norm, 0.0)
    is_not_well_defined = np.any(not_well_defined)

    if is_not_well_defined:
        warnings.warn(warn_msg, DifferentialGeometryWarning, stacklevel=2)
        norm[not_well_defined] = 1.0

    return norm[:, np.newaxis]


def coorientplane(curve: 'Curve', axis1: int = 0, axis2: int = 1) -> bool:
    """Returns True if a curve co-oriented to the given plane

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
    IndexError : Axis out of dimensions
    """

    if not curve or curve.size == 1:
        return True

    pb = curve[0]
    pe = curve[-1]

    return np.linalg.det([
        [pb[axis1], pe[axis1]],
        [pb[axis2], pe[axis2]],
    ]) > 0


def _maybe_curve_point(func):
    @functools.wraps(func)
    def wrapper(obj):
        if isinstance(obj, _base.CurvePoint):
            return getattr(obj.curve, func.__name__)[obj.idx]
        return func(obj)
    return wrapper


class CurvePointFunctionMixin:
    """Provides some curve point functions (diff geom computations) for Curve or CurvePoint

    Notes
    -----
    The class can be used only as mixin class for :class:`Curve` and :class:`CurvePoint`.

    """

    @cached_property
    @_maybe_curve_point
    def cumarclen(self: ty.Union['Curve', 'CurvePoint']) -> ty.Union[np.ndarray, float]:
        """Returns value of cumulative arc length for the curve or at the curve point

        Parametrization of the curve by the length of its arc.

        Returns
        -------
        val : Union[np.ndarray, float]
            The value of cumulative arc length for this curve or point

        """

        return np.hstack((0.0, np.cumsum(self.chordlen)))

    @cached_property
    @_maybe_curve_point
    def firstderiv(self: ty.Union['Curve', 'CurvePoint']) -> np.ndarray:
        """Returns first-order derivative at the curve points

        Returns
        -------
        fder : np.ndarray
             The 1xN array of first-order derivative at the curve point(s)

        See Also
        --------
        tangent
        secondderiv
        thirdderiv

        """

        return _gradient(self.data)

    @cached_property
    @_maybe_curve_point
    def secondderiv(self: ty.Union['Curve', 'CurvePoint']) -> np.ndarray:
        """Returns second-order derivative at the curve points

        Returns
        -------
        sder : np.ndarray
             The 1xN array of second-order derivative at the curve point(s)

        See Also
        --------
        firstderiv
        thirdderiv

        """

        return _gradient(self.firstderiv)

    @cached_property
    @_maybe_curve_point
    def thirdderiv(self: ty.Union['Curve', 'CurvePoint']) -> np.ndarray:
        """Returns third-order derivative at the curve points

        Returns
        -------
        tder : np.ndarray
             The 1xN array of third-order derivative at the curve point(s)

        See Also
        --------
        firstderiv
        secondderiv

        """

        return _gradient(self.secondderiv)

    @cached_property
    def tangent(self: ty.Union['Curve', 'CurvePoint']) -> np.ndarray:
        """Returns tangent vector for the curve points

        Notes
        -----
        This is alias for :func:`firstderiv` property.

        Returns
        -------
        tangent : np.ndarray
            The 1xN array of tangent vector for the curve point(s)

        See Also
        --------
        firstderiv
        frenet1

        """

        return self.firstderiv

    @cached_property
    @_maybe_curve_point
    def normal(self: ty.Union['Curve', 'CurvePoint']) -> np.ndarray:
        r"""Returns normal vector at the curve points

        .. math::
            \overline{e_2}(t) = \gamma''(t) - \langle \gamma''(t), e_1(t) \rangle \, e_1(t)

        Notes
        -----
        The normal vector, sometimes called the curvature vector,
        indicates the deviance of the curve from being a straight line.

        Returns
        -------
        normal : np.ndarray
            The 1xN array of normal vector at the curve point(s)

        See Also
        --------
        tangent
        frenet2
        curvature

        """

        curve = self

        if not curve:
            return np.array([], ndmin=curve.ndim, dtype=np.float64)

        sder = curve.secondderiv
        e1 = curve.frenet1

        return sder - dot1d(sder, e1)[:, np.newaxis] * e1

    @cached_property
    @_maybe_curve_point
    def binormal(self: ty.Union['Curve', 'CurvePoint']) -> np.ndarray:
        r"""Returns binormal vector at the curve points

        .. math::
            \overline{e_3}(t) = \gamma'''(t) - \langle \gamma'''(t), e_1(t) \rangle \, e_1(t)
            - \langle \gamma'''(t), e_2(t) \rangle \, e_2(t)

        Notes
        -----
        The binormal vector is always orthogonal to the tangent and normal vectors at every point of the curve.

        Returns
        -------
        binormal : np.ndarray
            The 1xN array of binormal vector at the curve point(s)

        See Also
        --------
        tangent
        normal
        frenet3
        torsion

        """

        curve = self

        if not curve:
            return np.array([], ndmin=curve.ndim, dtype=np.float64)

        tder = curve.thirdderiv
        e1 = curve.frenet1
        e2 = curve.frenet2

        return tder - dot1d(tder, e1)[:, np.newaxis] * e1 - dot1d(tder, e2)[:, np.newaxis] * e2

    @cached_property
    @_maybe_curve_point
    def speed(self: ty.Union['Curve', 'CurvePoint']) -> ty.Union[np.ndarray, float]:
        """Returns the speed at the curve point

        Notes
        -----
        The speed is the tangent (velocity) vector's magnitude (norm).
        In general, speed may be zero at some point if the curve has zero-length segments.

        Returns
        -------
        speed : float
            The speed value at the point(s)

        See Also
        --------
        tangent

        """

        return np.linalg.norm(self.tangent, axis=1)

    @cached_property
    @_maybe_curve_point
    def frenet1(self: ty.Union['Curve', 'CurvePoint']) -> np.ndarray:
        r"""Returns the first Frenet vector (unit tangent vector) at the curve point

        .. math::

            e_1(t) = \frac{\gamma'(t)}{||\gamma'(t)||}

        Returns
        -------
        e1 : np.ndarray
            The first Frenet vector (unit tangent) ath the curve point(s)

        See Also
        --------
        frenet2
        frenet3

        """

        curve = self

        if not curve:
            return np.array([], ndmin=curve.ndim, dtype=np.float64)

        not_well_defined_warn_msg = (
            'Cannot calculate the first Frenet vectors (unit tangent vectors). '
            'The curve has singularity and zero-length segments. '
            'Use "Curve.nonsingular" method to remove singularity from the curve data.'
        )

        norm = _frenet_vector_norm(curve.tangent, not_well_defined_warn_msg)
        return curve.tangent / norm

    @cached_property
    @_maybe_curve_point
    def frenet2(self: ty.Union['Curve', 'CurvePoint']) -> np.ndarray:
        r"""Returns the second Frenet vector (unit normal vector) at the curve point

        .. math::

            e_2(t) = \frac{e_1'(t)}{||e_1'(t)||}

        Returns
        -------
        e2 : np.ndarray
            The second Frenet vector (unit normal vector) at the curve point(s)

        Raises
        ------
        ValueError : Cannot compute unit vector if speed is equal to zero (division by zero)

        See Also
        --------
        normal
        frenet1
        frenet3

        """

        curve = self

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

    @cached_property
    @_maybe_curve_point
    def frenet3(self: ty.Union['Curve', 'CurvePoint']) -> np.ndarray:
        r"""Returns the third Frenet vector (unit binormal vector) at the curve point

        .. math::

            e_3(t) = \frac{\overline{e_3}(t)}{||\overline{e_3}(t)||}

        Returns
        -------
        e3 : np.ndarray
            The third Frenet vector (unit binormal vector) at the curve point(s)

        Raises
        ------
        ValueError : Cannot compute unit vector if speed is equal to zero (division by zero)

        See Also
        --------
        frenet1
        frenet2

        """

        curve = self

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

    @cached_property
    @_maybe_curve_point
    def curvature(self: ty.Union['Curve', 'CurvePoint']) -> ty.Union[np.ndarray, float]:
        r"""Returns the curvature value at the curve point

        The curvature of a plane curve or a space curve in three dimensions (and higher) is the magnitude of the
        acceleration of a particle moving with unit speed along a curve.

        Curvature formula for 2-dimensional (a plane) curve :math:`\gamma(t) = (x(t), y(t))`:

        .. math::

            k = \frac{y''x' - x''y'}{(x'^2 + y'^2)^\frac{3}{2}}

        and for 3-dimensional curve :math:`\gamma(t) = (x(t), y(t), z(t))`:

        .. math::

            k = \frac{||\gamma' \times \gamma''||}{||\gamma'||^3}

        and for n-dimensional curve in general:

        .. math::

            k = \frac{\sqrt{||\gamma'||^2||\gamma''||^2 - (\gamma' \cdot \gamma'')^2}}{||\gamma'||^3}

        Notes
        -----
        Curvature values at the ends of the curve can be calculated less accurately.

        Returns
        -------
        k : float
            The curvature value at this point

        See Also
        --------
        normal
        torsion

        """

        curve = self

        if not curve:
            return np.array([], ndmin=curve.ndim, dtype=np.float64)

        if curve.is2d:
            # Compute signed curvature value for a plane curve
            fd_x = curve.firstderiv[:, 0]
            fd_y = curve.firstderiv[:, 1]
            sd_x = curve.secondderiv[:, 0]
            sd_y = curve.secondderiv[:, 1]

            k = (sd_y * fd_x - sd_x * fd_y) / (fd_x * fd_x + fd_y * fd_y) ** 1.5
        else:
            # Compute curvature for 3 or higher dimensional curve
            e1_grad = _gradient(curve.frenet1)
            e1_grad_norm = np.linalg.norm(e1_grad, axis=1)
            tangent_norm = np.linalg.norm(curve.tangent, axis=1)

            k = e1_grad_norm / tangent_norm

        return k

    @cached_property
    @_maybe_curve_point
    def torsion(self: ty.Union['Curve', 'CurvePoint']) -> ty.Union[np.ndarray, float]:
        r"""Returns the torsion value at the curve point

        The second generalized curvature is called torsion and measures the deviance of the curve
        from being a plane curve. In other words, if the torsion is zero, the curve lies completely
        in the same osculating plane (there is only one osculating plane for every point t).

        It is defined as:

        .. math::

            \tau(t) = \chi_2(t) = \frac{\langle e_2'(t), e_3(t) \rangle}{\| \gamma'(t) \|}

        Returns
        -------
        tau : ndarray, float
            The torsion value at the curve point(s)

        See Also
        --------
        curvature

        """

        curve = self

        if not curve:
            return np.array([], ndmin=curve.ndim, dtype=np.float64)

        e2_grad = _gradient(curve.frenet2)
        tangent_norm = np.linalg.norm(curve.tangent, axis=1)

        return dot1d(e2_grad, curve.frenet3) / tangent_norm
