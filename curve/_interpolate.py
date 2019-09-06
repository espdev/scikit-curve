# -*- coding: utf-8 -*-

"""
This module provides routines for n-dimensional curve interpolation

The following interpolation methods are available out of the box:

    * ``linear`` -- linear interpolation
    * ``cubic`` -- cubic spline interpolation
    * ``hermite`` -- piecewise-cubic interpolation matching values and first derivatives
    * ``akima`` -- Akima interpolation
    * ``pchip`` -- PCHIP 1-d monotonic cubic interpolation
    * ``spline`` -- General k-order spline interpolation

"""

import typing as ty
from collections import abc
import warnings
import functools

import numpy as np
import scipy.interpolate as interp

if ty.TYPE_CHECKING:
    from curve._base import Curve


InterpPType = ty.Union[np.ndarray, int]

_INTERPOLATORS = {}  # type: ty.Dict[str, abc.Callable]


class InterpolationError(Exception):
    """Any interpolation errors
    """


class InterpolationWarning(RuntimeWarning):
    """Any interpolation warning
    """


def make_uniform_interp_grid(curve: 'Curve', pcount: int,
                             extrap_size: ty.Tuple[int, int] = (0, 0),
                             extrap_units='points'):
    """Creates uniform interpolation grid for given number of points

    This is helper function for creating uniform 1-d grids for interpolating of curves.

    Parameters
    ----------
    curve : Curve
        Curve object
    pcount : int
        The number of points
    extrap_size : Tuple[int, int]
        The tuple (left, right): the number of points for left/right extrapolation.
        By default (0, 0).
    extrap_units : str
        The units for extrapolation:
            ``points`` {default} -- extrap_size is interpreted as number of points
            ``length`` {default} -- extrap_size is interpreted as curve length

    Returns
    -------
    interp_grid : np.ndarray
        The array Mx1 with uniform interpolation grid where M is pcount + extrap_pcount

    Raises
    ------
    ValueError : invalid input arguments

    """

    if not curve:
        raise ValueError('The curve is empty.')
    if curve.size == 1:
        raise ValueError('The curve size is too few: {}'.format(curve.size))

    if pcount < 2:
        raise ValueError('There must be at least two interpolation points.')

    if (not isinstance(extrap_size, (tuple, list)) or
            len(extrap_size) != 2 or
            any(not isinstance(pc, (int, float)) or pc < 0 for pc in extrap_size)):
        raise ValueError('Invalid "extrap_size" argument value: {}'.format(extrap_size))

    interp_grid = np.linspace(0, curve.arclen, pcount)
    interp_chordlen = interp_grid[1] - interp_grid[0]

    if extrap_units == 'length':
        extrap_pcount = [int(sz / interp_chordlen) for sz in extrap_size]

        ext_left_pcount, ext_right_pcount = extrap_pcount

    elif extrap_units == 'points':
        ext_left_pcount, ext_right_pcount = extrap_size
    else:
        raise ValueError('Unknown "extrap_units": {}'.format(extrap_units))

    extrap_left_grid = np.linspace(-interp_chordlen * ext_left_pcount, -interp_chordlen, ext_left_pcount)
    extrap_right_grid = np.linspace(interp_grid[-1] + interp_chordlen,
                                    interp_grid[-1] + interp_chordlen * ext_right_pcount, ext_right_pcount)

    interp_grid = np.hstack([extrap_left_grid, interp_grid, extrap_right_grid])
    return interp_grid


def _make_interp_grid(curve: 'Curve', ti: InterpPType) -> np.ndarray:
    if isinstance(ti, int):
        grid = make_uniform_interp_grid(curve, pcount=ti)
    else:
        grid = np.array(ti, dtype=np.float64)

    if grid.ndim != 1:
        raise ValueError(
            'The interpolation grid should be 1xM array where M is number of points in interpolated curve')

    dt = np.diff(grid)

    if np.any(dt < 0) or np.any(np.isclose(dt, 0)):
        raise ValueError(
            'The values in the interpolation grid must be strictly increasing ordered.')

    if np.min(grid) > 0 or np.max(grid) < curve.arclen:
        warnings.warn((
            'The interpolation grid in range [{}, {}]. '
            'It does not cover the whole curve parametrization range [{}, {}].').format(
                np.min(grid), np.max(grid), 0, curve.arclen), InterpolationWarning)

    return grid


def register_interpolator_factory(method: str):
    """Registers interpolator factory

    This decorator can be used for registering custom interpolators.

    Parameters
    ----------
    method : str
        interpolation method

    """

    def decorator(func):
        if method in _INTERPOLATORS:
            raise ValueError('"{}" interpolation method already registered'.format(method))
        _INTERPOLATORS[method] = func
    return decorator


@register_interpolator_factory(method='linear')
def linear_interpolator_factory(curve: 'Curve', *,
                                extrapolate: bool = True):
    """Linearly interpolates a n-dimensional curve data

    Parameters
    ----------
    curve : Curve
        Curve object
    extrapolate : bool
        If bool, determines whether to extrapolate to out-of-bounds points.

    Returns
    -------
    interpolator : callable
        Linear interpolator function

    Notes
    -----
    This interpolator supports extrapolation.

    """

    cumarc_norm = np.hstack((0, np.cumsum(curve.chordlen)))

    def _interpolator(interp_grid: np.ndarray):
        if not extrapolate:
            if np.min(interp_grid) < 0 or np.max(interp_grid) > curve.arclen:
                warnings.warn((
                    '"extrapolate" is disabled but interpolation grid is out-of-bounds curve data. '
                    'The grid will be cut down to curve data bounds.'
                ), InterpolationWarning)

                drop_indices = np.flatnonzero((interp_grid < 0) | (interp_grid > curve.arclen))
                interp_grid = np.delete(interp_grid, drop_indices)

        tbins = np.digitize(interp_grid, cumarc_norm)

        n = curve.size

        tbins[(tbins <= 0)] = 1
        tbins[(tbins >= n) | np.isclose(interp_grid, 1)] = n - 1
        tbins -= 1

        s = (interp_grid - cumarc_norm[tbins]) / curve.chordlen[tbins]

        tbins_data = curve.data[tbins, :]
        tbins1_data = curve.data[tbins + 1, :]

        interp_data = (tbins_data + (tbins1_data - tbins_data) * s[:, np.newaxis])
        return interp_data

    return _interpolator


@register_interpolator_factory(method='cubic')
def cubic_interpolator_factory(curve: 'Curve', *,
                               bc_type='not-a-knot',
                               extrapolate: ty.Union[bool, str, None] = None):
    """Cubic spline interpolator

    Parameters
    ----------
    curve : Curve
        Curve object
    bc_type : str
        Boundary condition type. See scipy docs for details [1]_.
    extrapolate : bool, str, None
        If bool, determines whether to extrapolate to out-of-bounds
        points based on first and last intervals, or to return NaNs.
        If ‘periodic’, periodic extrapolation is used.
        If None (default), extrapolate is set to ‘periodic’
        for bc_type='periodic' and to True otherwise.

    Returns
    -------
    interpolator : CubicSpline
        cubic spline interpolator object

    Notes
    -----
    This interpolator supports extrapolation.

    References
    ----------
    .. [1] `CubicSpline
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html>`_
            on SciPy docs.

    """

    return interp.CubicSpline(curve.t, curve.data, axis=0, bc_type=bc_type, extrapolate=extrapolate)


@register_interpolator_factory(method='hermite')
def hermite_interpolator_factory(curve: 'Curve', *,
                                 extrapolate: ty.Union[bool, str, None] = None):
    """Piecewise-cubic interpolator matching values and first derivatives

    Parameters
    ----------
    curve : Curve
        Curve object
    extrapolate : bool, str, None
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs.
        If ‘periodic’, periodic extrapolation is used.
        If None (default), it is set to True.

    Returns
    -------
    interpolator : CubicHermiteSpline
        Hermite cubic spline interpolator object

    Notes
    -----
    This interpolator supports extrapolation.

    References
    ----------
    .. [1] `CubicHermiteSpline
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.html>`_
            on SciPy docs.

    """

    return interp.CubicHermiteSpline(curve.t, curve.data, curve.frenet1, axis=0, extrapolate=extrapolate)


@register_interpolator_factory(method='akima')
def akima_interpolator_factory(curve: 'Curve'):
    """Akima interpolator

    Parameters
    ----------
    curve : Curve
        Curve object

    Returns
    -------
    interpolator : Akima1DInterpolator
        Akima spline interpolator object

    Notes
    -----
    This interpolator does not support approximation

    References
    ----------
    .. [1] `Akima1DInterpolator
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html>`_
            on SciPy docs.

    """

    return interp.Akima1DInterpolator(curve.t, curve.data, axis=0)


@register_interpolator_factory(method='pchip')
def pchip_interpolator_factory(curve: 'Curve', *,
                               extrapolate: ty.Optional[bool] = None):
    """PCHIP 1-d monotonic cubic interpolation

    Parameters
    ----------
    curve : Curve
        Curve object
    extrapolate : bool, None
        Whether to extrapolate to out-of-bounds points based on first and last intervals,
        or to return NaNs.

    Returns
    -------
    interpolator : PchipInterpolator
        PCHIP interpolator object

    Notes
    -----
    This interpolator supports extrapolation.

    References
    ----------
    .. [1] `PchipInterpolator
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html>`_
            on SciPy docs.

    """

    return interp.PchipInterpolator(curve.t, curve.data, axis=0, extrapolate=extrapolate)


@register_interpolator_factory(method='spline')
def spline_interpolator_factory(curve: 'Curve', *,
                                w: ty.Optional[np.ndarray] = None,
                                k: int = 3,
                                extrapolate: ty.Union[int, str] = 'extrapolate'):
    """General weighted k-order spline interpolation

    Parameters
    ----------
    curve : Curve
        Curve object
    w : np.ndarray, None
        Weights for spline fitting. Must be positive. If None (default), weights are all equal
    k : int
        Degree of the spline. Must be 1 <= k <= 5
    extrapolate : int, str
        Controls the extrapolation mode for elements not in the interval
        defined by the knot sequence. See [1]_ for details.

    Returns
    -------
    interpolator : callable
        interpolation function

    Notes
    -----
    This interpolator supports extrapolation.

    References
    ----------
    .. [1] `InterpolatedUnivariateSpline
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.html>`_
            on SciPy docs.

    """

    splines = [
        interp.InterpolatedUnivariateSpline(curve.t, values, w=w, k=k, ext=extrapolate, check_finite=False)
        for values in curve.values()
    ]

    def _interpolator(grid):
        interp_data = np.empty((grid.size, curve.ndim))

        for i, spline in enumerate(splines):
            interp_data[:, i] = spline(grid)

        return type(curve)(interp_data, dtype=curve.dtype)

    return _interpolator


def interp_methods() -> ty.List[str]:
    """Returns the list of interpolation methods

    Returns
    -------
    methods : List[str]
        The list of interpolation methods

    """

    return list(_INTERPOLATORS.keys())


def get_interpolator_factory(method: str, **kwargs) -> abc.Callable:
    """Returns the interpolator factory for given method

    Parameters
    ----------
    method : str
        Interpolation method
    kwargs : mapping
        Additional arguments for construct interpolator. It is dependent from method.

    Returns
    -------
    interpolator_factory : callable
        Interpolator factory function

    See Also
    --------
    interp_methods

    Raises
    ------
    NameError : If interpolation method is unknown

    """

    if method not in _INTERPOLATORS:
        raise NameError('Cannot find interpolation method "{}"'.format(method))

    interpolator_factory = _INTERPOLATORS[method]

    if not kwargs:
        return interpolator_factory
    else:
        return functools.partial(interpolator_factory, **kwargs)


def interpolate(curve: 'Curve', ti: InterpPType, method: str, **kwargs) -> 'Curve':
    """Interpolates a n-dimensional curve data using given method and grid

    Parameters
    ----------
    curve : Curve
        Curve object
    ti : np.ndarray, int
        Interpolation grid or the number of points
    method : str
        Interpolation method

    Returns
    -------
    curve : Curve
        Interpolated curve

    Raises
    ------
    ValueError : invalid input data or parameters
    InterpolationError : any computation of interpolation errors

    """

    curve_type = type(curve)

    if not curve:
        warnings.warn('The curve is empty. Interpolation is not possible.', InterpolationWarning)
        return curve_type(ndmin=curve.ndim, dtype=curve.dtype)

    if curve.size == 1:
        raise ValueError('Cannot interpolate curve with single point.')

    if method not in _INTERPOLATORS:
        raise ValueError('Unknown interpolation method "{}"'.format(method))

    interpolator_factory = get_interpolator_factory(method)
    interpolator = interpolator_factory(curve, **kwargs)

    interp_grid = _make_interp_grid(curve, ti)

    try:
        interp_data = interpolator(interp_grid)
    except Exception as err:
        raise InterpolationError('Interpolation has failed: {}'.format(err)) from err

    return curve_type(interp_data, dtype=curve.dtype)
