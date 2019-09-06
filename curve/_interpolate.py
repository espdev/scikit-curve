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


InterpGridType = ty.Union[
    np.ndarray,
    int,
    'InterpolationGrid',
]

_INTERPOLATORS = {}  # type: ty.Dict[str, abc.Callable]


class InterpolationError(Exception):
    """Any interpolation errors
    """


class InterpolationWarning(RuntimeWarning):
    """Any interpolation warning
    """


class InterpolationGrid:
    """Interpolation grid interface
    """

    def __call__(self, curve: 'Curve') -> np.ndarray:
        raise NotImplementedError


class InterpolationUniformGrid(InterpolationGrid):
    """The helper class to create uniform interpolation grid

    The helper class to create 1-d uniform grid that can be used for interpolating curves.

    Parameters
    ----------
    fill : int, float
        Defines filling for interpolation grid. It is dependent on 'kind' argument.
    kind : str
        The kind of fill parameter for creating a grid
        If 'kind' is "point", 'fill' is the number of points (strictly).
        if 'kind' is "length", 'fill' is length of chord (approximately).

    See Also
    --------
    ExtrapolationUniformGrid

    """

    def __init__(self, fill: ty.Union[int, float], kind: str = 'point'):
        if kind not in ('point', 'length'):
            raise ValueError('Unknown "kind": {}'.format(kind))
        if kind == 'point':
            if not isinstance(fill, int):
                raise TypeError('"fill" must be an integer for "kind" == "point".')
            if fill < 2:
                raise ValueError('There must be at least two interpolation points.')
        elif kind == 'length':
            if not isinstance(fill, (int, float)):
                raise TypeError('"fill" must be an integer or float for "kind" == "length".')

        self.fill = fill
        self.kind = kind

    def __call__(self, curve: 'Curve') -> np.ndarray:
        if curve.size < 2:
            raise ValueError('The curve size {} is too few'.format(curve.size))

        if self.kind == 'length':
            pcount = int(curve.arclen / self.fill) + 1
        else:
            pcount = self.fill

        interp_grid = np.linspace(0, curve.arclen, pcount)
        return interp_grid


class ExtrapolationUniformGrid(InterpolationGrid):
    """The helper class to create uniform extrapolation pieces in interpolation grid

    The helper class to create 1-d uniform extrapolation pieces before and after interpolation interval.

    Parameters
    ----------
    interp_grid : InterpUniformGrid
        An interpolation grid constructed object.
    before : int, float
        Defines filling for "before" extrapolation piece. It is dependent on 'kind' argument.
    after : int, float
        Defines filling for "after" extrapolation piece. It is dependent on 'kind' argument.
    kind : str
        The kind of 'before' and 'after' parameters for creating a grid
        If 'kind' is "point", 'before'/`after' are the number of points on extrap pieces (strictly).
        if 'kind' is "length", 'before'/`after' are extrap piece lengths (approximately).

    See Also
    --------
    InterpolationUniformGrid

    """

    def __init__(self, interp_grid: InterpolationUniformGrid,
                 before: ty.Union[int, float] = 0,
                 after: ty.Union[int, float] = 0,
                 kind: str = 'point'):
        if kind not in ('point', 'length'):
            raise ValueError('Unknown "kind": {}'.format(kind))
        if kind == 'point':
            if not isinstance(before, int) or not isinstance(after, int):
                raise TypeError('"before" and "after" arguments must be an integer for "kind" == "point".')
        elif kind == 'length':
            if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
                raise TypeError('"before" and "after" arguments must be an integer or float for "kind" == "length".')

        self.interp_grid = interp_grid
        self.before = before
        self.after = after
        self.kind = kind

    def __call__(self, curve: 'Curve') -> np.ndarray:
        grid = self.interp_grid(curve)
        interp_chordlen = grid[1] - grid[0]

        if self.kind == 'point':
            pcount_before = self.before
            pcount_after = self.after
        else:
            pcount_before = int(self.before / interp_chordlen)
            pcount_after = int(self.after / interp_chordlen)

        grid_before = np.linspace(
            -interp_chordlen * pcount_before, -interp_chordlen, pcount_before)

        grid_after = np.linspace(
            grid[-1] + interp_chordlen, grid[-1] + interp_chordlen * pcount_after, pcount_after)

        return np.hstack([grid_before, grid, grid_after])


def _make_interp_grid(curve: 'Curve', pcount_or_grid: InterpGridType) -> np.ndarray:
    if isinstance(pcount_or_grid, InterpolationGrid):
        grid = pcount_or_grid(curve)
    elif isinstance(pcount_or_grid, int):
        grid = InterpolationUniformGrid(pcount_or_grid)(curve)
    else:
        grid = np.array(pcount_or_grid, dtype=np.float64)

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

    def _interpolator(interp_grid: np.ndarray):
        if not extrapolate:
            if np.min(interp_grid) < 0 or np.max(interp_grid) > curve.arclen:
                warnings.warn((
                    '"extrapolate" is disabled but interpolation grid is out-of-bounds curve data. '
                    'The grid will be cut down to curve data bounds.'
                ), InterpolationWarning)

                drop_indices = np.flatnonzero((interp_grid < 0) | (interp_grid > curve.arclen))
                interp_grid = np.delete(interp_grid, drop_indices)

        tbins = np.digitize(interp_grid, curve.t)

        n = curve.size

        tbins[(tbins <= 0)] = 1
        tbins[(tbins >= n) | np.isclose(interp_grid, 1)] = n - 1
        tbins -= 1

        s = (interp_grid - curve.t[tbins]) / curve.chordlen[tbins]

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


def interpolate(curve: 'Curve', pcount_or_grid: InterpGridType, method: str, **kwargs) -> 'Curve':
    """Interpolates a n-dimensional curve data using given method and grid

    Parameters
    ----------
    curve : Curve
        Curve object
    pcount_or_grid : np.ndarray, int, InterpolationUniformGrid, ExtrapolationUniformGrid
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

    interp_grid = _make_interp_grid(curve, pcount_or_grid)

    try:
        interp_data = interpolator(interp_grid)
    except Exception as err:
        raise InterpolationError('Interpolation has failed: {}'.format(err)) from err

    return curve_type(interp_data, dtype=curve.dtype)
