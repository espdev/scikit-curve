# -*- coding: utf-8 -*-

"""
This module provides routines for n-dimensional curve interpolation

Currently, the following interpolation methods are supported:

    * ``linear`` -- linear interpolation
    * ``cubic`` -- cubic spline interpolation
    * ``hermite`` -- piecewise-cubic interpolation matching values and first derivatives
    * ``akima`` -- Akima interpolation
    * ``pchip`` -- PCHIP 1-d monotonic cubic interpolation
    * ``spline`` -- General k-order spline interpolation

"""

import typing as ty

import numpy as np
import scipy.interpolate as interp

if ty.TYPE_CHECKING:
    from curve._base import Curve


InterpPType = ty.Union[np.ndarray, int]


class InterpolationError(Exception):
    """Any interpolation errors
    """


def _make_interp_grid(curve: 'Curve', t: InterpPType) -> np.ndarray:
    if isinstance(t, int):
        if t < 2:
            raise ValueError('There must be at least two interpolation points.')

        grid = np.linspace(0, curve.t[-1], t)
    else:
        grid = np.array(t, dtype=np.float64)

    if grid.ndim != 1:
        raise ValueError(
            'The interpolation grid should be 1xM array where M is number of points in interpolated curve')

    dt = np.diff(grid)

    if np.any(dt < 0) or np.any(np.isclose(dt, 0)):
        raise ValueError(
            'The values in the interpolation grid must be strictly increasing ordered.')

    return grid


def linear_interpolator(curve: 'Curve'):
    """Linearly interpolates a n-dimensional curve data

    Parameters
    ----------
    curve : Curve
        Curve object

    Returns
    -------
    interpolator : callable
        Linear interpolator function

    """

    def _interpolator(interp_grid: np.ndarray):
        # Normalization and scale interpolation grid to [0..1] range
        interp_grid = np.array(interp_grid)
        interp_grid -= interp_grid[0]
        interp_grid /= interp_grid[-1]

        chordlen_norm = curve.chordlen / curve.arclen
        cumarc_norm = np.hstack((0, np.cumsum(chordlen_norm)))
        tbins = np.digitize(interp_grid, cumarc_norm)

        n = curve.size

        tbins[(tbins < 0)] = 1
        tbins[(tbins >= n) | np.isclose(interp_grid, 1)] = n - 1
        tbins -= 1

        s = (interp_grid - cumarc_norm[tbins]) / chordlen_norm[tbins]

        tbins_data = curve.data[tbins, :]
        tbins1_data = curve.data[tbins + 1, :]

        interp_data = (tbins_data + (tbins1_data - tbins_data) * s[:, np.newaxis])

        return interp_data

    return _interpolator


def cubic_interpolator(curve: 'Curve', bc_type='not-a-knot'):
    """Cubic spline interpolator

    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html

    Parameters
    ----------
    curve : Curve
        Curve object
    bc_type : str
        Boundary condition type

    Returns
    -------
    interpolator : CubicSpline
        cubic spline interpolator object

    """

    return interp.CubicSpline(curve.t, curve.data, axis=0, bc_type=bc_type, extrapolate=False)


def hermite_interpolator(curve: 'Curve'):
    """Piecewise-cubic interpolator matching values and first derivatives

    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.html

    Parameters
    ----------
    curve : Curve
        Curve object

    Returns
    -------
    interpolator : CubicHermiteSpline
        Hermite cubic spline interpolator object

    """

    return interp.CubicHermiteSpline(curve.t, curve.data, curve.frenet1, axis=0, extrapolate=False)


def akima_interpolator(curve: 'Curve'):
    """Akima interpolator

    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html

    Parameters
    ----------
    curve : Curve
        Curve object

    Returns
    -------
    interpolator : Akima1DInterpolator
        Akima spline interpolator object

    """

    return interp.Akima1DInterpolator(curve.t, curve.data, axis=0)


def pchip_interpolator(curve: 'Curve'):
    """PCHIP 1-d monotonic cubic interpolation

    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html

    Parameters
    ----------
    curve : Curve
        Curve object

    Returns
    -------
    interpolator : PchipInterpolator
        PCHIP interpolator object

    """

    return interp.PchipInterpolator(curve.t, curve.data, axis=0, extrapolate=False)


def spline_interpolator(curve: 'Curve', w: ty.Optional[np.ndarray] = None, k: int = 3):
    """General k-order spline interpolation

    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.html

    Parameters
    ----------
    curve : Curve
        Curve object
    w : np.ndarray, None
        Weights for spline fitting. Must be positive. If None (default), weights are all equal
    k : int
        Degree of the spline. Must be 1 <= k <= 5

    Returns
    -------
    interpolator : callable
        interpolation function

    """

    splines = [
        interp.InterpolatedUnivariateSpline(curve.t, y, w=w, k=k, ext=2, check_finite=False)
        for y in curve.values()
    ]

    def _interpolator(grid):
        interp_data = np.empty((grid.size, curve.ndim))

        for i, spline in enumerate(splines):
            interp_data[:, i] = spline(grid)

        return type(curve)(interp_data, dtype=curve.dtype)

    return _interpolator


_INTERPOLATORS = {
    'linear': linear_interpolator,
    'cubic': cubic_interpolator,
    'hermite': hermite_interpolator,
    'akima': akima_interpolator,
    'pchip': pchip_interpolator,
    'spline': spline_interpolator,
}


def interpolate(curve: 'Curve', t: InterpPType, method: str, **kwargs) -> 'Curve':
    """Interpolates a n-dimensional curve data using given method and grid

    Parameters
    ----------
    curve : Curve
        Curve object
    t : np.ndarray, int
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
        return curve_type(ndmin=curve.ndim, dtype=curve.dtype)

    if curve.size == 1:
        raise ValueError('Cannot interpolate curve with single point.')

    if method not in _INTERPOLATORS:
        raise ValueError('Unknown interpolation method "{}"'.format(method))

    interpolator = _INTERPOLATORS[method](curve, **kwargs)
    interp_grid = _make_interp_grid(curve, t)

    try:
        interp_data = interpolator(interp_grid)
    except ValueError:
        raise
    except Exception as err:
        raise InterpolationError('Interpolation has failed') from err

    return curve_type(interp_data, dtype=curve.dtype)


def interp_methods() -> ty.List[str]:
    """Returns the list of interpolation methods

    Returns
    -------
    methods : List[str]
        The list of interpolation methods

    """

    return list(_INTERPOLATORS.keys())
