# -*- coding: utf-8 -*-

"""
This module provides routines for n-dimensional curve interpolation

The following interpolation methods are available out of the box:

    * ``linear`` -- linear interpolation
    * ``cubic`` -- cubic spline interpolation
    * ``hermite`` -- piecewise-cubic interpolation matching values and first derivatives
    * ``akima`` -- Akima interpolation
    * ``pchip`` -- PCHIP 1-d monotonic cubic interpolation
    * ``spline`` -- Smoothing weighted k-order spline interpolation/approximation
    * ``csaps`` -- Smoothing weighted natural cubic spline interpolation/approximation

"""

import typing as ty
from collections import abc
import warnings

import numpy as np
import scipy.interpolate as interp
import csaps

from skcurve._numeric import linrescale

if ty.TYPE_CHECKING:
    from skcurve._base import Curve  # noqa


_interpolators = {}  # type: ty.Dict[str, ty.Type['InterpolatorBase']]


InterpGridSpecType = ty.Union[
    int,
    np.ndarray,
    ty.Sequence[float],
    'InterpolationGrid',
]


class InterpolationError(Exception):
    """Any interpolation errors
    """


class InterpolationWarning(RuntimeWarning):
    """Any interpolation warnings
    """


class InterpolationGrid:
    """Interpolation grid interface

    The interface for all classes which create interpolation grids.

    Examples
    --------

    .. code-block:: python

        class LinspaceGrid(InterpolationGrid):
            def __init__(self, pcount):
                '''The constructor is used for initialization and setup'''
                self.pcount = pcount

            def __call__(self, curve):
                '''The __call__ method is used for creating an interpolation grid'''
                return np.linspace(curve.t[0], curve.t[-1], self.pcount)

        curve = Curve(...)

        grid = LinspaceGrid()
        grid_data = grid(curve)

        # or use it in `Curve.interpolate` method:
        curve_i = curve.interpolate(grid)

    See Also
    --------
    UniformInterpolationGrid
    UniformExtrapolationGrid

    """

    def __call__(self, curve: 'Curve') -> np.ndarray:
        raise NotImplementedError


class UniformInterpolationGrid(InterpolationGrid):
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
    UniformExtrapolationGrid

    """

    def __init__(self, fill: ty.Union[int, float], kind: str = 'point'):
        if kind not in ('point', 'length'):
            raise ValueError(f"Unknown 'kind': {kind}")
        if kind == 'point':
            if not isinstance(fill, int):
                raise TypeError("'fill' must be an integer for 'kind' == 'point'.")
            if fill < 2:
                raise ValueError('There must be at least two interpolation points.')
        elif kind == 'length':
            if not isinstance(fill, (int, float)):
                raise TypeError("'fill' must be an integer or float for 'kind' == 'length'.")

        self.fill = fill
        self.kind = kind

    def __call__(self, curve: 'Curve') -> np.ndarray:
        if curve.size < 2:
            raise ValueError(f'The curve size {curve.size} is too few')

        if self.kind == 'length':
            pcount = int(round(curve.arclen / self.fill) + 1)
        else:
            pcount = int(self.fill)

        return np.linspace(curve.t[0], curve.t[-1], pcount)


class UniformExtrapolationGrid(InterpolationGrid):
    """The helper class to create uniform extrapolation pieces in interpolation grid

    The helper class to create 1-d uniform extrapolation pieces before and after interpolation interval.

    Parameters
    ----------
    interp_grid : InterpolationGrid
        An interpolation grid `InterpolationGrid`-based object.
    before : int, float
        Defines filling for "before" extrapolation piece. It is dependent on 'kind' argument.
    after : int, float
        Defines filling for "after" extrapolation piece. It is dependent on 'kind' argument.
    kind : str
        The kind of 'before' and 'after' parameters for creating a grid
        If 'kind' is "point", 'before'/'after' are the number of points on extrap pieces (strictly).
        if 'kind' is "length", 'before'/'after' are extrap piece lengths (approximately).

    See Also
    --------
    UniformInterpolationGrid

    """

    def __init__(self, interp_grid: UniformInterpolationGrid,
                 before: ty.Union[int, float] = 0,
                 after: ty.Union[int, float] = 0,
                 kind: str = 'point'):
        if kind not in ('point', 'length'):
            raise ValueError(f"Unknown 'kind': {kind}")
        if kind == 'point':
            if not isinstance(before, int) or not isinstance(after, int):
                raise TypeError("'before' and 'after' arguments must be an integer for 'kind' == 'point'.")
        elif kind == 'length':
            if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
                raise TypeError("'before' and 'after' arguments must be an integer or float for 'kind' == 'length'.")

        self.interp_grid = interp_grid
        self.before = before
        self.after = after
        self.kind = kind

    def __call__(self, curve: 'Curve') -> np.ndarray:
        grid = self.interp_grid(curve)
        interp_chordlen = np.diff(grid).mean()

        if self.kind == 'point':
            pcount_before = self.before
            pcount_after = self.after
        else:
            pcount_before = int(round(self.before / interp_chordlen))
            pcount_after = int(round(self.after / interp_chordlen))

        grid_before = np.linspace(
            -interp_chordlen * pcount_before + grid[0], -interp_chordlen + grid[0], pcount_before)

        grid_after = np.linspace(
            grid[-1] + interp_chordlen, grid[-1] + interp_chordlen * pcount_after, pcount_after)

        return np.hstack([grid_before, grid, grid_after])


class PreservedSpeedInterpolationGrid(InterpolationGrid):
    """The helper class for creating interpolation grid with preserving the curve speed function

    This class creates a non-uniform interpolation grid
    that preserve the curve speed function along the curve.

    Absolute values of the speed at every curve point will be decrease
    (or increase) dependent to the number of the grid points.

    Parameters
    ----------
    pcount : int
        The number of points for interpolation grid.
    interp_kind : str
        kind of the speed function interpolation method. By default 'linear'.
        See for details: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html

    Examples
    --------

    .. code-block:: python

        >>> x = np.logspace(0, 1, 10)
        >>> y = np.logspace(0, 1, 10)
        >>> curve = Curve([x, y])
        >>> uniform_grid = UniformInterpolationGrid(fill=30, kind='point')
        >>> speed_grid = PreservedSpeedInterpolationGrid(pcount=30)
        >>> print(curve.t)
        [ 0.          1.1006533   3.05792239  6.53849373 12.72792206]
        >>> print(speed_grid(curve))
        [ 0.          0.50133887  1.14899588  1.94297104  2.92122314  4.15966978
          5.65831096  7.5521494   9.90868643 12.72792206]
        >>> print(uniform_grid(curve))
        [ 0.          1.41421356  2.82842712  4.24264069  5.65685425  7.07106781
          8.48528137  9.89949494 11.3137085  12.72792206]

    See Also
    --------
    UniformInterpolationGrid

    """

    def __init__(self, pcount: int, interp_kind: str = 'linear'):
        self.pcount = pcount
        self.interp_kind = interp_kind

    def __call__(self, curve: 'Curve') -> np.ndarray:
        if not curve.isparametric:
            x = np.linspace(0, 1, curve.size)
            xi = np.linspace(0, 1, self.pcount)

            interpolator = interp.interp1d(x, curve.cumarclen, kind=self.interp_kind)
            cumarclen_i = interpolator(xi)

            # Normalize cumulative chord lengths to the curve parametrization
            grid = linrescale(cumarclen_i, out_range=(0, curve.arclen))
        else:
            # If the curve is parametric we use uniform interpolation grid that preserves speed automatically
            grid = UniformInterpolationGrid(self.pcount, kind='point')(curve)

        return grid


class InterpolatorBase:
    """The base class for all interpolators

    Parameters
    ----------
    curve : Curve
        Curve object
    kwargs : mapping
        An interpolator parameters

    See Also
    --------
    register_interpolator

    """

    def __init__(self, curve: 'Curve', **kwargs):
        self._curve = curve

    def __call__(self, grid_spec: InterpGridSpecType) -> 'Curve':
        curve_type = type(self.curve)

        if not self.curve:
            warnings.warn('The curve is empty. Interpolation is not possible.', InterpolationWarning)
            return curve_type([], ndmin=self.curve.ndim, dtype=self.curve.dtype)

        if self.curve.size == 1:
            raise ValueError('Cannot interpolate curve with single point.')

        grid = self._get_interpolation_grid(grid_spec)

        try:
            inter_data = self._interpolate(grid)
        except Exception as err:
            raise InterpolationError(f'Interpolation has failed: {err}') from err

        if self.curve.isparametric:
            tdata = grid
        else:
            tdata = None

        return curve_type(inter_data, tdata=tdata, dtype=self.curve.dtype)

    @property
    def curve(self) -> 'Curve':
        return self._curve

    def _interpolate(self, grid: np.ndarray) -> np.ndarray:
        """Implements interpolation on the given grid

        This method should implement the interpolation for given grid and curve.

        Parameters
        ----------
        grid : np.ndarray
            The array 1xM with interpolation grid data

        Returns
        -------
        interp_data : np.ndarray
            Interpolated MxN data

        """
        raise NotImplementedError

    def _get_interpolation_grid(self, grid_spec: InterpGridSpecType) -> np.ndarray:
        """Returns interpolation grid data using grid spec and curve object

        Parameters
        ----------
        grid_spec : InterpGridSpecType
            Grid specification

        Returns
        -------
        grid_data : np.ndarray
            Grid data for given spec and curve object

        """

        if isinstance(grid_spec, InterpolationGrid):
            grid = grid_spec(self.curve)
        elif isinstance(grid_spec, int):
            grid = UniformInterpolationGrid(grid_spec)(self.curve)
        elif isinstance(grid_spec, (np.ndarray, abc.Sequence)):
            grid = np.array(grid_spec, dtype=np.float64)
        else:
            raise ValueError(f'Invalid type {type(grid_spec)} of interpolation grid')

        if grid.ndim != 1:
            raise ValueError(
                'The interpolation grid should be 1xM array where M is number of points in interpolated curve')
        if not np.issubdtype(grid.dtype, np.number):
            raise ValueError(f'Invalid dtype {grid.dtype} of interpolation grid')

        dt = np.diff(grid)

        if np.any(dt < 0) or np.any(np.isclose(dt, 0)):
            raise ValueError(
                'The values in the interpolation grid must be strictly increasing ordered.')

        t_start, t_end = self.curve.t[0], self.curve.t[-1]

        if np.min(grid) > t_start or np.max(grid) < t_end:
            warnings.warn(
                f'The interpolation grid in range [{np.min(grid)}, {np.max(grid)}]. '
                f'It does not cover the whole curve parametrization range [{t_start}, {t_end}].',
                InterpolationWarning)

        return grid


def register_interpolator(method: str):
    """Registers an interpolator class

    This decorator can be used for registering custom interpolator classes.

    Parameters
    ----------
    method : str
        Interpolation method

    See Also
    --------
    InterpolatorBase

    """

    def decorator(cls: ty.Type[InterpolatorBase]):
        if method in _interpolators:
            raise ValueError(f"'{method}' interpolation method already registered for {_interpolators[method]}")
        if not issubclass(cls, InterpolatorBase):
            raise TypeError(f"{cls} is not a subclass of 'InterpolatorBase'")
        _interpolators[method] = cls
    return decorator


@register_interpolator(method='linear')
class LinearInterpolator(InterpolatorBase):
    """Linearly interpolates a n-dimensional curve data

    Parameters
    ----------
    curve : Curve
        Curve object
    extrapolate : bool
        Determines whether to extrapolate to out-of-bounds.

    """

    def __init__(self, curve: 'Curve', *, extrapolate: bool = True):
        super().__init__(curve)

        fill_value = None  # type: ty.Optional[str]
        if extrapolate:
            fill_value = 'extrapolate'

        self.linear = interp.interp1d(curve.t, curve.data, kind='linear', fill_value=fill_value, axis=0)

    def _interpolate(self, grid: np.ndarray) -> np.ndarray:
        return self.linear(grid)


@register_interpolator(method='cubic')
class CubicSplineInterpolator(InterpolatorBase):
    """Cubic spline interpolator

    Parameters
    ----------
    curve : Curve
        Curve object
    bc_type : str
        Boundary condition type:
            * 'natural' (default)
            * 'not-a-knot'
            * 'clamped'

        If bc_type is a 2-tuple, the first and the second value will be applied at the
        curve start and end respectively.

        See scipy docs for details [1]_.

    extrapolate : bool, str, None
        If bool, determines whether to extrapolate to out-of-bounds
        points based on first and last intervals, or to return NaNs.
        If ‘periodic’, periodic extrapolation is used.
        If None (default), extrapolate is set to ‘periodic’
        for bc_type='periodic' and to True otherwise.

    References
    ----------
    .. [1] `CubicSpline
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html>`_
            on SciPy docs.

    """

    def __init__(self, curve: 'Curve', *,
                 bc_type: ty.Union[str, ty.Tuple[ty.Any, ty.Any]] = 'natural',
                 extrapolate: ty.Optional[ty.Union[bool, str]] = None):
        super().__init__(curve)

        self.spline = interp.CubicSpline(
            curve.t, curve.data, axis=0, bc_type=bc_type, extrapolate=extrapolate)

    def _interpolate(self, grid: np.ndarray) -> np.ndarray:
        return self.spline(grid)


@register_interpolator(method='hermite')
class CubicHermiteSplineInterpolator(InterpolatorBase):
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

    References
    ----------
    .. [1] `CubicHermiteSpline
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.html>`_
            on SciPy docs.

    """

    def __init__(self, curve: 'Curve', *, extrapolate: ty.Optional[ty.Union[bool, str]] = None):
        super().__init__(curve)

        self.spline = interp.CubicHermiteSpline(
            curve.t, curve.data, curve.frenet1, axis=0, extrapolate=extrapolate)

    def _interpolate(self, grid: np.ndarray) -> np.ndarray:
        return self.spline(grid)


@register_interpolator(method='akima')
class AkimaInterpolator(InterpolatorBase):
    """Akima interpolator

    Parameters
    ----------
    curve : Curve
        Curve object

    Notes
    -----
    This interpolator does not support extrapolation

    References
    ----------
    .. [1] `Akima1DInterpolator
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html>`_
            on SciPy docs.

    """

    def __init__(self, curve: 'Curve'):
        super().__init__(curve)
        self.akima = interp.Akima1DInterpolator(curve.t, curve.data, axis=0)

    def _interpolate(self, grid: np.ndarray) -> np.ndarray:
        return self.akima(grid)


@register_interpolator(method='pchip')
class PchipInterpolator(InterpolatorBase):
    """PCHIP 1-d monotonic cubic interpolation

    Parameters
    ----------
    curve : Curve
        Curve object
    extrapolate : bool, None
        Whether to extrapolate to out-of-bounds points based on first and last intervals,
        or to return NaNs.

    References
    ----------
    .. [1] `PchipInterpolator
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html>`_
            on SciPy docs.

    """

    def __init__(self, curve: 'Curve', *, extrapolate: ty.Optional[bool] = None):
        super().__init__(curve)
        self.pchip = interp.PchipInterpolator(curve.t, curve.data, axis=0, extrapolate=extrapolate)

    def _interpolate(self, grid: np.ndarray) -> np.ndarray:
        return self.pchip(grid)


@register_interpolator(method='spline')
class SplineInterpolator(InterpolatorBase):
    """General weighted k-order smoothing spline interpolation

    Parameters
    ----------
    curve : Curve
        Curve object
    k : int
        Degree of the spline. Must be in the range [1..5]
    smooth : float
        Positive smoothing factor. See [1]_ for details. By default 0 -- k-interpolant.
    weights : np.ndarray, None
        Weights for spline fitting. Must be positive. If None (default), weights are all equal
    extrapolate : int, str
        Controls the extrapolation mode for elements not in the interval
        defined by the knot sequence. See [1]_ for details.

    References
    ----------
    .. [1] `InterpolatedUnivariateSpline
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html>`_
            on SciPy docs.

    """

    def __init__(self, curve: 'Curve', *,
                 k: int = 3,
                 smooth: float = 0.0,
                 weights: ty.Optional[np.ndarray] = None,
                 extrapolate: ty.Union[int, str] = 0):
        super().__init__(curve)

        self.splines = [
            interp.UnivariateSpline(
                curve.t, values, w=weights, k=k, s=smooth, ext=extrapolate, check_finite=False)
            for values in curve.values()
        ]

    def _interpolate(self, grid: np.ndarray) -> np.ndarray:
        interp_data = np.empty((grid.size, self.curve.ndim))

        for i, spline in enumerate(self.splines):
            interp_data[:, i] = spline(grid)

        return interp_data


@register_interpolator(method='csaps')
class CsapsInterpolator(InterpolatorBase):
    """Cubic spline approximation

    Cubic spline approximation using [1]_.

    Parameters
    ----------
    smooth : Optional[float]
        Smoothing parameter in the range [0..1]. See for details [1]_.
        By default ``smooth`` is equal to 1.0 -- cubic-interpolant.
        If ``smooth`` is None, the smoothing parameter will be calculated automatically.
    weights : Optional[np.ndarray]
        Weights for spline fitting. Must be positive. If None (default), weights are all equal.

    References
    ----------
    .. [1] `csaps <https://github.com/espdev/csaps>`_

    """

    def __init__(self, curve: 'Curve',
                 smooth: ty.Optional[float] = 1.0,
                 weights: ty.Optional[np.ndarray] = None):
        super().__init__(curve)
        self.csaps = csaps.CubicSmoothingSpline(
            curve.t, curve.data, weights=weights, smooth=smooth, axis=0)

    def _interpolate(self, grid: np.ndarray) -> np.ndarray:
        return self.csaps(grid)


def interp_methods() -> ty.List[str]:
    """Returns the list of available interpolation methods

    Returns
    -------
    methods : List[str]
        The list of available interpolation methods

    """

    return list(_interpolators.keys())


def get_interpolator(method: str, curve: 'Curve', **params) -> InterpolatorBase:
    """Creates and returns the interpolator instance for given method

    Creates an interpolator instance for given method, curve and parameters.

    Parameters
    ----------
    method : str
        Interpolation method
    curve : Curve
        Curve object
    params : mapping
        The interpolator parameters

    Returns
    -------
    interpolator : InterpolatorBase
        Interpolator instance

    See Also
    --------
    interp_methods

    Raises
    ------
    NameError : If interpolation method is unknown
    InterpolationError : Cannot create interpolator

    """

    if method not in _interpolators:
        raise NameError(f"Cannot find the interpolator for given method '{method}'")

    interpolator_cls = _interpolators[method]

    try:
        return interpolator_cls(curve, **params)
    except Exception as err:
        raise InterpolationError(
            f"Cannot create interpolator '{interpolator_cls}': {err}") from err


def interpolate(curve: 'Curve', grid_spec: InterpGridSpecType, method: str, **params) -> 'Curve':
    """Interpolates a n-dimensional curve data using given method and grid

    Parameters
    ----------
    curve : Curve
        Curve object
    grid_spec : np.ndarray, int, InterpolationGrid
        Interpolation grid specification:
            * The number of points (uniform grid)
            * The array 1xM or Sequence[float]
            * InterpolationGrid-based object
    method : str
        Interpolation method. See `interp_methods`.
    params : mapping
        Additional parameters for given interpolation method

    Returns
    -------
    curve : Curve
        Interpolated curve object

    Raises
    ------
    ValueError : invalid input data or parameters
    InterpolationError : any computation of interpolation errors

    """

    interpolator = get_interpolator(method, curve, **params)
    return interpolator(grid_spec)
