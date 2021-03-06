# -*- coding: utf-8 -*-

"""
This module provides routines for smoothing n-dimensional curves

"""

import typing as ty
import typing_extensions as ty_ext
import collections.abc as abc

import numpy as np
import scipy.signal as signal
import scipy.signal.windows as windows
import scipy.ndimage as ndimage

if ty.TYPE_CHECKING:
    from skcurve._base import Curve  # noqa


_smooth_methods = {}  # type: ty.Dict[str, 'SmoothingMethod']


class SmoothingError(Exception):
    """Any smoothing errors
    """


class SmoothingMethod(ty_ext.Protocol):
    """Defines smoothing method callable protocol type

    Callable signature::

        (curve: Curve, **params: Any) -> numpy.ndarray

    Parameters

        - curve : The curve object
        - **params : Additional any key-word parameters

    Returns

        - array : MxN array with smoothing results

    See Also
    --------
    register_smooth_method

    """

    def __call__(self, curve: 'Curve', **params: ty.Any) -> np.ndarray: ...


def register_smooth_method(method: str):
    """Registers new smoothing method

    Parameters
    ----------
    method : str
        The smoothing method name

    See Also
    --------
    smooth_methods
    SmoothingMethod

    """

    def decorator(method_callable: SmoothingMethod):
        if method in _smooth_methods:
            raise ValueError(
                f"'{method}' smoothing method already registered for {_smooth_methods[method]}")
        _smooth_methods[method] = method_callable

    return decorator


def smooth_methods() -> ty.List[str]:
    """Returns the list of available smoothing methods

    Returns
    -------
    methods : List[str]
        The list of available smoothing methods

    See Also
    --------
    get_smooth_method

    """

    return list(_smooth_methods.keys())


def get_smooth_method(method: str) -> SmoothingMethod:
    """Creates and returns the smoothing method for the given name

    Parameters
    ----------
    method : str
        Smoothing method

    Returns
    -------
    smooth_method : Callable
        Smoothing method callable

    See Also
    --------
    smooth_methods

    Raises
    ------
    NameError : If smooth method is unknown

    """

    if method not in _smooth_methods:
        raise NameError(f"Cannot find the smoothing method '{method}'")

    return _smooth_methods[method]


@register_smooth_method('savgol')
def savgol_filter(curve: 'Curve', window_length: int, polyorder: int, *,
                  deriv: int = 0, delta: float = 1.0,
                  mode: str = 'interp', cval: float = 0.0) -> np.ndarray:
    """Savitzky-Golay smoothing filter [1]_

    References
    ----------
    .. [1] `Savitzky-Golay filter
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html#scipy.signal.savgol_filter>`_
    """

    return signal.savgol_filter(
        curve.data,
        window_length=window_length,
        polyorder=polyorder,
        deriv=deriv,
        delta=delta,
        mode=mode,
        cval=cval,
        axis=0,
    )


@register_smooth_method('window')
def window_filter(curve: 'Curve',
                  window_size: int, window_type: ty.Union[str, abc.Callable] = 'hann',
                  mode: str = 'reflect', cval: float = 0.0) -> np.ndarray:
    """Smoothes a curve using moving average filter with the given window [1]_

    References
    ----------
    .. [1] `The windows in scipy
           <https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows>`_

    """

    if callable(window_type):
        try:
            window = window_type(window_size)
        except Exception as err:
            raise ValueError(
                f'Cannot create the window using {window_type}: {err}') from err
    else:
        window = windows.get_window(window_type, window_size, fftbins=False)

    window /= window.sum()

    return ndimage.convolve1d(
        curve.data,
        weights=window,
        mode=mode,
        cval=cval,
        axis=0,
    )


def smooth(curve: 'Curve', method: str, **params) -> 'Curve':
    """Smoothes a n-dimensional curve using the given method and its parameters

    Parameters
    ----------
    curve : Curve
        A curve object
    method : str
        Smoothing method
    params : mapping
        The parameters of smoothing method

    Returns
    -------
    curve : Curve
        Smoothed curve with type `numpy.float64`

    Raises
    ------
    ValueError : Input data or parameters have invalid values
    TypeError : Input data or parameters have invalid type
    SmoothingError : Smoothing has failed

    See Also
    --------
    smooth_methods

    """

    smooth_method = get_smooth_method(method)

    try:
        smoothed_data = smooth_method(curve, **params)
    except (ValueError, TypeError):
        raise
    except Exception as err:
        raise SmoothingError(f'Smoothing has failed: {err}') from err

    return type(curve)(smoothed_data)
