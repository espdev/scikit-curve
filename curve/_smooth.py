# -*- coding: utf-8 -*-

"""
This module provides routines for smoothing n-dimensional curves

"""

import typing as ty
import collections.abc as abc

import numpy as np
import scipy.signal as signal
import scipy.signal.windows as windows
import scipy.ndimage as ndimage

if ty.TYPE_CHECKING:
    from curve._base import Curve


class SmoothingError(Exception):
    """Any smoothing errors
    """


_smoothing_filters = {}


def register_smooth_filter(method: str):
    def decorator(filter_callable):
        if method in _smoothing_filters:
            raise ValueError('"{}" smoothing method already registered for {}'.format(
                method, _smoothing_filters[method]))
        _smoothing_filters[method] = filter_callable
    return decorator


@register_smooth_filter('savgol')
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


@register_smooth_filter('window')
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
                'Cannot create the window using {}: {}'.format(window_type, err)) from err
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


def smooth_methods() -> ty.List[str]:
    """Returns the list of available smoothing methods

    Returns
    -------
    methods : List[str]
        The list of available smoothing methods

    """

    return list(_smoothing_filters.keys())


def get_smooth_filter(method: str) -> abc.Callable:
    """Creates and returns the smoothing filter for the given method

    Parameters
    ----------
    method : str
        Smoothing method

    Returns
    -------
    smooth_filter : Callable
        Smoothing filter callable

    See Also
    --------
    smooth_methods

    Raises
    ------
    NameError : If smooth method is unknown

    """

    if method not in _smoothing_filters:
        raise NameError('Cannot find the smoothing filter for given method "{}"'.format(method))

    return _smoothing_filters[method]


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

    smooth_filter = get_smooth_filter(method)

    try:
        smoothed_data = smooth_filter(curve, **params)
    except (ValueError, TypeError):
        raise
    except Exception as err:
        raise SmoothingError('Smoothing has failed: {}'.format(err)) from err

    return type(curve)(smoothed_data)
