# -*- coding: utf-8 -*-

"""
The module provides numeric routines

"""

import typing as ty
import numpy as np


def isequal(obj1: np.ndarray, obj2: np.ndarray, **kwargs) -> np.ndarray:
    """Returns a boolean array where two arrays are element-wise equal

    Notes
    -----
    int/float dtype independent equal check

    Parameters
    ----------
    obj1 : np.ndarray
        The first object
    obj2 : np.ndarray
        The second object
    kwargs : dict
        Additional arguments for equal function

    Returns
    -------
    res : np.ndarray
        Result array

    """

    if np.issubdtype(obj1.dtype, np.integer) and np.issubdtype(obj2.dtype, np.integer):
        cmp = np.equal
    else:
        cmp = np.isclose

    return cmp(obj1, obj2, **kwargs)


def allequal(obj1: np.ndarray, obj2: np.ndarray, axis: ty.Optional[int] = None, **kwargs) -> np.ndarray:
    """Test whether all array elements along a given axis evaluate to True.

    Parameters
    ----------
    obj1 : np.ndarray
        The first object
    obj2 : np.ndarray
        The second object
    axis : int, None
        Axis for test equal. By default None
    kwargs : dict
        Additional arguments for equal function

    Returns
    -------
    res : np.ndarray
        The result array

    """

    return np.all(isequal(obj1, obj2, **kwargs), axis=axis)


def dot1d(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
    """Computes row-wise dot product of two MxN arrays

    Parameters
    ----------
    data1 : np.ndarray
        The first MxN array
    data2 : np.ndarray
        The second MxN array

    Returns
    -------
    res : np.ndarray
        The array 1xM with row-wise dot product result

    """

    return np.einsum('ij,ij->i', data1, data2)


def linrescale(in_data: np.ndarray,
               in_range: ty.Optional[ty.Tuple[float, float]] = None,
               out_range: ty.Optional[ty.Tuple[float, float]] = None,
               out_dtype: ty.Optional[np.dtype] = None) -> np.ndarray:
    """Linearly transforms values from input range to output range

    Parameters
    ----------
    in_data : array-like
        Input data
    in_range : list-like
        Input range. Tuple of two items: ``[min, max]``. By default: ``[min(in_data), max(in_data)]``
    out_range : list-like
        Output range. Tuple of two items: ``[min max]``. By default: ``[0, 1]``
    out_dtype : numpy.dtype
        Output data type. By default ``numpy.float64``

    Returns
    -------
    out_data : numpy.ndarray
        Transformed data

    Examples
    --------

    .. code-block:: python

        >>> import numpy as np
        >>>
        >>> data = np.arange(0, 11)
        >>> out = linrescale(data)
        >>> print out
        array([ 0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])

    """

    in_data = np.asarray(in_data, dtype=np.float64)

    if in_range is None:
        in_range = (np.min(in_data), np.max(in_data))
    if out_range is None:
        out_range = (0, 1)

    in_data = (in_data - in_range[0]) / (in_range[1] - in_range[0])
    out_data = in_data * (out_range[1] - out_range[0]) + out_range[0]

    if out_dtype is not None:
        out_data = out_data.astype(out_dtype)
    return out_data
