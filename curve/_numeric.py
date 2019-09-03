# -*- coding: utf-8 -*-

"""
The module provides numeric routines

"""

import typing as t
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


def allequal(obj1: np.ndarray, obj2: np.ndarray, axis: t.Optional[int] = None, **kwargs) -> np.ndarray:
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
