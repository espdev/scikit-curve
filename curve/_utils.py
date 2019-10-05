# -*- coding: utf-8 -*-

import numpy as np


def as2d(arr: np.ndarray, axis: int) -> np.ndarray:
    """Transforms the shape of N-D array-like to 2-D MxN array

    The function transforms N-D array-like to 2-D MxN array along given axis,
    where M is the number of elements and N is dimension.

    Parameters
    ----------
    arr : np.ndarray, array-like
        N-D array or array-like object
    axis : int
        Axis that will be used for transform array shape

    Returns
    -------
    arr2d : np.ndarray
        2-D MxN array

    Raises
    ------
    ValueError : axis is out of array axes

    """

    arr = np.asarray(arr)

    if arr.size == 0:
        return np.reshape([], (0, 2)).astype(arr.dtype)

    orig_axis = axis
    axis = arr.ndim + axis if axis < 0 else axis

    if axis >= arr.ndim:
        raise ValueError('axis {} is out of array shape {}.'.format(orig_axis, arr.shape))

    tr_axes = list(range(arr.ndim))
    tr_axes.insert(0, tr_axes.pop(axis))
    new_shape = (arr.shape[axis], np.prod(arr.shape) // arr.shape[axis])

    return arr.transpose(tr_axes).reshape(new_shape)
