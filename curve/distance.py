# -*- coding: utf-8 -*-

"""
The module provides distance metrics

"""

import collections.abc as abc
import typing as t
import functools

from scipy.spatial import distance


_METRICS = {
    'braycurtis': distance.braycurtis,
    'canberra': distance.canberra,
    'chebyshev': distance.chebyshev,
    'cityblock': distance.cityblock,
    'correlation': distance.correlation,
    'cosine': distance.cosine,
    'euclidean': distance.euclidean,
    'jensenshannon': distance.jensenshannon,
    'mahalanobis': distance.mahalanobis,
    'minkowski': distance.minkowski,
    'seuclidean': distance.seuclidean,
    'sqeuclidean': distance.sqeuclidean,
    'wminkowski': distance.wminkowski,
}


def known_metrics() -> t.List[str]:
    """Returns the list of well known metric names

    Returns
    -------
    names : List[str]
        The list of well known metrics

    """

    return list(_METRICS.keys())


def get_metric(name: str, **kwargs) -> abc.Callable:
    """Returns metric function for given metric name

    Parameters
    ----------
    name : str
        Metric name
    **kwargs : any
        Additional keyword-arguments for metric

    Returns
    -------
    metric : callable
        Metric callable or partial callable if **kwargs were set

    Raises
    ------
    NameError : If metric name is unknown

    """

    if name not in _METRICS:
        raise NameError('Cannot find metric with name "{}"'.format(name))

    metric = _METRICS[name]

    if not kwargs:
        return metric
    else:
        return functools.partial(metric, **kwargs)
