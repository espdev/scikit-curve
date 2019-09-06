# -*- coding: utf-8 -*-

from ._version import __version__  # noqa

from curve._base import Axis, Point, CurvePoint, Curve
from curve._distance import known_metrics, get_metric
from curve._diffgeom import DifferentialGeometryWarning
from curve._interpolate import (
    InterpolationError,
    InterpolationWarning,
    InterpolationUniformGrid,
    interp_methods,
    get_interpolator_factory,
    register_interpolator_factory,
)
from curve._numeric import isequal, allequal, dot1d

__all__ = [
    # base
    'Axis',
    'Point',
    'CurvePoint',
    'Curve',

    # distance
    'known_metrics',
    'get_metric',

    # diffgeom
    'DifferentialGeometryWarning',

    # interpolate
    'InterpolationError',
    'InterpolationWarning',
    'InterpolationUniformGrid',
    'get_interpolator_factory',
    'register_interpolator_factory',
    'interp_methods',

    # numeric
    'isequal',
    'allequal',
    'dot1d',
]
