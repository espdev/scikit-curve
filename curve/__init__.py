# -*- coding: utf-8 -*-

from ._version import __version__  # noqa

from curve._base import Axis, Point, CurvePoint, Curve
from curve._distance import known_metrics, get_metric
from curve._diffgeom import DifferentialGeometryWarning
from curve._interpolate import (
    InterpolationError,
    InterpolationWarning,
    InterpolationGrid,
    UniformInterpolationGrid,
    UniformExtrapolationGrid,
    InterpolatorBase,
    interp_methods,
    get_interpolator,
    register_interpolator,
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
    'InterpolationGrid',
    'UniformInterpolationGrid',
    'UniformExtrapolationGrid',
    'InterpolatorBase',
    'interp_methods',
    'get_interpolator',
    'register_interpolator',

    # numeric
    'isequal',
    'allequal',
    'dot1d',
]
