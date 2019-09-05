# -*- coding: utf-8 -*-

from ._version import __version__  # noqa

from curve._base import Axis, Point, CurvePoint, Curve
from curve._distance import known_metrics, get_metric
from curve._diffgeom import DifferentialGeometryWarning
from curve._interpolate import (
    InterpolationError, InterpolationWarning, make_uniform_interp_grid, interp_methods
)
from curve._numeric import isequal, allequal, dot1d

__all__ = [
    'Axis',
    'Point',
    'CurvePoint',
    'Curve',
    'known_metrics',
    'get_metric',
    'DifferentialGeometryWarning',
    'InterpolationError',
    'InterpolationWarning',
    'make_uniform_interp_grid',
    'interp_methods',
    'isequal',
    'allequal',
    'dot1d',
]
