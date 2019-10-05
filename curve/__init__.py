# -*- coding: utf-8 -*-

from ._version import __version__  # noqa

from curve._base import Axis, Point, Segment, CurvePoint, CurveSegment, Curve
from curve._distance import known_metrics, get_metric
from curve._diffgeom import DifferentialGeometryWarning
from curve._interpolate import (
    InterpolationError,
    InterpolationWarning,
    InterpolationGrid,
    UniformInterpolationGrid,
    UniformExtrapolationGrid,
    PreservedSpeedInterpolationGrid,
    InterpolatorBase,
    interp_methods,
    get_interpolator,
    register_interpolator,
)
from curve._smooth import (
    SmoothingError,
    smooth_methods,
    get_smooth_filter,
    register_smooth_filter,
)
from curve._intersect import SegmentsIntersection, NotIntersected
from curve._numeric import isequal, allequal, dot1d

__all__ = [
    # base
    'Axis',
    'Point',
    'Segment',
    'CurvePoint',
    'CurveSegment',
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
    'PreservedSpeedInterpolationGrid',
    'InterpolatorBase',
    'interp_methods',
    'get_interpolator',
    'register_interpolator',

    # smoothing
    'SmoothingError',
    'smooth_methods',
    'get_smooth_filter',
    'register_smooth_filter',

    # intersection
    'SegmentsIntersection',
    'NotIntersected',

    # numeric
    'isequal',
    'allequal',
    'dot1d',
]
