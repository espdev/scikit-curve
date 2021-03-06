# -*- coding: utf-8 -*-

from ._version import __version__  # noqa

from skcurve._base import Axis, Point, Segment, CurvePoint, CurveSegment, Curve
from skcurve._distance import known_metrics, get_metric
from skcurve._diffgeom import DifferentialGeometryWarning
from skcurve._interpolate import (
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
from skcurve._smooth import (
    SmoothingError,
    smooth_methods,
    get_smooth_method,
    register_smooth_method,
)
from skcurve._geomalg import GeometryAlgorithmsWarning
from skcurve._intersect import (
    IntersectionInfo,
    IntersectionType,
    SegmentsIntersection,
    NOT_INTERSECTED,
    intersect_methods,
    get_intersect_method,
    default_intersect_method,
    set_default_intersect_method,
    register_intersect_method,
)
from skcurve._numeric import isequal, allequal, dot1d

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
    'get_smooth_method',
    'register_smooth_method',

    # geomalgo
    'GeometryAlgorithmsWarning',

    # intersection
    'IntersectionInfo',
    'IntersectionType',
    'SegmentsIntersection',
    'NOT_INTERSECTED',
    'intersect_methods',
    'get_intersect_method',
    'default_intersect_method',
    'set_default_intersect_method',
    'register_intersect_method',

    # numeric
    'isequal',
    'allequal',
    'dot1d',
]
