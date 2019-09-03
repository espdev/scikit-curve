# -*- coding: utf-8 -*-

from ._version import __version__  # noqa

from curve._base import Axis, Point, CurvePoint, Curve
from curve._distance import known_metrics, get_metric
from curve._diffgeom import DifferentialGeometryWarning
from curve._numeric import isequal, allequal, rowdot

__all__ = [
    'Axis',
    'Point',
    'CurvePoint',
    'Curve',
    'known_metrics',
    'get_metric',
    'DifferentialGeometryWarning',
    'isequal',
    'allequal',
    'rowdot',
]
