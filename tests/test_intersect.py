# -*- coding: utf-8 -*-

import functools
import pytest

import numpy as np

from skcurve import Point, Segment, Curve, CurveSegment, curves


skip = functools.partial(pytest.param, marks=pytest.mark.skip)


@pytest.mark.parametrize('segment1, segment2, intersect_point', [
    # ----------------------
    # 2D

    # intersected
    (Segment(Point([1, 1]), Point([2, 2])),
     Segment(Point([1, 2]), Point([2, 1])), Point([1.5, 1.5])),
    # intersected perpendicular
    (Segment(Point([0, 0]), Point([0, 2])),
     Segment(Point([-1, 1]), Point([1, 1])), Point([0, 1])),
    # not intersected parallel
    (Segment(Point([1, 2]), Point([2, 2])),
     Segment(Point([1, 1]), Point([2, 1])), None),
    # not intersected collinear
    (Segment(Point([0, 0]), Point([1, 1])),
     Segment(Point([1.1, 1.1]), Point([2, 2])), None),
    # overlapped #1
    (Segment(Point([1, 1]), Point([2, 2])),
     Segment(Point([0.5, 0.5]), Point([1.5, 1.5])), Point([1.25, 1.25])),
    # overlapped #2
    (Segment(Point([1, 1]), Point([2, 2])),
     Segment(Point([0.5, 0.5]), Point([2.5, 2.5])), Point([1.5, 1.5])),
    # not intersected two singular
    (Segment(Point([2, 2]), Point([2, 2])),
     Segment(Point([1, 1]), Point([1, 1])), None),
    # not intersected one singular
    (Segment(Point([1, 1]), Point([2, 2])),
     Segment(Point([-1, 1.5]), Point([-1, 1.5])), None),
    # intersected two singular
    (Segment(Point([0, 0]), Point([0, 0])),
     Segment(Point([0, 0]), Point([0, 0])), Point([0, 0])),
    # intersected one singular
    (Segment(Point([1, 1]), Point([2, 2])),
     Segment(Point([1.5, 1.5]), Point([1.5, 1.5])), Point([1.5, 1.5])),

    # ----------------------
    # 3D

    # intersected
    (Segment(Point([1, 1, 1]), Point([2, 2, 2])),
     Segment(Point([1, 2, 1]), Point([2, 1, 2])), Point([1.5, 1.5, 1.5])),
    # intersected perpendicular
    (Segment(Point([0, 0, 0]), Point([0, 0, 2])),
     Segment(Point([-1, 0, 1]), Point([1, 0, 1])), Point([0, 0, 1])),
    # not intersected two singular # 1
    (Segment(Point([2, 2, 2]), Point([2, 2, 2])),
     Segment(Point([1, 1, 1]), Point([1, 1, 1])), None),
    # not intersected two singular # 2
    (Segment(Point([-5, 1, 2]), Point([-5, 1, 2])),
     Segment(Point([3, -4, 1]), Point([3, -4, 1])), None),
    # not intersected one singular
    (Segment(Point([1, 1, 1]), Point([2, 2, 2])),
     Segment(Point([-2, 5, 1.5]), Point([-2, 5, 1.5])), None),
    # intersected two singular
    (Segment(Point([1, 1, 1]), Point([1, 1, 1])),
     Segment(Point([1, 1, 1]), Point([1, 1, 1])), Point([1, 1, 1])),
    # intersected one singular
    (Segment(Point([1, 1, 1]), Point([2, 2, 2])),
     Segment(Point([1.5, 1.5, 1.5]), Point([1.5, 1.5, 1.5])), Point([1.5, 1.5, 1.5])),
])
def test_intersect_segments(segment1, segment2, intersect_point):
    intersection = segment1.intersect(segment2)
    assert intersection.intersect_point == intersect_point


@pytest.mark.parametrize('data1, data2, segments1, segments2, intersect_points', [
    # ----------------------
    # 2D

    # no intersections
    ([(1, 2, 3, 4), (1, 2, 3, 4)], [(4, 5, 6, 7), (7, 6, 5, 4)], [], [], []),
    # parallel, no intersections
    ([(1, 2, 3, 4), (1, 2, 3, 4)], [(1.5, 2.5, 3.5, 4.5), (0.5, 1.5, 2.5, 3.5)], [], [], []),
    # 1 pt
    ([(1, 2, 3, 4), (1, 2, 3, 4)], [(2, 3, 4, 5), (5, 4, 3, 2)], [2], [1], [Point([3.5, 3.5])]),
    # overlap 1 pt
    ([(1, 2), (1, 2)], [(0, 3), (0, 3)], [0], [0], [Point([1.5, 1.5])]),
    # overlap 1 pt
    ([(1, 2), (1, 2)], [(2, 3), (2, 3)], [0], [0], [Point([2, 2])]),
    # overlap 3 pt
    ([(1, 2, 3), (1, 2, 3)], [(0, 2.5, 4), (0, 2.5, 4)], [0, 1, 1], [0, 0, 1], [Point([1.5, 1.5]),
                                                                                Point([2.25, 2.25]),
                                                                                Point([2.75, 2.75])]),
    # overlap 5 pt
    ([(1, 2, 3, 4), (1, 2, 3, 4)],
     [(0.5, 2.5, 3.5, 4.5), (0.5, 2.5, 3.5, 4.5)], [0, 1, 1, 2, 2], [0, 0, 1, 1, 2], [Point([1.5, 1.5]),
                                                                                      Point([2.25, 2.25]),
                                                                                      Point([2.75, 2.75]),
                                                                                      Point([3.25, 3.25]),
                                                                                      Point([3.75, 3.75])]),
    # orthogonal curves
    ([(2.5, 2.5, 2.5, 2.5), (1, 2, 3, 4)], [(1, 2, 3, 4), (2.5, 2.5, 2.5, 2.5)], [1], [1], [Point([2.5, 2.5])]),
    # self intersect
    ([(1, 2, 3, 3, 2.5, 2.5, 2.5, 2.5), (1, 1, 1, 2, 2, 1.5, 0.5, 0)], None, [1], [5], [Point([2.5, 1.0])]),
    ([(1, 2, 3, 2, 1), (1, 0, 1, 3, 1)], None, [0], [3], [Point([1.0, 1.0])]),

    # ----------------------
    # 3D

    # no intersections (parallel lines: X and Y are equal, Z different)
    ([(1, 2), (1, 2), (1, 2)], [(1, 2), (1, 2), (2, 3)], [], [], []),
    # no intersections (parallel lines: X and Y are equal, Z different)
    ([(1, 2), (1, 2), (1, 2)], [(0.8, 1), (1, 2), (1, 2)], [], [], []),
    # no intersections
    ([(1, 2), (1, 2), (1, 2)], [(1.1, 2.1), (1, 2.5), (1, 2)], [], [], []),
    # no intersections (skewness)
    ([(1, 2), (1, 2), (1, 2)], [(0.5, 2), (2.5, 1), (1, 2)], [], [], []),
    # no intersections (no skewness, intersection is out of segments)
    ([(1, 2), (1, 2), (1, 2)], [(0.8, 1), (1, 2), (1, 2)], [], [], []),
    # no intersections (skewness)
    ([(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)],
     [(1, 2, 3, 4), (4, 3, 2, 1), (2, 3, 4, 5)], [], [], []),
    # 1 pt
    ([(1, 2), (1, 2), (1, 2)],
     [(0.5, 2), (0.5, 2), (1, 2)], [0], [0], [Point([2.0, 2.0, 2.0])]),
    # 1 pt
    ([(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)],
     [(1, 2, 3, 4), (4, 3, 2, 1), (1, 2, 3, 4)], [1], [1], [Point([2.5, 2.5, 2.5])]),
    # overlap 1 pt
    ([(1, 2), (1, 2), (2, 2)],
     [(1, 2), (1, 2), (2, 2)], [0], [0], [Point([1.5, 1.5, 2.0])]),
    # overlap 1 pt
    ([(1, 2), (2, 2), (1, 2)],
     [(1, 2), (2, 2), (1, 2)], [0], [0], [Point([1.5, 2.0, 1.5])]),
    # overlap 1 pt
    ([(1, 2), (1, 2), (1, 2)],
     [(0.5, 2.5), (0.5, 2.5), (0.5, 2.5)], [0], [0], [Point([1.5, 1.5, 1.5])]),
    # overlap 5 pt
    ([(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)],
     [(0.5, 2.5, 3.5, 4.5), (0.5, 2.5, 3.5, 4.5), (0.5, 2.5, 3.5, 4.5)],
     [0, 1, 1, 2, 2], [0, 0, 1, 1, 2], [Point([1.5, 1.5, 1.5]),
                                        Point([2.25, 2.25, 2.25]),
                                        Point([2.75, 2.75, 2.75]),
                                        Point([3.25, 3.25, 3.25]),
                                        Point([3.75, 3.75, 3.75])]),
])
def test_intersect_curves(data1, data2, segments1, segments2, intersect_points):
    if data2:
        curve1 = Curve(data1)
        curve2 = Curve(data2)

        intersections = curve1.intersect(curve2)
    else:
        curve1 = Curve(data1)
        curve2 = curve1

        intersections = curve1.intersect()

    assert len(intersections) == len(segments1)

    for i, intersection in enumerate(intersections):
        assert CurveSegment(curve1, index=segments1[i]) == intersection.segment1
        assert CurveSegment(curve2, index=segments2[i]) == intersection.segment2
        assert intersect_points[i] == intersection.intersect_point


def test_intersect_curves_almost():
    curve1 = curves.helix(t_start=-np.pi, t_stop=np.pi * 3, p_count=100)
    curve2 = curves.helix(t_start=-np.pi, t_stop=np.pi * 3, a=-1, b=-1, p_count=100)

    intersections = curve1.intersect(curve2, method='almost', dist_tol=0.01)

    assert len(intersections) == 2

    expected_intersect_points = [
        Point([-0.9986155794073, -0.0000159984365, -1.5708163062868]),
        Point([0.9986155794073, 0.0000159984365, 1.5708163062868]),
    ]

    for intersection, expected_point in zip(intersections, expected_intersect_points):
        assert intersection.intersect_point == expected_point


@pytest.mark.parametrize('curve1, curve2, dist_tol, extra_tol, num_ints', [
    (curves.helix(p_count=456), curves.helix(a=-1, b=-1, p_count=321), 0.001, 0.1, 6),
    (curves.helix(p_count=456), curves.irregular_helix(p_count=321), 0.01, 0.1, 1),
])
def test_intersect_curves_almost_remove_extra(curve1, curve2, dist_tol, extra_tol, num_ints):
    intersections = curve1.intersect(curve2, method='almost', dist_tol=dist_tol, extra_tol=extra_tol, remove_extra=True)
    assert len(intersections) == num_ints
