# -*- coding: utf-8 -*-

import pytest
from curve import Curve, Point


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
    ([(1, 2), (1, 2), (1, 2)],
     [(1, 2), (1, 2), (2, 3)], [], [], []),
    # no intersections
    ([(1, 2), (1, 2), (1, 2)],
     [(1.1, 2.1), (1, 2.5), (1, 2)], [], [], []),
    # no intersections
    ([(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)],
     [(1, 2, 3, 4), (4, 3, 2, 1), (2, 3, 4, 5)], [], [], []),
    # 1 pt
    ([(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)],
     [(1, 2, 3, 4), (4, 3, 2, 1), (1, 2, 3, 4)], [1], [1], [Point([2.5, 2.5, 2.5])]),
    # overlap 5 pt
    ([(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)],
     [(0.5, 2.5, 3.5, 4.5), (0.5, 2.5, 3.5, 4.5), (0.5, 2.5, 3.5, 4.5)],
     [0, 1, 1, 2, 2], [0, 0, 1, 1, 2], [Point([1.5, 1.5, 1.5]),
                                        Point([2.25, 2.25, 2.25]),
                                        Point([2.75, 2.75, 2.75]),
                                        Point([3.25, 3.25, 3.25]),
                                        Point([3.75, 3.75, 3.75])]),
])
def test_curves_intersect(data1, data2, segments1, segments2, intersect_points):
    if data2:
        curve1 = Curve(data1)
        curve2 = Curve(data2)

        intersections = curve1.intersect(curve2)
    else:
        curve = Curve(data1)
        intersections = curve.intersect()

    assert len(intersections) == len(segments1)

    for i, intersection in enumerate(intersections):
        assert segments1[i] == intersection.segment1.idx
        assert segments2[i] == intersection.segment2.idx
        assert intersect_points[i] == intersection.intersect_point
