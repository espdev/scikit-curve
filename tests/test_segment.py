# -*- coding: utf-8 -*-

import pytest
import numpy as np

from curve import Point, Segment, Curve, GeometryAlgorithmsWarning


@pytest.mark.parametrize('p1, p2, ndim', [
    (Point([1, 1]), Point([2, 2]), 2),
    (Point([1, 1, 1]), Point([2, 2, 2]), 3),
])
def test_create_segment(p1, p2, ndim):
    segment = Segment(p1, p2)
    assert segment.ndim == ndim
    assert segment.p1 == p1
    assert segment.p2 == p2


def test_equal_segments():
    segment1 = Segment(Point([1, 1]), Point([2, 2]))
    segment2 = Segment(Point([1, 1]), Point([2, 2]))
    segment3 = Segment(Point([2, 2]), Point([1, 1]))

    assert segment1 == segment2
    assert segment1 != segment3


def test_singular():
    segment = Segment(Point([1, 1]), Point([1, 1]))
    assert segment.singular


@pytest.mark.parametrize('t, expected_point', [
    (0., Point([1, 1])),
    (1., Point([2, 2])),
    (0.5, Point([1.5, 1.5])),
])
def test_segment_point(t, expected_point):
    segment = Segment(Point([1, 1]), Point([2, 2]))
    assert segment.point(t) == expected_point


@pytest.mark.parametrize('point, expected_t', [
    (Point([1, 1]), 0.),
    (Point([2, 2]), 1.),
    (Point([1.5, 1.5]), 0.5),
])
def test_segment_t_2d(point, expected_t):
    segment = Segment(Point([1, 1]), Point([2, 2]))
    assert segment.t(point) == pytest.approx(expected_t)


@pytest.mark.parametrize('point, expected_t', [
    (Point([1, 1, 1]), 0.),
    (Point([2, 2, 2]), 1.),
    (Point([1.5, 1.5, 1.5]), 0.5),
])
def test_segment_t_3d(point, expected_t):
    segment = Segment(Point([1, 1, 1]), Point([2, 2, 2]))
    assert segment.t(point) == pytest.approx(expected_t)


@pytest.mark.parametrize('points1, points2, expected_angle', [
    # 2d
    ((Point([1, 1]), Point([2, 2])), (Point([0, 0]), Point([3, 3])), 0.),
    ((Point([1, 1]), Point([2, 2])), (Point([3, 3]), Point([2, 2])), np.pi),
    ((Point([1, 1]), Point([2, 2])), (Point([3, 3]), Point([0, 0])), np.pi),
    ((Point([1, 1]), Point([1, 2])), (Point([3, 3]), Point([5, 3])), np.pi / 2),
    ((Point([1, 1]), Point([1, 2])), (Point([0, 0]), Point([2, 2])), np.pi / 4),
    ((Point([1, 1]), Point([1, 2])), (Point([2, 2]), Point([0, 0])), 3 * np.pi / 4),

    # 3d
    ((Point([1, 1, 1]), Point([2, 2, 2])), (Point([0, 0, 0]), Point([3, 3, 3])), 0.),
    ((Point([1, 1, 1]), Point([2, 2, 2])), (Point([3, 3, 3]), Point([2, 2, 2])), np.pi),
    ((Point([1, 1, 1]), Point([2, 2, 2])), (Point([3, 3, 3]), Point([0, 0, 0])), np.pi),
    ((Point([1, 2, 1]), Point([1, 2, 2])), (Point([1, 2, 2]), Point([1, 2, 1])), np.pi),
    ((Point([1, 1, 1]), Point([1, 2, 1])), (Point([3, 3, 1]), Point([5, 3, 1])), np.pi / 2),
    ((Point([1, 1, 1]), Point([1, 2, 1])), (Point([0, 0, 1]), Point([2, 2, 1])), np.pi / 4),
    ((Point([1, 1, 1]), Point([1, 2, 1])), (Point([2, 2, 1]), Point([0, 0, 1])), 3 * np.pi / 4),
])
def test_angle(points1, points2, expected_angle):
    segment1 = Segment(*points1)
    segment2 = Segment(*points2)

    assert segment1.angle(segment2) == pytest.approx(expected_angle)


def test_angle_nan():
    segment1 = Segment(Point([1, 1]), Point([1, 2]))
    segment2 = Segment(Point([0, 0]), Point([0, 0]))

    with pytest.warns(GeometryAlgorithmsWarning):
        assert np.isnan(segment1.angle(segment2))


@pytest.mark.parametrize('points1, points2, expected_flag', [
    # 2d
    ((Point([1, 1]), Point([2, 2])), (Point([0, 0]), Point([3, 3])), True),
    ((Point([1, 1]), Point([2, 2])), (Point([3, 3]), Point([1, 1])), True),
    ((Point([0, 0]), Point([2, 2])), (Point([-1, -1]), Point([1, 1])), True),
    ((Point([0, 1]), Point([0, 2])), (Point([0, 2]), Point([0, 3])), True),
    ((Point([0, 1]), Point([0, 2])), (Point([1, 1]), Point([1, 2])), False),
    ((Point([0, 0]), Point([1, 2])), (Point([0, 0]), Point([1, 2.01])), False),

    # 3d
    ((Point([1, 1, 0]), Point([1, 2, 0])), (Point([1, 2, 0]), Point([1, 1, 0])), True),
    ((Point([1, 2, 1]), Point([1, 2, 2])), (Point([1, 2, 2]), Point([1, 2, 1])), True),
    ((Point([0, 0, 0]), Point([1, 1, 1])), (Point([1.5, 1.5, 1.5]), Point([3, 3, 3])), True),
    ((Point([0, 0, 1]), Point([1, 1, 2])), (Point([0, 0, 0]), Point([1, 1, 3])), False),
])
def test_collinear(points1, points2, expected_flag):
    segment1 = Segment(*points1)
    segment2 = Segment(*points2)

    assert segment1.collinear(segment2) == expected_flag


@pytest.mark.parametrize('points1, points2, expected_flag', [
    # 2d
    ((Point([1, 1]), Point([2, 2])), (Point([0, 0]), Point([3, 3])), True),
    ((Point([1, 0]), Point([2, 0])), (Point([2, 1]), Point([1, 1])), True),
    ((Point([0, 1]), Point([1, 1])), (Point([1, 2]), Point([2, 1])), False),

    # 3d
    ((Point([1, 1, 0]), Point([1, 2, 0])), (Point([1, 2, 0]), Point([1, 1, 0])), True),
    ((Point([0, 0, 1]), Point([1, 1, 2])), (Point([0, 0, 0]), Point([1, 1, 3])), False),
])
def test_parallel(points1, points2, expected_flag):
    segment1 = Segment(*points1)
    segment2 = Segment(*points2)

    assert segment1.parallel(segment2) == expected_flag


@pytest.mark.parametrize('points1, points2, expected_flag', [
    # 2d
    ((Point([1, 1]), Point([2, 2])), (Point([0, 0]), Point([3, 3])), True),
    ((Point([1, 0]), Point([2, 0])), (Point([2, 1]), Point([1, 1])), True),
    ((Point([0, 1]), Point([1, 1])), (Point([1, 2]), Point([2, 1])), True),

    # 3d
    ((Point([1, 1, 1]), Point([2, 2, 2])), (Point([1, 1, 2]), Point([2, 2, 1])), True),
    ((Point([0, 0, 1]), Point([1, 1, 2])), (Point([0, 0, 0]), Point([1, 1, 3])), True),
    ((Point([1, 1, 2]), Point([1, 2, 3])), (Point([0, 0, 0]), Point([2, 3, 1])), False),
])
def test_coplanar(points1, points2, expected_flag):
    segment1 = Segment(*points1)
    segment2 = Segment(*points2)

    assert segment1.coplanar(segment2) == expected_flag


@pytest.mark.parametrize('points1, points2, expected_points', [
    ((Point([1, 1]), Point([2, 2])), (Point([0, 0]), Point([1.5, 1.5])), (Point([1, 1]), Point([1.5, 1.5]))),
    ((Point([2, 2]), Point([1, 1])), (Point([0, 0]), Point([1.5, 1.5])), (Point([1, 1]), Point([1.5, 1.5]))),
    ((Point([1, 1]), Point([2, 2])), (Point([1.5, 1.5]), Point([0, 0])), (Point([1, 1]), Point([1.5, 1.5]))),
    ((Point([2, 2]), Point([1, 1])), (Point([1.5, 1.5]), Point([0, 0])), (Point([1, 1]), Point([1.5, 1.5]))),
    ((Point([1, 1]), Point([2, 2])), (Point([0, 0]), Point([3, 3])), (Point([1, 1]), Point([2, 2]))),
    ((Point([2, 2]), Point([1, 1])), (Point([0, 0]), Point([3, 3])), (Point([1, 1]), Point([2, 2]))),
    ((Point([1, 1]), Point([2, 2])), (Point([3, 3]), Point([0, 0])), (Point([1, 1]), Point([2, 2]))),
    ((Point([2, 2]), Point([1, 1])), (Point([3, 3]), Point([0, 0])), (Point([1, 1]), Point([2, 2]))),
    ((Point([0, 0]), Point([1, 1])), (Point([2, 2]), Point([3, 3])), None),
    ((Point([1, 1]), Point([0, 0])), (Point([2, 2]), Point([3, 3])), None),
    ((Point([0, 0]), Point([1, 1])), (Point([3, 3]), Point([2, 2])), None),
    ((Point([1, 1]), Point([0, 0])), (Point([3, 3]), Point([2, 2])), None),
    ((Point([3, 3]), Point([2, 2])), (Point([1, 1]), Point([0, 0])), None),
    ((Point([0, 0]), Point([1, 1])), (Point([1, 1]), Point([2, 2])), (Point([1, 1]), Point([1, 1]))),
    ((Point([0, 0]), Point([1, 1])), (Point([2, 2]), Point([1, 1])), (Point([1, 1]), Point([1, 1]))),
    ((Point([1, 1]), Point([0, 0])), (Point([1, 1]), Point([2, 2])), (Point([1, 1]), Point([1, 1]))),
    ((Point([1, 1]), Point([2, 2])), (Point([1, 1]), Point([0, 0])), (Point([1, 1]), Point([1, 1]))),
])
def test_overlap(points1, points2, expected_points):
    segment1 = Segment(*points1)
    segment2 = Segment(*points2)

    if expected_points is None:
        assert segment1.overlap(segment2) == expected_points
    else:
        assert segment1.overlap(segment2) == Segment(*expected_points)


@pytest.mark.parametrize('points1, points2, expected_intersect_point, isoverlap', [
    # 2d
    ((Point([1, 1]), Point([2, 2])), (Point([1, 2]), Point([2, 1])), Point([1.5, 1.5]), False),
    ((Point([1, 1]), Point([2, 2])), (Point([3, 3]), Point([0, 0])), Point([1.5, 1.5]), True),
    ((Point([1, 1]), Point([2, 2])), (Point([-5, 2]), Point([2, 10])), None, None),

    # 3d
    ((Point([1, 1, 1]), Point([2, 2, 2])), (Point([1, 2, 1]), Point([2, 1, 2])), Point([1.5, 1.5, 1.5]), False),
    ((Point([1, 1, 1]), Point([2, 2, 2])), (Point([0, 0, 0]), Point([3, 3, 3])), Point([1.5, 1.5, 1.5]), True),
    ((Point([1, 1, 2]), Point([1, 2, 3])), (Point([0, 0, 0]), Point([2, 3, 1])), None, None),
])
def test_intersect(points1, points2, expected_intersect_point, isoverlap):
    segment1 = Segment(*points1)
    segment2 = Segment(*points2)

    if expected_intersect_point is None:
        assert segment1.intersect(segment2) == expected_intersect_point
    else:
        intersection = segment1.intersect(segment2)
        assert intersection.intersect_point == expected_intersect_point
        assert intersection.isoverlap == isoverlap


def test_to_curve():
    segment = Segment(Point([1, 1]), Point([2, 2]))
    assert segment.to_curve() == Curve([(1, 2), (1, 2)])


def test_swap():
    segment = Segment(Point([1, 1]), Point([2, 2]))
    assert segment.swap() == Segment(Point([2, 2]), Point([1, 1]))
