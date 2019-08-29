# -*- coding: utf-8 -*-

import pytest
import numpy as np

from curve import Curve, Point


@pytest.mark.parametrize('data, size, ndim, dtype', [
    (([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), 5, 2, np.float64),
    (([1, 2], [1, 2], [1, 2]), 2, 3, np.float32),
    (([1], [1], [1], [1]), 1, 4, np.int32),
    (([1, 2]), 2, 1, np.float),
    ((), 0, 1, np.int32),
    (Curve([[1, 2, 3], [4, 5, 6]]), 3, 2, np.int32),
])
def test_construct(data, size, ndim, dtype):
    """Tests creating the instance of 'Curve' class
    """
    curve = Curve(data, dtype=dtype)

    assert len(curve) == size
    assert curve.size == size
    assert curve.ndim == ndim
    assert curve.dtype == dtype


def test_from_points():
    """Tests creating the instance of 'Curve' class from points
    """
    points = [
        Point([1, 5, 9]),
        Point([2, 6, 10]),
        Point([3, 7, 11]),
        Point([4, 8, 12])
    ]

    curve = Curve.from_points(points)
    assert curve.data == pytest.approx(np.array(points))


def test_eq():
    curve1 = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    curve2 = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    assert curve1 == curve2


@pytest.mark.parametrize('curve_data', [
    [[1, 2, 3], [5, 6, 7]],
    [[2, 3, 4, 5], [6, 7, 8, 9]],
])
def test_ne(curve_data):
    curve1 = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    curve2 = Curve(curve_data)
    assert curve1 != curve2


def test_reversed():
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    assert reversed(curve) == Curve([(4, 3, 2, 1), (8, 7, 6, 5)])


@pytest.mark.parametrize('point_data', [
    [1, 5],
    [2, 6],
    [4, 8],
    [3, 7],
])
def test_contains_point(point_data):
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    assert Point(point_data) in curve


@pytest.mark.parametrize('curve_data', [
    [[1, 2], [5, 6]],
    [[2], [6]],
    [[2, 3], [6, 7]],
    [[1, 2, 3], [5, 6, 7]],
    [[2, 3, 4], [6, 7, 8]],
    [[3, 4], [7, 8]],
    [[1, 2, 3, 4], [5, 6, 7, 8]],
])
def test_contains_curve(curve_data):
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    assert Curve(curve_data) in curve


@pytest.mark.parametrize('data', [
    10,
    Point([10, 20]),
    Curve([[10, 20], [30, 40]]),
])
def test_not_contains(data):
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    assert data not in curve


@pytest.mark.parametrize('point_data, start, stop, expected_index', [
    ([1, 5], None, None, 0),
    ([3, 7], 1, None, 2),
    ([2, 6], None, None, 1),
    ([2, 6], 2, None, 4),
    ([4, 8], None, 4, 3),
])
def test_index(point_data, start, stop, expected_index):
    curve = Curve([(1, 2, 3, 4, 2), (5, 6, 7, 8, 6)])
    assert curve.index(Point(point_data), start, stop) == expected_index


@pytest.mark.parametrize('point_data, expected_count', [
    ([0, 0], 0),
    ([1, 5], 1),
    ([3, 7], 2),
    ([2, 6], 3),
    ([4, 8], 1),
])
def test_count(point_data, expected_count):
    curve = Curve([(1, 2, 3, 4, 2, 3, 2), (5, 6, 7, 8, 6, 7, 6)])
    assert curve.count(Point(point_data)) == expected_count


@pytest.mark.parametrize('item, expected_data', [
    (0, [1, 5]),
    (1, [2, 6]),
    (-1, [4, 8]),
    (-2, [3, 7]),
])
def test_get_item_point(item, expected_data):
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    assert curve[item] == Point(expected_data)


@pytest.mark.parametrize('item, expected_data', [
    (slice(None, 2), [(1, 2), (5, 6)]),
    (slice(1, 3), [(2, 3), (6, 7)]),
    (slice(-2, -1), [(3,), (7,)]),
    (slice(-2, None), [(3, 4), (7, 8)]),
    (slice(None), [(1, 2, 3, 4), (5, 6, 7, 8)]),
])
def test_get_item_curve(item, expected_data):
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    assert curve[item] == Curve(expected_data)


@pytest.mark.parametrize('item, expected_data', [
    ((slice(None, None), 0), np.array([1, 2, 3, 4])),
    ((slice(None, 2), 1), np.array([5, 6])),
])
def test_get_item_values(item, expected_data):
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    assert curve[item] == pytest.approx(expected_data)


def test_concatenate():
    left_curve = Curve([(1, 2), (5, 6)])
    right_curve = Curve([(3, 4), (7, 8)])
    expected_curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])

    assert left_curve + right_curve == expected_curve

    left_curve += right_curve
    assert left_curve == Curve([(1, 2, 3, 4), (5, 6, 7, 8)])


def test_insert_point():
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    point = Point([10, 20])

    curve1 = curve.insert(1, point)
    assert curve1 == Curve([(1, 10, 2, 3, 4), (5, 20, 6, 7, 8)])


def test_insert_curve():
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    sub_curve = Curve([(10, 20), (30, 40)])

    curve1 = curve.insert(-3, sub_curve)
    assert curve1 == Curve([(1, 10, 20, 2, 3, 4), (5, 30, 40, 6, 7, 8)])


def test_append_point():
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    point = Point([10, 20])

    curve1 = curve.append(point)
    assert curve1 == Curve([(1, 2, 3, 4, 10), (5, 6, 7, 8, 20)])


def test_append_curve():
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    sub_curve = Curve([(10, 20), (30, 40)])

    curve1 = curve.append(sub_curve)
    assert curve1 == Curve([(1, 2, 3, 4, 10, 20), (5, 6, 7, 8, 30, 40)])


@pytest.mark.parametrize('index, expected_data', [
    (0, [(2, 3, 4), (6, 7, 8)]),
    (1, [(1, 3, 4), (5, 7, 8)]),
    (-1, [(1, 2, 3), (5, 6, 7)]),
    (-2, [(1, 2, 4), (5, 6, 8)]),
])
def test_delete_point(index, expected_data):
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    assert curve.delete(index) == Curve(expected_data)


@pytest.mark.parametrize('index, expected_data', [
    (slice(None, 2), [(3, 4), (7, 8)]),
    (slice(-2, None), [(1, 2), (5, 6)]),
    (slice(None, None, 2), [(2, 4), (6, 8)]),
])
def test_delete_curve(index, expected_data):
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    assert curve.delete(index) == Curve(expected_data)
