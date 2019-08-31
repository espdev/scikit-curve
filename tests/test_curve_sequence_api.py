# -*- coding: utf-8 -*-

import pytest
import numpy as np

from curve import Curve, Point, Axis


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


@pytest.mark.parametrize('curve_data', [
    [],
    [[], [], []],
])
def test_bool(curve_data):
    curve = Curve(curve_data)
    assert not curve


@pytest.mark.parametrize('curve_data1, curve_data2', [
    ([(1, 2, 3, 4), (5, 6, 7, 8)], [(1, 2, 3, 4), (5, 6, 7, 8)]),
    ([], []),
    ([[], []], [[], []]),
])
def test_eq(curve_data1, curve_data2):
    assert Curve(curve_data1) == Curve(curve_data2)


@pytest.mark.parametrize('curve_data', [
    [[1, 2, 3], [5, 6, 7]],
    [[2, 3, 4, 5], [6, 7, 8, 9]],
    [],
    [[], []],
])
def test_ne(curve_data):
    curve1 = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    curve2 = Curve(curve_data)
    assert curve1 != curve2


def test_reversed():
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    assert reversed(curve) == Curve([(4, 3, 2, 1), (8, 7, 6, 5)])


def test_reverse():
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    curve.reverse()
    assert curve == Curve([(4, 3, 2, 1), (8, 7, 6, 5)])


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


@pytest.mark.parametrize('index, value, expected_data', [
    (1, Point([10, 20]), [(1, 10, 3, 4), (5, 20, 7, 8)]),
    ([1, 2], Point([10, 20]), [(1, 10, 10, 4), (5, 20, 20, 8)]),
    (slice(0, 2), Curve([[10, 20], [30, 40]]), [(10, 20, 3, 4), (30, 40, 7, 8)]),
    ((slice(None, None), 0), np.array([10, 20, 30, 40]), [(10, 20, 30, 40), (5, 6, 7, 8)]),
])
def test_set_item(index, value, expected_data):
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    curve[index] = value

    assert curve == Curve(expected_data)


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


@pytest.mark.parametrize('axis, expected_data', [
    (Axis.X, np.array([1, 2, 3, 4])),
    (Axis.Y, np.array([5, 6, 7, 8])),
    (Axis.Z, np.array([9, 10, 11, 12])),
    (-1, np.array([9, 10, 11, 12])),
])
def test_get_values(axis, expected_data):
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)])
    assert curve.values(axis) == pytest.approx(expected_data)


def test_insert_dim():
    curve = Curve([(1, 2, 3, 4), (9, 10, 11, 12)])
    curve1 = curve.insert_dim(1, [5, 6, 7, 8])

    assert curve1 == Curve([(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)])


def test_append_dim():
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8)])
    curve1 = curve.append_dim([9, 10, 11, 12])

    assert curve1 == Curve([(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)])


@pytest.mark.parametrize('index, expected_data', [
    (Axis.X, [(5, 6, 7, 8), (9, 10, 11, 12)]),
    (Axis.Y, [(1, 2, 3, 4), (9, 10, 11, 12)]),
    (Axis.Z, [(1, 2, 3, 4), (5, 6, 7, 8)]),
    (-1, [(1, 2, 3, 4), (5, 6, 7, 8)]),
    ([Axis.X, Axis.Z], [(5, 6, 7, 8)]),
])
def test_delete_dim(index, expected_data):
    curve = Curve([(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)])
    curve1 = curve.delete_dim(index)

    assert curve1 == Curve(expected_data)


@pytest.mark.parametrize('curve_data, expected_data', [
    ([(4, 2, 3, 1), (8, 6, 7, 5)], [(4, 2, 3, 1), (8, 6, 7, 5)]),
    ([(3, 2, 1, 4, 2), (7, 6, 5, 8, 6)], [(3, 2, 1, 4), (7, 6, 5, 8)]),
    ([(3, 1, 2, 1, 4, 2), (7, 5, 6, 5, 8, 6)], [(3, 1, 2, 4), (7, 5, 6, 8)]),
])
def test_unique(curve_data, expected_data):
    assert Curve(curve_data).unique() == Curve(expected_data)


@pytest.mark.parametrize('curve_data, isa, expected_data', [
    ([(1, 2, np.nan, 3, 2, 4), (5, 6, 1, 7, np.inf, 8)],
     lambda x: np.isnan(x) | np.isinf(x), [(1, 2, 3, 4), (5, 6, 7, 8)]),
    ([(1, 2, 3, 4), (5, 6, 7, 8)],
     lambda x: x == [2, 6], [(1, 3, 4), (5, 7, 8)]),
])
def test_drop(curve_data, isa, expected_data):
    assert Curve(curve_data).drop(isa) == Curve(expected_data)
