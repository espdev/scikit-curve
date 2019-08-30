# -*- coding: utf-8 -*-

import pytest
import numpy as np

from curve import Point


@pytest.mark.parametrize('data, ndim, dtype', [
    ([1], 1, np.int32),
    ([1, 2], 2, np.float64),
    ([1, 2, 3], 3, np.float32),
])
def test_construct(data, ndim, dtype):
    """Tests creating the instance of 'Curve' class
    """
    point = Point(data, dtype=dtype)

    assert len(point) == ndim
    assert point.ndim == ndim
    assert point.dtype == dtype


def test_eq():
    p1 = Point([1, 2])
    p2 = Point([1, 2])
    assert p1 == p2


def test_eq_integer():
    p1 = Point([1, 2], dtype=int)
    p2 = Point([1, 2], dtype=int)
    assert p1 == p2


@pytest.mark.parametrize('point_data', [
    [1, 2, 3],
    [2, 3, 4, 5],
])
def test_ne(point_data):
    p1 = Point([1, 2, 3, 4])
    p2 = Point(point_data)
    assert p1 != p2


def test_reversed():
    point = Point([1, 2, 3])
    assert reversed(point) == Point([3, 2, 1])


@pytest.mark.parametrize('index, expected_data', [
    (0, 1),
    (1, 2),
    (-1, 4),
    (-2, 3),
    (slice(0, 2), Point([1, 2])),
    ([1, 3], Point([2, 4])),
])
def test_get_item(index, expected_data):
    point = Point([1, 2, 3, 4])
    assert point[index] == expected_data


def test_dot_product():
    p1 = Point([3, 7])
    p2 = Point([-1, 4])

    assert p1 @ p2 == pytest.approx(25)
    assert p2 @ p1 == pytest.approx(25)


def test_distance():
    p1 = Point([3, 7])
    p2 = Point([-1, 4])

    assert p1.distance(p2) == pytest.approx(5.0)
    assert p1.distance(p2, metric='sqeuclidean', w=1.5) == pytest.approx(37.5)
