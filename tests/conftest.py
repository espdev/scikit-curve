# -*- coding: utf-8 -*-

import pytest
import numpy as np

from curve import Curve


@pytest.fixture
def circle_curve_2d():
    def _circle_curve_2d(n=100, r=1):
        theta = np.linspace(0, 2*np.pi, n)
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        return Curve([x, y])
    return _circle_curve_2d


@pytest.fixture
def circle_curve_3d():
    def _circle_curve_3d(n=200, r=10, theta=0.73):
        t = np.linspace(0, 2 * np.pi, n)
        x = np.cos(t) * r
        y = np.sin(t) * r
        z = np.ones_like(t)

        rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ])

        ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ])

        curve = Curve([x, y, z])
        curve_data = np.array(curve.data)

        for i, p in enumerate(curve):
            data = rx @ p.data
            data = ry @ data
            curve_data[i] = data

        return Curve(curve_data)

    return _circle_curve_3d


@pytest.fixture
def curve_3d():
    def _curve_3d(n=200):
        t = np.linspace(0, 2 * np.pi, n)
        x = np.cos(t)
        y = np.sin(t)
        z = x * y

        return Curve([x, y, z])
    return _curve_3d
