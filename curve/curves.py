# -*- coding: utf-8 -*-

"""
The module provides parametric equations for well-known curves

"""

import numpy as np
from scipy.special import fresnel

from curve import Curve


def arc(t_start: float = 0.0,
        t_stop: float = np.pi * 2,
        p_count: int = 49,
        r: float = 1.0,
        c: float = 0.0) -> Curve:
    r"""Produces arc or full circle curve

    Produces arc using the following parametric equations:

    .. math::

        x = cos(\theta) \dot r + c
        y = sin(\theta) \dot r + c

    By default computes full circle.

    Parameters
    ----------
    t_start : float
        Start theta
    t_stop : float
        Stop theta
    p_count : int
        The number of points
    r : float
        Circle radius
    c : float
        Circle center

    Returns
    -------
    curve : Curve
        Acr curve

    """

    theta = np.linspace(t_start, t_stop, p_count)

    x = np.cos(theta) * r + c
    y = np.sin(theta) * r + c

    return Curve([x, y])


def lemniscate_of_bernoulli(t_start: float = 0.0,
                            t_stop: float = np.pi*2,
                            p_count: int = 101,
                            c: float = 1.0) -> Curve:
    """Produces Lemniscate of Bernoulli curve

    Parameters
    ----------
    t_start
    t_stop
    p_count
    c

    Returns
    -------

    """

    t = np.linspace(t_start, t_stop, p_count)

    c_sq2 = c * np.sqrt(2)
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    denominator = sin_t ** 2 + 1

    x = (c_sq2 * cos_t) / denominator
    y = (c_sq2 * cos_t * sin_t) / denominator

    return Curve([x, y])


def archimedean_spiral(t_start: float = 0.0,
                       t_stop: float = 5 * np.pi,
                       p_count: int = 200,
                       a: float = 1.5,
                       b: float = -2.4) -> Curve:
    """Produces Archimedean spiral curve

    Parameters
    ----------
    t_start
    t_stop
    p_count
    a
    b

    Returns
    -------

    """

    t = np.linspace(t_start, t_stop, p_count)
    x = (a + b * t) * np.cos(t)
    y = (a + b * t) * np.sin(t)

    return Curve([x, y])


def euler_spiral(t_start: float = -3 * np.pi / 2,
                 t_stop: float = 3 * np.pi / 2,
                 p_count: int = 1000) -> Curve:
    """Produces Euler spiral curve

    Parameters
    ----------
    t_start
    t_stop
    p_count

    Returns
    -------

    """

    t = np.linspace(t_start, t_stop, p_count)
    ssa, csa = fresnel(t)

    return Curve([csa, ssa])


def lissajous(t_start: float = 0.0,
              t_stop: float = 2*np.pi,
              p_count: int = 101,
              a_ampl: float = 1.0,
              b_ampl: float = 1.0,
              a: float = 3.0,
              b: float = 2.0,
              d: float = 0.0,) -> Curve:
    """

    Parameters
    ----------
    t_start
    t_stop
    p_count
    a_ampl
    b_ampl
    a
    b
    d

    Returns
    -------

    """

    t = np.linspace(t_start, t_stop, p_count)

    x = a_ampl * np.sin(a * t + d)
    y = b_ampl * np.sin(b * t)

    return Curve([x, y])


def helix(t_start: float = -3 * np.pi,
          t_stop: float = 3 * np.pi,
          p_count: int = 100,
          a: float = 1.0,
          b: float = 1.0) -> Curve:
    """Produces 3-d helix curve

    Parameters
    ----------
    t_start : float
    t_stop : float
    p_count : int
    a : float
    b : float

    Returns
    -------

    """

    theta = np.linspace(t_start, t_stop, p_count)
    x = np.sin(theta) * a
    y = np.cos(theta) * a
    z = theta * b

    return Curve([x, y, z])


def irregular_helix(t_start: float = -4 * np.pi,
                    t_stop: float = 4 * np.pi,
                    z_start: float = -2.0,
                    z_stop: float = 2.0,
                    p_count: int = 100) -> Curve:
    """Produces 3-d irregular helix curve

    Parameters
    ----------
    t_start
    t_stop
    z_start
    z_stop
    p_count

    Returns
    -------

    """

    theta = np.linspace(t_start, t_stop, p_count)
    z = np.linspace(z_start, z_stop, p_count)
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    return Curve([x, y, z])
