# -*- coding: utf-8 -*-

import typing as ty

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt

if ty.TYPE_CHECKING:
    from curve import Curve


def _multicolor_curveplot(curve, name, cmap, ax, **kwargs):
    """Plots a curve parameter as multicolored line with given colormap
    """

    points = curve.data[:, np.newaxis]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    curve_values = list(curve.values())
    multicolor_values = getattr(curve, name)

    marker = kwargs.pop('marker', None)

    if curve.is2d:
        line_collection = LineCollection
    else:
        line_collection = Line3DCollection

    norm = Normalize(multicolor_values.min(), multicolor_values.max())
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    colors = mappable.to_rgba(multicolor_values)

    curve_segments = line_collection(segments, zorder=0, **kwargs)
    curve_segments.set_array(multicolor_values)
    curve_segments.set_norm(norm)
    curve_segments.set_cmap(cmap)

    if curve.is2d:
        line = ax.add_collection(curve_segments)
        ax.autoscale()
    else:
        line = ax.add_collection3d(curve_segments)
        ax.auto_scale_xyz(*curve_values)

    if marker:
        ax.scatter(*curve_values, c=colors, marker=marker, **kwargs)

    cbar = ax.figure.colorbar(mappable, ax=ax)
    cbar.ax.set_ylabel(name)

    return line


def curveplot(curve: 'Curve',
              *args,
              param: ty.Optional[str] = None,
              param_cmap: str = 'plasma',
              show_normals: bool = False,
              ax: ty.Optional[plt.Axes] = None,
              **kwargs) -> ty.Optional[plt.Axes]:
    """Plots a curve

    The function plots 2-d or 3-d curve using matplotlib.

    Parameters
    ----------
    curve : Curve
        Curve object
    param : Optional[str]
        The curve parameter name for show as multicolor line, "curvature", for example
    param_cmap : Optional[str]
        The colormap for show the curve parameter as multicolor line
    show_normals : bool
        Draw normal vectors
    ax : Optional[Axes]
        MPL axes

    Returns
    -------
    ax : Axes
        If axes object was created

    """

    if curve.ndim > 3:
        raise ValueError('Cannot plot the {}-dimensional curve.'.format(curve.ndim))

    return_axes = False

    if ax is None:
        return_axes = True
        fig = plt.figure()

        if curve.is2d:
            ax = fig.add_subplot(1, 1, 1)
        elif curve.is3d:
            ax = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        if curve.is3d and not isinstance(ax, Axes3D):
            raise TypeError('Cannot plot 3-d curve on 2-d axes')

    values = list(curve.values())

    if param:
        _multicolor_curveplot(curve, param, param_cmap, ax, **kwargs)
    else:
        ax.plot(*values, *args, **kwargs)

    if show_normals:
        normals = [curve.frenet2[:, i] for i in range(curve.ndim)]

        if curve.is2d:
            ax.quiver(*values, *normals,
                      units='height', width=0.002, zorder=0, color='gray')
        else:
            # FIXME: quiver3d in mpl is a piece of shit
            # Maybe we could use this solution
            # https://stackoverflow.com/a/22867877/419926

            # length = np.abs(np.array(values)).mean() / 3
            #
            # ax.quiver(*values, *normals,
            #           length=length, normalize=True, arrow_length_ratio=0.1, color='gray')
            raise NotImplementedError('Plot normals is not implemented for 3-d curves')

    title = 'Curve (points: {}, length: ~{:.2f})'.format(curve.size, curve.arclen)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if curve.is3d:
        ax.set_zlabel('Z')

    if curve.is2d:
        ax.axis('equal')

    if return_axes:
        return ax
