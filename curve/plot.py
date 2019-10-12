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


class plot:
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

    """

    def __init__(self,
                 curve: 'Curve',
                 *args: ty.Any,
                 param: ty.Optional[str] = None,
                 param_cmap: str = 'plasma',
                 show_normals: bool = False,
                 axes: ty.Optional[plt.Axes] = None,
                 **kwargs: ty.Any) -> None:
        if curve.ndim > 3:
            raise ValueError(
                'Cannot plot the {}-dimensional curve.'.format(curve.ndim))

        self._curve = curve
        self._param = param
        self._param_cmap = param_cmap
        self._show_normals = show_normals
        self._axes = axes
        self._args = args
        self._kwargs = kwargs

        self._plot()

    @property
    def axes(self) -> plt.Axes:
        return self._axes

    def plot(self,
             curve: 'Curve',
             *args: ty.Any,
             param: ty.Optional[str] = None,
             param_cmap: str = 'plasma',
             show_normals: bool = False,
             **kwargs: ty.Any) -> 'plot':
        return plot(
            curve,
            *args,
            param=param,
            param_cmap=param_cmap,
            show_normals=show_normals,
            axes=self._axes,
            **kwargs
        )

    def _multicolor_plot(self):
        """Plots a curve parameter as multicolored line with given colormap
        """

        points = self._curve.data[:, np.newaxis]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        curve_values = list(self._curve.values())
        multicolor_values = getattr(self._curve, self._param)

        marker = self._kwargs.pop('marker', None)

        if self._curve.is2d:
            line_collection = LineCollection
        else:
            line_collection = Line3DCollection

        norm = Normalize(multicolor_values.min(), multicolor_values.max())
        mappable = ScalarMappable(norm=norm, cmap=self._param_cmap)
        colors = mappable.to_rgba(multicolor_values)

        curve_segments = line_collection(segments, zorder=0, **self._kwargs)
        curve_segments.set_array(multicolor_values)
        curve_segments.set_norm(norm)
        curve_segments.set_cmap(self._param_cmap)

        if self._curve.is2d:
            self._axes.add_collection(curve_segments)
            self._axes.autoscale()
        else:
            self._axes.add_collection3d(curve_segments)
            self._axes.auto_scale_xyz(*curve_values)

        if marker:
            self._axes.scatter(*curve_values, c=colors, marker=marker, **self._kwargs)

        cbar = self._axes.figure.colorbar(mappable, ax=self._axes)
        cbar.ax.set_ylabel(self._param)

    def _plot(self):
        if self._axes is None:
            fig = plt.figure()

            if self._curve.is2d:
                self._axes = fig.add_subplot(1, 1, 1)
            elif self._curve.is3d:
                self._axes = fig.add_subplot(1, 1, 1, projection='3d')
        else:
            if self._curve.is3d and not isinstance(self._axes, Axes3D):
                raise TypeError('Cannot plot 3-d curve on 2-d axes')

        values = list(self._curve.values())

        if self._param:
            self._multicolor_plot()
        else:
            self._axes.plot(*values, *self._args, **self._kwargs)

        if self._show_normals:
            normals = [self._curve.frenet2[:, i] for i in range(self._curve.ndim)]

            if self._curve.is2d:
                self._axes.quiver(*values, *normals,
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

        # TODO: add curve info to legend
        # title = 'Curve (points: {}, length: ~{:.2f})'.format(
        #     self._curve.size, self._curve.arclen)
        # self._axes.set_title(title)

        if self._curve.is2d:
            self._axes.axis('equal')
