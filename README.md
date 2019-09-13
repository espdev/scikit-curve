# scikit-curve

[![PyPI version](https://img.shields.io/pypi/v/scikit-curve.svg)](https://pypi.python.org/pypi/scikit-curve)
![Supported Python versions](https://img.shields.io/pypi/pyversions/scikit-curve.svg)
[![BSD License](https://img.shields.io/pypi/l/scikit-curve.svg)](https://github.com/espdev/scikit-curve/blob/master/LICENSE)
[![Build status](https://travis-ci.org/espdev/scikit-curve.svg?branch=master)](https://travis-ci.org/espdev/scikit-curve)
[![Docs status](https://readthedocs.org/projects/scikit-curve/badge/)](https://scikit-curve.readthedocs.io/en/latest/)

:warning: :construction: **UNDER DEVELOPMENT** :construction:  :warning:

A toolkit to manipulate n-dimensional geometric curves in Python.

```python
import matplotlib.pyplot as plt

from curve.curves import lissajous
from curve.plot import curveplot
from curve import PreservedSpeedInterpolationGrid

curve = lissajous(p_count=51)

grid = PreservedSpeedInterpolationGrid(301)
curve_i = curve.interpolate(grid, method='hermite')

curveplot(curve_i, param='speed', normals=True, marker='.')

plt.show()
```

![lissajous plot](https://github.com/espdev/scikit-curve/blob/master/assets/lissajous_plot.png)
