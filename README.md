# scikit-curve

A set of tools to manipulate n-dimensional geometric curves in Python.


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

![lissajous](assets/lissajous_plot.png)
