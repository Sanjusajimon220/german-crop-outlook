import numpy as np

from phase32_hyras_testsites import cells_in_box


def test_cells_in_box_keeps_only_grid_cells_inside_the_site_box_plus_margin():
    gx, gy = np.meshgrid(np.arange(0, 20000, 1000.0), np.arange(0, 20000, 1000.0))
    k = cells_in_box(gx.ravel(), gy.ravel(), (5000, 5000, 13000, 13000), margin=1000)
    x, y = gx.ravel()[k], gy.ravel()[k]
    assert x.min() == 4000 and x.max() == 14000 and y.min() == 4000 and y.max() == 14000 and len(k) == 121
