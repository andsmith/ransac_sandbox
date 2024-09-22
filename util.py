import numpy as np
import matplotlib.pyplot as plt


def make_line_data(n_pts, n_outliers, noise_sigma=0.01):
    # make 2d points in a line, add noise & outlier points
    # make sure they are sufficiently far away, so the line is long enough for 
    # the demo to be visualized clearly.
    min_sep = 0.6
    while True:
        p0 = np.random.rand(2).reshape(-1, 1)
        p1 = np.random.rand(2).reshape(-1, 1)
        if np.linalg.norm(p1 - p0) > min_sep:
            break
    t = np.random.rand(n_pts)
    line_pts = np.tile(p0, (1, n_pts)) * t + np.tile(p1, (1, n_pts)) * (1. - t)
    line_pts += np.random.randn(n_pts*2).reshape(2, -1)*noise_sigma
    error_pts = np.random.rand(n_outliers*2).reshape(-1, 2)
    return np.vstack((line_pts.T, error_pts))

