"""
Demo of RANSAC algorithm, fitting a line to a set of points with outliers.
"""
import numpy as np
import matplotlib.pyplot as plt

from util import make_line_data, plot_line, point_line_distances, fit_line
from ransac import RansacModel, RansacDataFeatures, solve_ransac


class RansacLineData(RansacDataFeatures):
    """
    A dataset of 2d points for RANSAC, with a line model.
    """

    def __init__(self, data):
        """
        :param data: a Nx2 array of points
        """
        super().__init__(data)

    def _extract_features(self):
        """
        Features for fitting a line to points are the points themselves.
        """
        self._features = list(self._dataset)

    def plot(self, ax=None, inlier_mask=None):
        """
        Plot the data points, coloring inliers and outliers differently.
        :param inlier_mask: boolean array indicating which points are inliers\
        """
        ax = ax if ax is not None else plt.gca()

        # Plot all points
        if inlier_mask is None:
            ax.plot(self._dataset[:, 0], self._dataset[:,
                    1], 'k.', label='all points')
        else:
            outliers = np.logical_not(inlier_mask)
            ax.plot(self._dataset[outliers, 0], self._dataset[outliers, 1],
                    'r.', label='outliers')
            ax.plot(self._dataset[inlier_mask, 0], self._dataset[inlier_mask, 1],
                    'b.', label='inliers')

        return ax


class RansacLine(RansacModel):
    """
    A 2d line estimator for RANSAC robust to outliers.
    """
    _N_MIN_FEATURES = 2  # need 2 points to define a line

    def __init__(self, features):
        """
        Set model params a, b, c of the line a*x + b*y + c = 0.
        """
        super().__init__(features=features)

    def __str__(self):
        return "Line: %.3f*x + %.3f*y + %.3f = 0" % tuple(self._model_params)

    @staticmethod
    def _fit(features):
        """
        Fit a line to the set of points:
            - if N=2, the line will be through the points exactly.
            - otherwise, the line will be the least squares fit.

        :param features: N points in 2d space
        :returns: a LineEstimator object w/ the model params.
        """
        return fit_line(np.array(features))

    def evaluate(self, features):
        """
        Return the perpendicular distance from each point to the line.

        :param data: a Nx2 array of points
        :returns: an array of N distances, one for each point
        """
        return point_line_distances(np.array(features), *self._model_params)

    def plot(self, ax=None, show_features_used=True, plt_args=(),  plt_kwargs={}):
        """
        Plot the line on the current axis.
        """
        ax = ax if ax is not None else plt.gca()

        if show_features_used:
            points = np.array(self._features)  # draw a circle around these
            ax.scatter(points[:, 0], points[:, 1],  s=200,
                       label="model features", facecolors='none', edgecolors='b',linewidth =3)

        plot_line(*self._model_params, ax=ax,
                  plt_args=plt_args, plt_kwargs=plt_kwargs)