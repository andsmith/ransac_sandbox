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
        self._fig, self._ax = None, None

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

    @staticmethod
    def _animation_setup():
        """
        Set up the animation for plotting the model.
        """
        RansacModel._FIG, RansacModel._AXES = plt.subplots(1, 2)

    def evaluate(self, features):
        """
        Return the perpendicular distance from each point to the line.

        :param data: a Nx2 array of points
        :returns: an array of N distances, one for each point
        """
        return point_line_distances(np.array(features), *self._model_params)

    def plot_iteration(self, data, current_model, best_model, is_final=False):
        """
        2 plots, left is current iteration, with all points plotted, minimum sample, inliers/outliers distinguished 
        & fit line.  Right is the same for the best iteration's model
        """

        if RansacModel._AXES is None:
            RansacLine._animation_setup()
        axes = RansacModel._AXES

        if is_final:
            print("n_features_used: ", len(current_model['sample']))
            title = 'Final RANSAC model after %i iterations\niter %i had %i inliers (%.2f %%)' % \
                (current_model['iter']+1, best_model['iter']+1,
                 np.sum(best_model['inliers']),
                 100*np.mean(best_model['inliers']))

            fig, ax = plt.subplots(1)

            # Final plot has data w/ inliers
            data.plot(ax, inlier_mask=current_model['inliers'])

            # the sample used to find the inliers
            ax.scatter(*np.array(best_model['sample']).T, s=50,
                       label='sample', facecolors='none', edgecolors='b', linewidth=1)

            # the model fitting the sample
            plot_line(*best_model['model'].get_params(), ax=ax,
                      plt_args=['b-'], plt_kwargs={'label': 'best model\niter %i' % (best_model['iter']+1)})

            # the model fitting the consensus set (inliers)
            plot_line(*current_model['model'].get_params(), ax=ax,
                      plt_args=['g-'], plt_kwargs={'label': 'final model'})

            # a least-squares fit to all points, as a baseline
            plot_line(*fit_line(data._dataset), ax=ax,
                      plt_args=['k--'], plt_kwargs={'label': 'LS fit\nall data'})

            ax.legend()
            ax.set_title(title)
        else:
            title = 'iteration %i found %i inliers' % \
                    (current_model['iter']+1, np.sum(current_model['inliers']))

            title_best = 'best iteration (%i): %i inliers' % (best_model['iter']+1, np.sum(best_model['inliers']))

            # Clear the axes
            for a in axes:
                a.clear()

            def _plot(ax, model, line_label, plot_str='b-'):
                """
                Plot the points, sample features, model (line), inliers/outliers.
                """
                data.plot(ax, inlier_mask=model['inliers'])
                ax.scatter(*np.array(model['model']._features).T, s=50,
                           label='model features', facecolors='none', edgecolors='b', linewidth=1)
                plot_line(*model['model'].get_params(), ax=ax,
                          plt_args=[plot_str], plt_kwargs={'label': line_label})

            _plot(axes[0], current_model, '')
            _plot(axes[1], best_model, 'fit line')
            axes[0].set_title(title)
            axes[1].set_title(title_best)
            axes[1].legend()
            plt.draw()
