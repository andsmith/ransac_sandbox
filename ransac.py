"""
Base classes for RANSAC solver:
    * Feature Extractor (RansacDataFeatures)
    * Model Fitter (RansacModel)
    * 
"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class RansacDataFeatures(ABC):
    """
    Base class for representing/preprocessing the data used in RANSAC solvers.
    """

    def __init__(self, data):
        self._dataset = data
        self._features = []
        self._extract_features()

    @abstractmethod
    def _extract_features(self):
        """
        Extract all features from the data, storing them in self._features.
        """
        pass

    @abstractmethod
    def plot(self, ax=None, inlier_mask=None):
        """
        Plot the dataset & features (req to animate demos).
        :param ax: the axis to plot on
        :param inlier_mask: boolean array indicating which points are inliers, or None if no distinction is to be visualized
        :returns: the axis
        """
        pass

    def get_features(self):
        return self._features.copy()


class RansacModel(ABC):
    """
    Base class for representing the model used in RANSAC solvers.  (e.g. a line, a curve, an image homography, etc.)
    """
    _N_MIN_FEATURES = None  # minimum number of features needed to fit the model (override in subclass)

    def __init__(self, features):
        self._features = features  # remember for plotting
        self._model_params = self._fit(features)

        self._check_params()

    @classmethod
    def _check_params(cls):
        if cls._N_MIN_FEATURES is None:
            raise ValueError(
                "Subclass must set _N_MIN_FEATURES to the minimum number of features needed to fit the model")

    @abstractmethod
    def _fit(self):
        """
        Fit the model to the features.  There will either be the minimum number required to fit a model,
        or possibly more (the final inlier set)
        :returns: the model parameters
        """
        pass

    @abstractmethod
    def evaluate(self, features):
        """
        evaluate model w/ the set of features

        :param features: list of N features to evaluate (as returned by RansacDataset.get_features)
        :returns: list of N model outputs (floats)
        """
        pass

    @abstractmethod
    def plot(self, ax=None, show_features_used=True):
        """
        Plot the model on the axis (assume the dataset has already been plotted there).

        :param ax: the axis to plot on
        :param show_features_used: if True, plot the features used to fit the model
        :returns: the axis
        """
        pass

    def get_params(self):
        return self._model_params

    @classmethod
    def get_n_min_features(cls):
        """
        Get the minimum number of features needed to fit the model (constant)
        """
        return cls._N_MIN_FEATURES


def solve_ransac(data, model_type, max_error, max_iter=100, animate_pause_sec=None):
    """
    Run the RANSAC algorithm on the dataset:
        - generate a random sample of the minimum number of features needed to fit the model (the "minimal set")
        - fit the model to the minimal set
        - evaluate the model on all features, separaing inliers from outliers
        - return a model estimated from the largest set of inliers found (the "consensus set")

    :param data: the RansacDataFeatures object containing the data & extracted features to fit
    :param model_type: the type of model to fit to the data (subclass of RansacModel)
    :param max_error: threshold for inliers (will depend on implementation of RansacModel.evaluate)
    :param max_iter: maximum number of iterations to run (no consensus found)

    :param animate_pause_sec: if/how to visualize
        - None: no plotting
        - 0: plot each iteration, waiting for user input to continue
        - >0: plot each iteration, pausing for the given number of seconds

    :returns: best model found after max_iter, dict with keys:
        - 'best_iter': iteration number of the best model
        - 'best_sample': the minimal set of features used to fit the model
        - 'best_model': the model object (subclass of RansacModel)
        - 'final_model': the model object fit to the final consensus set
        - 'inliers': boolean array indicating which features are inliers (according to the best model)
        - 'features': the features used to fit the model
    """
    features = data.get_features()
    n_features = len(features)
    n_min = model_type.get_n_min_features()

    best_so_far = None

    if animate_pause_sec is not None:
        """
        Plot has two subplots, left is current iteration, right is the best so far.
        """
        plt.ion()
        fig, ax = plt.subplots(1, 2)

    for iter in range(max_iter):

        # generate a random sample
        sample_inds = np.random.choice(n_features, n_min, replace=False)
        sample = [features[j] for j in sample_inds]

        # fit the model to the sample
        model = model_type(sample)
        errors = model.evaluate(features)
        inliers = errors <= max_error

        current_model = {'best_iter': iter, 'best_sample': sample,
                         'best_model': model, 'inliers': inliers}

        if best_so_far is None or np.sum(inliers) > sum(best_so_far['inliers']):
            best_so_far = current_model
            print("New best model found on iteration %i: %i inliers" %
                  (iter, np.sum(inliers)))

        #########################################
        # ANIMATION STUFF BELOW

        if animate_pause_sec is not None:
            wait_for_keypress =  animate_pause_sec == 0 
            title ='Iteration %s of %i found %i inliers' % (iter+1, max_iter, np.sum(inliers))
            if wait_for_keypress:
                title = "Press any key to continue...\n\n"+title

            # clear both axes
            for a in ax:
                a.clear()

            # plot the current iteration
            data.plot(ax[0], inliers)
            model.plot(ax[0], show_features_used=True, plt_args=('b-',), plt_kwargs={'label': 'current model'})
            ax[0].set_title(title)

            # plot the best so far
            data.plot(ax[1], best_so_far['inliers'])
            best_so_far['best_model'].plot(ax[1], show_features_used=True, plt_args=('b-',), plt_kwargs={'label': 'best model yet'})
            ax[1].set_title('Best iteration (%i): %i inliers' % (best_so_far['best_iter']+1,
                                                                 np.sum(best_so_far['inliers'])))

            # plt.draw()
            plt.legend()
            if wait_for_keypress:
                plt.waitforbuttonpress()
            else:
                plt.pause(animate_pause_sec)

    good_features = np.array(features)[best_so_far['inliers']]
    final_model = model_type(good_features)
    result = {'final_model': final_model, 'features': features}
    result.update(best_so_far)

    return result
