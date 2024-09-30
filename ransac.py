"""
Base classes for RANSAC solver:
    * Feature Extractor (RansacDataFeatures)
    * Model Fitter (RansacModel)
    *
"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

import logging


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

    _FIG, _AXES = None, None  # class-level variables for plotting different models in the same figure

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

    @staticmethod
    @abstractmethod
    def _animation_setup():
        """
        Set RANSAC animation up if needed, e.g. fig, axes, etc.
        """

        pass

    @abstractmethod
    def plot_iteration(self, data, current_model, best_model, is_final=False):
        """
        Plot the current status (or final).

        if is_final=False, plot (and show):
            - the data
            - the current model, & samples used to fit it, & the inliers/outliers
            - the best model so far, & the inliers/outliers
            - titles/axis labels/legends as needed

        elif is_final=True (I.E. self._model_params have been estimated from the consensus set), plot a visualization of:

            - the data
            - the best iteration's model, as above
            - the current model (i.e. the final model fit to the consensus set)

        :param data: the RansacDataFeatures object containing the data & extracted features

        :param current_model: dict: {'iter': current iteration number
                             'sample': sample used to fit the best model  (should be self._features)
                             'model': RansacModel object, fit from that sample (i.e. self)
                             'inliers': inliers, boolean array index of data.
                             }
        :param best_model: dict: (same from best iteration so far)
        :param is_final: bool, True if the final model is being plotted ()
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

    :returns: dict with the following structure:
        {'best': {'iter': best iteration number,
                  'sample': sample used to fit the best model,
                  'model': best model,
                  'inliers': boolean array, the consensus set of inliers},
        'final': {'model': final model fit to the consensus set,
                  'iter':  total iterations run },
        'features': the list of extracted features,
        'inliers': boolean array of inliers from the final model fit to the consensus set
        }
    """
    features = data.get_features()
    n_features = len(features)
    n_min = model_type.get_n_min_features()

    best_so_far = None
    ax = None
    logging.info("Running RANSAC on %i features" % n_features)

    if animate_pause_sec is not None:
        logging.info("Animating RANSAC with pause of %.2f sec" % animate_pause_sec)
        plt.ion()

    for iter in range(max_iter):

        # generate a random sample
        sample_inds = np.random.choice(n_features, n_min, replace=False)
        sample = [features[j] for j in sample_inds]

        # fit the model to the sample
        model = model_type(sample)
        errors = model.evaluate(features)
        inliers = errors <= max_error

        current_model = {'iter': iter, 'sample': sample,
                         'model': model, 'inliers': inliers}
        
        logging.info("Iteration %i: %i inliers" % (iter, np.sum(inliers)))

        if best_so_far is None or np.sum(inliers) > sum(best_so_far['inliers']):
            best_so_far = current_model
            print("New best model found on iteration %i: %i inliers" %
                  (iter, np.sum(inliers)))

        if animate_pause_sec is not None:
            model.plot_iteration(data, current_model, best_so_far, is_final=False)
            if animate_pause_sec == 0:
                plt.waitforbuttonpress()
            else:
                plt.pause(animate_pause_sec)
    # gather inliers from best iteration (consensus set)
    good_features = np.array(features)[best_so_far['inliers']]

    # fit the final model to the consensus set
    final_model = {'model': model_type(good_features),
                   'iter': max_iter,
                   'sample': good_features,
                   'inliers': best_so_far['inliers']}

    result = {'best': best_so_far, 'final': final_model, 'features': features, 'consensus': best_so_far['inliers']}

    if animate_pause_sec is not None:
        model.plot_iteration(data, final_model, best_so_far, is_final=True)
        plt.pause(0)

    result.update(best_so_far)

    return result
