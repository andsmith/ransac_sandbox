"""
Create and evaluate a single trial (parameter set).
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from image_util import TestImage
import logging

from tune_corner_detection import CornerDetectionTrial


def test_corner_detection_trial(noise_frac, n_trials=10, params=None, kind='single'):
    """
    Test the CornerDetectionTrial class.
    """
    img_size = 400, 400

    _params = dict(blockSize=2,
                   ksize=3,
                   k=0.04)
    if params is not None:
        _params.update(params)

    trial = CornerDetectionTrial(0, img_size, _params, n_trials, noise_frac, kind=kind)
    trial.eval()

    logging.info('CornerDetectionTrial found:')
    logging.info('\tn_trials:  %s' % (trial.n_reps,))
    logging.info('\tscore:  %s' % (trial.score,))
    logging.info('\tmean_corners_1:  %s' % (trial.mean_corners_1,))
    if kind == 'double':
        logging.info('\tmean_corners_2:  %s' % (trial.mean_corners_2,))

    assert 0 <= trial.score <= 1, 'CornerDetectionTrial score out of expected range:  %.2f' % trial.score
    logging.info('CornerDetectionTrial test passed.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    size = 400, 400
    noise_frac = 0.1
    params = dict(blockSize=2,
                  ksize=9,
                  k=0.06)

    test_corner_detection_trial(noise_frac=noise_frac, n_trials=3, params=params)
    logging.info('All tests passed.')
