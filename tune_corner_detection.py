"""
Iterate through conrner detection parameters to find the best combination for synthetic test images,
Both images should have roughly the same number of corners (50-2/300?), but the transformed image many have
fewer if some locations were transformed out of the image bounds.

for every set of parameters to cv2.cornerHarris, param:
  for trial in [1, n_trials]
    * Create an image pair, detect corners in the first, corners in the second.
    * Let score_1 be the score for the corner detector on image 1 (see score_1 for details)
    * Transform the corners in the first image to coordinates in the second image (removing those out of bounds), call these the "true" corners.
    * Let score_2 be the score for image 2, calculated from comparing the detected corners to the "true" corners (see score_2 for details)
    * score = score_1 * score_2

    (all scores in [0, 1] where higher is better)
    
The tuning algorithm shows grids of these scores and returns the parameters with the highest score.

"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from image_util import TestImage
from multiprocessing import Pool, cpu_count
import logging


class CornerDetectionTrial(object):
    """
    Represent test for 1 set of detection params (1 or more random trials).
    """

    def __init__(self, t_i, img_size, params, n_reps, noise_frac, margin=10, kind='double'):
        """
        (set up, but don't run trial.)

        Will test params for these kinds of trials:
            * 'single':  Scoring the detector on the first (untransformed) image.
            * 'double':  Scoring the detector on both images.

        :param t_i: index of this trial
        :param img_size: tuple of image dimensions (width, height)
        :param params: dict of corner detection parameters (blockSize, ksize, k)
        :param n_reps: number of trials to run for this set of params
        :param noise_frac: fraction of image pixels to randomly change in creating image 2
        :param margin: (param to TestImage.find_corners) margin around image edges to ignore corners
        :param kind: 'single' or 'double' (see above)
        """
        if kind not in ['single', 'double']:
            raise ValueError('kind must be "single" or "double"')
        self.kind = kind
        self.i = t_i
        self.img_size = img_size
        self.noise_frac = noise_frac
        self.params = params
        self.n_reps = n_reps
        self.margin = margin

        self._n_corners_1 = []
        self._n_corners_2 = []

        self.mean_corners_1 = None
        self.mean_corners_2 = None

        self._scores = []
        self.score = None  # average of _scores

    def _trial(self):
        """
        Run a single trial of corner detection w/current params on a pair of synthetic test images.
        :returns: score of this trial, number of corners detected in image 1, number of corners detected in image 2 (or None if kind='single')
        """
        # Create the first image and detect corners
        image1 = TestImage(size=self.img_size)
        corners1 = np.array(image1.find_corners(margin=self.margin, harris_kwargs=self.params))
        score1 = self._score_1(corners1)
        if self.kind == 'single':
            return score1, len(corners1), None
        
        # Create the second image, detect corners, and transform the corners from the first image to the second.
        image2, transf = image1.transform(self.noise_frac)
        corners2 = np.array(image2.find_corners(margin=self.margin, harris_kwargs=self.params))
        true_corners2= image2.transform_coords(transf, corners1)
        score = score1 * self._score_2(corners1, corners2, true_corners2)

        return score, len(corners1), len(corners2)

    def _score_1(self, corners1):
        """
        Calculate a score for the corner detector on image 1:
            Assume the corners are sensible (this can be visually verified), all we care is that a "reasonable" number of corners are detected.
            The shape of our score function will be piecewise linear:
                * 0 - min_corners: 0
                * min_corners - low_corner: increasing linearly to 1
                * low_corners - high_corners_a: 1
                * high_corners_a - high_corners_b: decreasing linearly to 0.5
                * high_corners_b - max_corners: decreasing linearly to 0
                * max_corners - inf: 0
        """
        err_params = {'min_corners': 20,
                      'low_corners': 40,
                      'high_corners_a': 70,
                      'high_corners_b': 150,
                      'max_corners': 250}
        
        n_corners = len(corners1)
        if n_corners < err_params['min_corners']:
            return 0.
        elif n_corners < err_params['low_corners']:
            return (n_corners - err_params['min_corners']) / (err_params['low_corners'] - err_params['min_corners'])
        elif n_corners < err_params['high_corners_a']:
            return 1.
        elif n_corners < err_params['high_corners_b']:
            return 1 - (n_corners - err_params['high_corners_a']) / (err_params['high_corners_b'] - err_params['high_corners_a'])
        elif n_corners < err_params['max_corners']:
            return 0.5 - (n_corners - err_params['high_corners_b']) / (err_params['max_corners'] - err_params['high_corners_b'])
        else:
            return 0.
    # 
    def _score_2(self, corners1, corners2, true_corners2):  
        """
        Calculate a score for image 2, calculated from comparing the detected corners to the "true" corners:
            * The "true" corners are the corners in image 1 transformed to the coordinates of image 2.
            * The score is the accuracy at the break-even point of the precision-recall curve (using distance as the metric).

        :param corners1: corners detected in image 1
        :param corners2: corners detected in image 2
        :param transform_func: transformation function from coordinate system of image 1 to image 2
        :returns: score for image 2
        """
        corners1 = np.array(corners1)
        corners2 = np.array(corners2)
        true_corners2 = np.array(true_corners2)

        if len(corners1) == 0 or len(corners2) == 0 or len(true_corners2) == 0:
            return 0
        # Calculate the distance between each corner in image 2 and the closest corner in image 1
        dists = np.linalg.norm(true_corners2[:, None] - corners2[None], axis=2)
        min_dists = np.min(dists, axis=0)

        
        # Sort the distances and calculate the precision and recall at each point
        idxs = np.argsort(min_dists)
        n_true = len(corners1)
        n_detected = len(corners2)
        precision = np.zeros(n_detected)
        recall = np.zeros(n_detected)
        for i in range(n_detected):
            n_correct = np.sum(min_dists[idxs[:i]] < 1)
            precision[i] = n_correct / (i + 1)
            recall[i] = n_correct / n_true
        # Find the break-even point of the precision-recall curve
        i = np.argmax(np.abs(precision - recall))
        
        # Finally, return the accuracy at the break-even point
        return (precision[i] + recall[i]) / 2.

    def eval(self):
        """
        Run corner detection on synthetic test images with the given parameters, n_reps times.
        """
        for _ in range(self.n_reps):
            score, nc1, nc2 = self._trial()
            self._n_corners_1.append(nc1)
            self._n_corners_2.append(nc2)
            self._scores.append(score)

        self.score = np.mean(self._scores)
        self.mean_corners_1 = np.mean(self._n_corners_1)
        self.mean_corners_2 = np.mean(self._n_corners_2) if self.kind == 'double' else None
        logging.info("\ttrial %s complete (%i repetitions):  " % (self.params, self.n_reps))
        logging.info("\t\tmean score:  %.5f" % ( self.score,))
        logging.info("\t\tmean corners in 1:  %i" % ( self.mean_corners_1, ))

        if self.mean_corners_2 is not None:
            logging.info("\t\tmean corners in 2:  %i\n" % ( self.mean_corners_2, ))
        return self
        

def _eval_trial(trial):
    # Map this function to a list of CornerDetectionTrial objects for parallel processing.
    return trial.eval()



def tune(img_size, blockSize_vals, kSize_vals, k_vals, noise_frac, n_trials_per_param=50, n_cores=0, kind='double', plot=False):
    """
    Iterate through corner detection parameters to find the best combination for synthetic test images.
    (params to cv2.cornerHarris)
    :param img_size: tuple of image dimensions (width, height)
    :param blockSize_vals: list of block sizes for corner detection
    :param kSize_vals: list of Sobel kernel sizes for corner detection
    :param k_vals: list of Harris corner detection parameter k
    :param n_trials_per_param: number of trials per parameter combination
    :param n_cores: number of cores to use for parallel processing (0 for all cores)
    """
    # Generate parameter combination list
    param_combos = list(product(blockSize_vals, kSize_vals, k_vals))
    def _get_kw_params(params):
        return dict(blockSize=params[0], ksize=params[1], k=params[2])
    param_combos = [_get_kw_params(params) for params in param_combos]
    trials = [CornerDetectionTrial(i, img_size, params, n_trials_per_param, noise_frac, kind=kind)
              for i, params in enumerate(param_combos)]

    # Run corner detection on each image with each parameter combination
    n_cores = cpu_count()+1 if n_cores == 0 else n_cores
    logging.info("About to run %i trials on %i cores..." % (len(trials), n_cores))
    if n_cores == 1:
        for trial in trials:
            trial.eval()
    else:
        with Pool(n_cores) as p:
            results = p.map(_eval_trial, trials)
            trials = results
    scores = np.array([trial.score for trial in trials])

    # Find the parameter combination with the highest average number of corners detected
    best_params = param_combos[np.argmax(scores)]
    best_score = np.max(scores)

    if plot:
        # Create a subplot for every blockSize value, plot a grid of scores for each kSize, k value pair tested w/ that blockSize.
        # Show the plot w/a color bar.
        if len(blockSize_vals) > 1:
            n_plots = len(blockSize_vals)
            n_rows = int(np.floor(np.sqrt(n_plots)))
            n_cols = int(np.ceil(n_plots / n_rows))
            fig, axes = plt.subplots(n_rows, n_cols)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 1)
            axes = [axes]

        for i in range(n_plots):

            axes[i].set_title('blockSize = %d' % blockSize_vals[i])
            scores_blockSize = scores[i::len(blockSize_vals)]
            scores_blockSize = scores_blockSize.reshape(len(kSize_vals), len(k_vals))
            cax = axes[i].matshow(scores_blockSize, cmap='viridis')
            fig.colorbar(cax, ax=axes[i])

            # use at most 4 k_val ticks
            n_ticks = 3
            if len(k_vals) > (n_ticks-1):
                k_val_step = int(len(k_vals) / (n_ticks-1))
                axes[i].set_xticks(range(0, len(k_vals), k_val_step), ["%.3f" % k for k in k_vals[::k_val_step]])
            else:
                axes[i].set_xticks(range(len(k_vals)), ["%.3f" % k for k in k_vals])
            axes[i].set_yticks(range(len(kSize_vals)), kSize_vals)
            axes[i].tick_params(axis='x', top=False, labeltop=False, bottom=True, labelbottom=True)

            # label horizontal axis of bottom row only
            if i >= n_plots - n_cols:
                axes[i].set_xlabel('k')
            # label vertical axis of left column only
            if i % n_cols == 0:
                axes[i].set_ylabel('kSize')

        # hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')

        plt.suptitle("Harris Corner Detection Parameter Tuning:  type %s" % kind)
        plt.show()

    return best_params, best_score


def optimize(noise_frac,kind, plot=True):

    img_size = 600, 600

    if True:
        # full range of parameters to test:
        blockSize_vals = [2, 4, 8, 14, 20]
        kSize_vals = [1, 3, 5, 7, 9]
        k_vals = np.linspace(0.04, 0.08, 5)
    else:
        # smaller range of parameters to test for debugging
        blockSize_vals = [1, 2, 3, 4, 5]
        kSize_vals = [3, 5]
        k_vals = [.04, .06]

    best_params, score = tune(img_size, blockSize_vals, kSize_vals, k_vals,
                              noise_frac, n_trials_per_param=50, n_cores=0, kind=kind, plot=plot)
    logging.info('\n\nBest parameters (score=%.5f):\n%s\n\n' % (score,  best_params))
    return best_params, score



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    noise_frac = 0.05
    optimize(noise_frac=noise_frac,kind='double', plot=True)
    #optimize(noise_frac=noise_frac,kind='double')
