"""
Demo of RANSAC algorithm, finding the affine transform between two images.
"""
import numpy as np
import matplotlib.pyplot as plt
from ransac import RansacModel, RansacDataFeatures
import cv2
from util import fit_affine_transform
from image_util import TestImage
import logging

PATCH_SIZE = 21
MARGIN = 15
SMOOTHING = 1.


class RansacImageData(RansacDataFeatures):
    """
    A dataset of two images for RANSAC, with an affine transform model.
        RansacImageData.get_features() returns a list of pairs of indices matching above threshold.
        each pair (i,j) corresponds to a pair of corners in the two images, the coordinates and descriptors can be 
        accessed via the member variables corners_1, corners_2, desc_1, desc_2.  (Nx2 and N element arrays)

    """

    def __init__(self, data, harris_kwargs=None, matcher_threshold=0.5):
        """
        :param data: a tuple (img1, img2) of TestImages.
        :param harris_kwargs: dictionary of parameters for the Harris corner detector.
        :param matcher_threshold: threshold for matching descriptors (see TestImage.compare_descriptors). 
        """
        self._harris_kwargs = harris_kwargs
        self._margin = 15
        self._m_threshold = matcher_threshold
        logging.info("RansacImageData created, extracting features...")
        super().__init__(data)  # calls _extract_features

    def _extract_features(self, filter_unused=True):
        """
        Find interest points in both images, extract their descriptors, match them.
        A "feature" for RANSAC is a pair of points with a high matching score.
        Create a list of such pairs.
        """
        img1, img2 = self._dataset

        # detect interest points (corners)
        self.corners_1 = img1.find_corners(harris_kwargs=self._harris_kwargs, margin=self._margin)
        self.corners_2 = img2.find_corners(harris_kwargs=self._harris_kwargs, margin=self._margin)
        logging.info('\tFound %d corners in image 1.' % len(self.corners_1))
        logging.info('\tFound %d corners in image 2.' % len(self.corners_2))
        self._desc_1 = [img1.get_patch_descriptor(c, PATCH_SIZE, smoothing=SMOOTHING) for c in self.corners_1]
        self._desc_2 = [img2.get_patch_descriptor(c, PATCH_SIZE, smoothing=SMOOTHING) for c in self.corners_2]

        # match interest points to get candidates features (corresponding pairs)
        matches = []
        for i, d1 in enumerate(self._desc_1):
            for j, d2 in enumerate(self._desc_2):
                if TestImage.compare_descriptors(d1, d2) < self._m_threshold:
                    matches.append((i, j))
        logging.info('\tFound %d candidate features.' % len(matches))

        if filter_unused:
            # find used & unused corners
            used_1, used_2 = list(set([m[0] for m in matches])), list(set([m[1] for m in matches]))
            
            # create reverse look-up to re map indices, so matches[_] = (i,j) indexes corner lists with only used corners
            used_1_map = {i: j for j, i in enumerate(used_1)}
            used_2_map = {i: j for j, i in enumerate(used_2)}

            matches = [(used_1_map[i], used_2_map[j]) for i, j in matches]
            self.corners_1 = [self.corners_1[i] for i in used_1]
            self.corners_2 = [self.corners_2[j] for j in used_2]


        self._features = matches


class RansacAffine(RansacModel):
    """
    An affine transform model for RANSAC.
    """

    def __init__(self, data, inlier_threshold, training_inds, iter=None):
        self._training_inds = training_inds
        super().__init__(data, inlier_threshold, training_inds, iter)
        self._fig, self._ax = None, None

    @staticmethod
    def _get_ptarrs_from_pairs(point_pair_list=None, feature_mask=None):
        """
        Extract the coordinates of the features one way or the other.
        :param point_pair_list: a list of pairs of points in 2d.
        :param feature_mask: a boolean array indicating which of self.features to use.
        """
        if point_pair_list is not None:
            src_pts = np.array([f[0] for f in point_pair_list])
            dst_pts = np.array([f[1] for f in point_pair_list])
        elif feature_mask is not None:
            return RansacAffine._get_ptarrs_from_pairs([f for i, f in enumerate(self.features) if feature_mask[i]])
        return src_pts, dst_pts

    def _fit(self):
        """
        Fit an affine transform to subset of features, score all featues, find in/outliers.
        """
        # Extract the coordinates of the features
        train_features = [self.features[i] for i in self._training_inds]
        src_pts = np.array([f[0] for f in train_features])
        dst_pts = np.array([f[1] for f in train_features])

        # Fit an affine transform
        self._model_params = fit_affine_transform(src_pts, dst_pts)

    def _evaluate(self, point_pair_list):
        """
        Evaluate the model on each feature.
        (distance between transformed source point and destination point)

        :param features: a list of pairs of points in 2d.
        :returns: an array of errors, one for each feature
        """
        # Extract the coordinates of the features
        src_pts = np.array([f[0] for f in point_pair_list])
        dst_pts = np.array([f[1] for f in point_pair_list])

        transformed_pts = cv2.transform(src_pts[None, :, :], self._model)
        errors = np.linalg.norm(transformed_pts[0] - dst_pts, axis=1)
        return errors

    @classmethod
    def animate_setup(cls):
        """
        Set up the animation for plotting each iteration.
        """
        cls._FIG, cls._AXES = plt.subplots(2, 2)

    def plot_iteration(self, data, current_model, best_model, which='gray', is_final=False):
        """
        if is_final = False:
            four plots, above are plots for the current model and below are plots for the best model so far

        if is_final = True:
            Show the same four windows, but replacing the upper two with the "final" model, estimated from the consensus set.
            Open a second window with two plots, the first showing image 1 in grayscale and the green outline of the bounding box of image 2, transformed backwards,
            and the second showing image 2 transformed into image 1's space (i.e. filling in the green box in image21 with the actual warped image2)

        """
        if not is_final:
            # plot the current model
            self._plot_model(data, current_model, self._AXES[0], which=which)
            self._plot_model(data, best_model, self._AXES[1], which=which)
            plt.show()
        else:
            # plot the final model
            self._plot_model(data, current_model, self._AXES[0], which=which)
            self._plot_model(data, best_model, self._AXES[1], which=which)
            plt.show()
            self._plot_final(data, current_model, best_model)
            plt.show()

    def _plot_model(self, data, model, axes, which='gray'):
        """
        Plot the model in the given axis, two windows:
              * axes[0] is image1 and detected corners in green + symbols, the sample points with blue circles around them.
              * axes[1] i  image 2, detected corners in + symbols (colored red/blue by inlier status), the transformed corners
                in green circles with green lines to corresponding corners in image 2 (i.e. all the inliers)
        """
        img1, img2 = data
        if which == 'gray':
            axes[0].imshow(img1.gray, cmap='gray')
            axes[1].imshow(img2.gray, cmap='gray')
        else:
            axes[0].imshow(img1.color)
            axes[1].imshow(img2.color)

        ax.plot([c[1] for c in self._features], [c[0] for c in self._features], 'g+')
        ax.plot([c[1] for c in model._features], [c[0] for c in model._features], 'bo')
        ax.set_title('Image 1 with corners and sample points')
        ax.axis('off')
