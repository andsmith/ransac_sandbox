"""
Demo of RANSAC algorithm, finding the affine transform between two images.
"""
import numpy as np
import matplotlib.pyplot as plt
from ransac import RansacModel, RansacDataFeatures
import cv2
from util_affine import Affine
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
        # publics, set in _extract_features:
        self.corners_1, self.corners_2 = None, None  # x, y
        self.desc_1, self.desc_2 = None, None
        self.size = data[0].gray.shape[::-1]

        self._matches = {}  # self._matches[i] = list of indices into corners_2 matching corners_1[i], for sampling
        self._matches_flat = []  # list of all (i,j) pairs of indices matching above threshold
        self._matches_idx = {}  # reverse-lookup for pairs -> index in _matches_flat
        self._harris_kwargs = harris_kwargs
        self._margin = 15
        self._m_threshold = matcher_threshold
        logging.info("RansacImageData created, extracting features...")
        super().__init__(data)  # calls _extract_features

    def get_sample_inds(self, n):
        """
        Don't pick any two features that have the same corner in common.
        :param n: number of features to sample
        :return: n indices into self._matches_flat
        """
        candidates = range(len(self._matches_flat))

        sample = []
        for _ in range(n):
            if len(candidates) == 0:
                raise ValueError(
                    "Not enough candidates to sample from after excluding multiple matches to/from the same feature.")
            pair_ind = np.random.choice(candidates, 1)[0]
            sample.append(pair_ind)
            pair = self._matches_flat[pair_ind]
            candidates = [c for c in candidates if self._matches_flat[c][0] != pair[0] and
                          self._matches_flat[c][1] != pair[1]]
            print("Candidates after removing %s:  %i" % (pair, len(candidates)))
        return sample

    def get_features(self,  indices=None):
        if indices is not None:
            return [self._matches_flat[i] for i in indices]
        else:
            return self._matches_flat

    def _extract_features(self):
        """
        Find interest points in both images, extract their descriptors, match them.
        A "feature" for RANSAC is a pair of points with a high matching score.
        Create a list of such pairs (the "matches").
        """
        img1, img2 = self._dataset

        # detect interest points (corners)
        self.corners_1 = img1.find_corners(harris_kwargs=self._harris_kwargs, margin=self._margin)
        self.corners_2 = img2.find_corners(harris_kwargs=self._harris_kwargs, margin=self._margin)
        logging.info('\tFound %d corners in image 1.' % len(self.corners_1))
        logging.info('\tFound %d corners in image 2.' % len(self.corners_2))
        self.desc_1 = [img1.get_patch_descriptor(c[0], c[1], PATCH_SIZE, smoothing=SMOOTHING) for c in self.corners_1]
        self.desc_2 = [img2.get_patch_descriptor(c[0], c[1], PATCH_SIZE, smoothing=SMOOTHING) for c in self.corners_2]

        # match interest points to get candidates features (corresponding pairs)
        for i, d1 in enumerate(self.desc_1):
            for j, d2 in enumerate(self.desc_2):
                if TestImage.compare_descriptors(d1, d2) < self._m_threshold:
                    self._matches_flat.append((i, j))
                    self._matches_idx[(i, j)] = len(self._matches_flat) - 1
                    self._matches[i] = self._matches.get(i, [])
                    self._matches[i].append(j)

        self._corners1_to_sample = list(set([f[0] for f in self._matches_flat]))
        self.n_features = len(self._matches_flat)

        logging.info('\tFound %d candidate features, using %i of %i corners in image 1.' % (self.n_features,
                                                                                            len(self._corners1_to_sample),
                                                                                            len(self.corners_1)))

    _SPACING = 20

    def plot_side_by_side(self, axis, which='gray'):
        img1, img2 = self._dataset
        if which == 'gray':
            spacer = np.zeros((img1.gray.shape[0], RansacImageData._SPACING), dtype=np.uint8) + 255
            axis.imshow(np.concatenate((img1.gray, spacer, img2.gray), axis=1), cmap='gray')
        elif which == 'color':
            spacer = np.zeros((img1.rgb_img.shape[0], RansacImageData._SPACING, 3), dtype=np.uint8) + 255
            axis.imshow(np.concatenate((img1.rgb_img, spacer, img2.rgb_img), axis=1))

        x_offset = img1.gray.shape[1] + RansacImageData._SPACING

        return x_offset

    def plot_features(self, axis, sample_inds=None, draw_image='gray'):
        """
        Draw images side-by-side with all corners detected, and lines of the same color from each corner in image2 to the corresponding corner in image1.
        If "sample_inds" is specified, draw the sample correspondences (points/lines) in green and only show these
        """
        CORNER_MARKERSIZE = 5
        LINEWIDTH = 1

        x_offset = self.plot_side_by_side(axis, draw_image)
        sample_inds

        def _draw_features(f_list, color=None):
            """
            Draw f_lists's points in both images, indicated color or different colors.
            Draw each line between corresponding points in the same color as the point in the first image.
            """
            c1_inds = list(set([f[0] for f in f_list]))

            if sample_inds is None:
                # Draw corners in image 1 in all colors
                points = [axis.plot(*corner_coord, 'o', markersize=CORNER_MARKERSIZE)[0]
                          for corner_coord in self.corners_1]
                point_colors = {i: point.get_color() for i, point in enumerate(points)}

                # Draw corners in image 2 in red
                axis.plot(self.corners_2[:, 0] + x_offset,
                          self.corners_2[:, 1], 'o', markersize=CORNER_MARKERSIZE, color='r')
            else:
                # Draw corners in image 1 green if not in the sample,
                other_inds = [i for i in range(len(self.corners_1)) if i not in c1_inds]
                [axis.plot(*self.corners_1[c, :], 'o', color='g', markersize=CORNER_MARKERSIZE)[0] for c in other_inds]
                # and sample points in red
                _ = [axis.plot(*self.corners_1[c, :], 'o', markersize=CORNER_MARKERSIZE, color='r')[0] for c in c1_inds]
                point_colors = {c1_inds[i]: 'r' for i in range(len(c1_inds))}
                point_colors.update({other_inds[i]: 'g' for i in range(len(other_inds))})

                # Draw corners in image 2 in red/green similarly
                c2_inds = list(set([f[1] for f in f_list]))
                c2_other_inds = [i for i in range(len(self.corners_2)) if i not in c2_inds]
                [axis.plot(self.corners_2[c, 0] + x_offset, self.corners_2[c, 1], 'o', color='g', markersize=CORNER_MARKERSIZE)[0]
                    for c in c2_other_inds]
                _ = [axis.plot(self.corners_2[c, 0] + x_offset, self.corners_2[c, 1], 'o', markersize=CORNER_MARKERSIZE, color='r')[0]
                     for c in c2_inds]

            # and draw the line:
            for (c_ind_1, c_ind_2) in f_list:
                c1, c2 = self.corners_1[c_ind_1], self.corners_2[c_ind_2]
                axis.plot([c1[0], c2[0] + x_offset], [c1[1], c2[1]], color=point_colors[c_ind_1], linewidth=LINEWIDTH)

        if sample_inds is not None:
            features = [self._matches_flat[i] for i in sample_inds]
            _draw_features(features, color='g')
            title = "Minimum feature sample, 3 random correspondences "
        else:

            _draw_features(self._matches_flat)
            title = "All corner correspondences (features) with match score > %.3f" % self._m_threshold
        axis.set_title(title)
        axis.axis('off')


class RansacAffine(RansacModel):
    """
    An affine transform model for RANSAC.
    """
    _N_MIN_FEATURES = 3

    def __init__(self, data, inlier_threshold, training_inds, iter=None):
        """
        Initialize the model with the given features.
        :param data: RansacImageData object containing the data & extracted features.
        :param inlier_threshold: threshold for inliers (will depend on implementation of RansacModel.evaluate)
        :param training_inds: list of indices of the features used to fit this RansacModel.
        :param iter: iteration number, for bookkeeping
        """
        self._training_inds = training_inds
        super().__init__(data, inlier_threshold, training_inds, iter)
        self._fig, self._ax = None, None

    def _corner_coords_from_inds(self, feature_inds):
        train_features = self.data.get_features(indices=feature_inds)
        src_pts = np.array([self.data.corners_1[f[0]] for f in train_features])
        dst_pts = np.array([self.data.corners_2[f[1]] for f in train_features])
        return src_pts, dst_pts

    def _fit(self):
        """
        Fit an affine transform to this model's training data, score all featues, find in/outliers.
        """
        # Extract the coordinates of the features
        logging.info("Fitting model to %d features" % len(self._training_inds))
        logging.info("\tCorner index pairs: %s" % (self.data.get_features(indices=self._training_inds)))
        src_pts, dst_pts = self._corner_coords_from_inds(self._training_inds)
        self._model_params = Affine.from_point_pairs(self.data.size, src_pts, dst_pts)
        scores = self._get_feature_scores()
        self.inlier_mask = scores < self.thresh

    def _get_feature_scores(self):
        """
        Score all features (possible pairs of corners).
        """
        src_inds = [f[0] for f in self.data.get_features()]
        dest_inds = [f[1] for f in self.data.get_features()]
        src_pts = np.array([self.data.corners_1[i] for i in src_inds])
        dst_pts = np.array([self.data.corners_2[i] for i in dest_inds])
        moved_src_pts = self._model_params.apply(src_pts)
        distances = np.linalg.norm(moved_src_pts - dst_pts, axis=1)
        return distances

    @classmethod
    def _animation_setup(cls):
        """
        Set up the animation for plotting each iteration (2 windows, 1x2 subplots)
        """
        current_fig, current_axes = plt.subplots(2, 2)
        best_fig, best_axes = plt.subplots(2, 2)
        cls._AXES = (current_axes, best_axes)
        cls._FIGS = (current_fig, best_fig)

    def plot_nesting_bboxes(self, axis, which='gray'):
        """
        Plot grayscale image1, bounding box of image 2 in green. (i.e. corners of image transformed to image1 space) 
        """
        bbox_2 = np.array([[0, 0],
                           [0, self.data.size[1]],
                           [self.data.size[0], self.data.size[1]],
                           [self.data.size[0], 0],
                           [0, 0]])
        t_inv = self._model_params.invert()
        bbox_2_back = t_inv.apply(bbox_2)
        img1, img2 = self.data._dataset
        if which == 'gray':
            axis.imshow(img1.gray, cmap='gray')
        else:
            axis.imshow(img1.color)
        axis.plot(bbox_2_back[:, 1], bbox_2_back[:, 0], 'g-', linewidth=2)
        axis.set_title('Image 1 with bounding box of image 2')
        axis.axis('off')
        axis.set_aspect('equal')

    def plot_induced_transform(self, axis, which='gray'):
        """
        Grayscale image2, detected corners in image 2,  detected corners of image1 transformed to image 2 space and plotted.
        """
        img1, img2 = self.data
        if which == 'gray':
            axis[0].imshow(img1.gray, cmap='gray')
            axis[1].imshow(img2.gray, cmap='gray')
        else:
            axis[0].imshow(img1.color)
            axis[1].imshow(img2.color)

        # draw the bounding box of image 2 in image 1 space
        img2_corners = np.array([[0, 0], [0, img2.gray.shape[1]], [img2.gray.shape[0], img2.gray.shape[1]],
                                 [img2.gray.shape[0], 0]])
        inv = self._model_params.inverse()
        img2_corners_back = inv.apply(img2_corners)
        axis[0].plot(img2_corners_back[:, 1], img2_corners_back[:, 0], 'g-', linewidth=2)

        # draw the transformed corners of image 1 in image 2 space.
        img1_corners = np.array([[0, 0], [0, img1.gray.shape[1]], [img1.gray.shape[0], img1.gray.shape[1]],
                                 [img1.gray.shape[0], 0]])
        img1_corners = cv2.transform(img1_corners[None, :, :], self._model)[0]
        axis[1].plot(img1_corners[:, 1], img1_corners[:, 0], 'g-')

        axis[0].set_title('Image 1 with bounding box of image 2')
        axis[1].set_title('Image 2 with transformed corners of image 1')
        axis[0].axis('off')
        axis[1].axis('off')

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


def _test(plot=False):
    # Freeze random state
    # np.random.seed(210)
    if plot:
        plt.ion()

    matcher_threshold = 0.4
    size = 500, 500
    noise_level = 0.1
    args = dict(blockSize=8,
                ksize=23,
                k=0.045)

    # data
    img1 = TestImage(size)
    img2, orig_transf = img1.transform(noise_level)
    data = (img1, img2)

    ransac_data = RansacImageData(data, harris_kwargs=args, matcher_threshold=matcher_threshold)
    if False:
        fig, ax = plt.subplots(1)
        ransac_data.plot_features(ax, draw_image='color')
        fig, ax = plt.subplots(1)
        train = ransac_data.get_sample_inds(3)
        ransac_data.plot_features(ax, train)
        plt.show()

    # model
    inlier_threshold = 5.0

    transf = RansacAffine(ransac_data, inlier_threshold, [0, -1, 30])

    if plot:
        fig,ax = plt.subplots(1)
        ransac_data.plot_side_by_side(ax,which='color')
        fig, ax = plt.subplots(1, 2)

        # transf.plot_iteration(ransac_data, transf, transf, is_final=False)
        transf._model_params = orig_transf
        transf.plot_nesting_bboxes(ax[0])
        plt.show()


        plt.pause(0)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test(plot=True)
