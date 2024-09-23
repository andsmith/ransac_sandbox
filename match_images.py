"""
Demo of RANSAC algorithm, finding the affine transform between two images.
"""
import numpy as np
import matplotlib.pyplot as plt
from util import make_image_pair, fit_affine_transform
from ransac import RansacModel, RansacDataFeatures, solve_ransac
import cv2


class RansacImageData(RansacDataFeatures):
    """
    A dataset of two images for RANSAC, with an affine transform model.
    """

    def __init__(self, data):
        """
        :param data: a tuple (img1, img2) of images
        """
        super().__init__(data)

    def _extract_features(self):
        """
        Find interest points in both images, extract their descriptors, match them.
        A "feature" for RANSAC is a pair of points with a high matching score.
        Create a list of such pairs.
        """
        # Find interest points in both images
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(self._dataset[0], None)
        kp2, des2 = orb.detectAndCompute(self._dataset[1], None)

        # Match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # Find pairs of points with a high matching score
        good_matches = []
        for m in matches:
            if m.distance < 50:
                good_matches.append((kp1[m.queryIdx].pt, kp2[m.trainIdx].pt))

        self._features = good_matches

    def plot(self, ax=None, inlier_mask=None):
        """
        Plot the images one above the other, on the same axis.
        Draw dots on all 
        :param inlier_mask: boolean array indicating which points are inliers
        """
        ax = ax if ax is not None else plt.gca()

        # Plot all points
        if inlier_mask is None:
            ax.imshow(self._dataset[0], cmap='gray', alpha=0.5, label='image 1')
            ax.imshow(self._dataset[1], cmap='gray', alpha=0.5, label='image 2')
        else:
            ax.imshow(self._dataset[0], cmap='gray', alpha=0.5, label='image 1')
            ax.imshow(self._dataset[1], cmap='gray', alpha=0.5, label='image 2')
            ax.imshow(self._dataset[1], cmap='gray', alpha=0.5, label='image 2')

        return ax