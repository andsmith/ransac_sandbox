"""
Class for representing synthetic images for testing homography estimation.
Uses Harris corner detection to find corners in the image, and extracts local feature descriptors (histograms)
for each corner.  The descriptors can be compared using the Symetrized Kulback-Leibler divergence.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from make_test_imgs import draw_img, transform_img


class TestImage(object):
    """
    Class for synthetic image homography testing.

    Generate a test image with a known transformation between it and a second image.
    Perform (Harris) corner detection tuned for this kind of image.
    Extract local feature descriptors (represented internally as histograms, somewhat invariant to this restricted set of images)
    Compare two descriptors (using Symetrized Kulback-Leibler divergence) to return a score.
    """

    def __init__(self, size, n_rects=10, n_circle_colors=30, n_rect_colors=3):
        """
        :param size: the size of the image (width, height)
        :param n_rects: the number of rectangles to draw on top of the circles
        :param n_circle_colors: the number of colors to use for the circles
        :param n_rect_colors: the number of colors to use for the rectangles
        """
        self.size=size
        self.img, self.palette = draw_img(size, n_rects, n_circle_colors, n_rect_colors)
        self._init()

    def _init(self):
        self.n_colors = self.palette.shape[0]
        self.rgb_img = self.palette[self.img].astype(self.palette.dtype)
        self.gray = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def transform(self, noise_frac):
        """
        Create anew TestImage from self.
        :param image: a 2d array of pixel values, each in [0, palette.shape[0])
        :param noise_frac: the fraction of pixels to randomly change
        """
        img2, transf  = transform_img(self.img, noise_frac, max_color_ind=self.n_colors)
        r = TestImage(size=self.size)
        r.img = img2
        r.palette = self.palette
        r._init()
        return r, transf

    @staticmethod
    def compare_descriptors(hist1, hist2):
        return 0.5 * (np.sum(hist1 * np.log(hist1 / hist2)) + np.sum(hist2 * np.log(hist2 / hist1)))

    def get_patch(self, x, y, patch_size, which='index'):
        if which not in ['index', 'rgb']:
            raise ValueError("which must be one of 'index' or 'rgb'")
        patch_size = patch_size // 2
        x, y = int(x), int(y)
        src = self.img if which == 'index' else self.rgb_img
        return src[y-patch_size:y+patch_size, x-patch_size:x+patch_size]

    def get_patch_descriptor(self, x, y, patch_size, smoothing=0.1):
        """
        Get the histogram of the patch around a point.
        :param x: x-coordinate of the center of the patch
        :param y: y-coordinate of the center of the patch
        :param patch_size: the size of the patch (square, will cover x,y +/- patch_size//2)
        :param smoothing: a small value to add to each bin to avoid dividing by zero in the histogram comparison
        """
        window = self.get_patch(x, y, patch_size)
        hist = np.array([np.sum(window == i) for i in range(self.n_colors)]).reshape(-1)
        hist += smoothing
        hist /= np.sum(hist)
        return hist

    def find_corners(self, margin=0, harris_kwargs=None):
        """
        Find corners in an image using the Harris corner detector.
        :param margin: don't return corners within this many pixels of the edge
        :param harris_kwargs: additional keyword arguments to pass to cv2.cornerHarris
            - blockSize: the size of the window to consider for each corner
            - ksize: the size of the Sobel kernel to use for the derivative (must be odd)
            - k: a free parameter in the Harris detector equation (increasing K decreases the number of corners)

        :returns: an Nx2 array of corner coordinates
        """
        default_harris_kwargs = dict(blockSize=2,
                                     ksize=3,
                                     k=0.04)
        if harris_kwargs is not None:
            default_harris_kwargs.update(harris_kwargs)

        dst = cv2.cornerHarris(self.gray, **default_harris_kwargs)

        # Dilate to mark the corners
        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value, it may vary depending on the image.
        # dst[dst > 0.01*dst.max()] = [0, 0, 255]

        # Get sub-pixel accuracy on the corners
        ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # Define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(self.gray, np.float32(
            centroids), (5, 5), (-1, -1), criteria)

        # Filter out corners near the edge
        corners = corners[(corners[:, 0] > margin) & (corners[:, 0] < self.img.shape[1]-margin) &
                          (corners[:, 1] > margin) & (corners[:, 1] < self.img.shape[0]-margin)]
        return corners


def _test_corner_detector():
    """
    Create a 400x400 test image pair, find corners, and show the results in a plot.
    return images and detected corners
    """
    args = dict(blockSize=2,
                ksize=3,
                k=0.04)

    q_img1 = TestImage((400, 400))
    q_img2, transf = q_img1.transform(0.02)
    img1, img2 = q_img1.rgb_img, q_img2.rgb_img

    # _disp_image_pair(q_img1, q_img2)

    # find corners in both images & plot them
    corners1 = q_img1.find_corners(margin=10, harris_kwargs=args) 
    corners2 = q_img2.find_corners(margin=10, harris_kwargs=args)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax[0].scatter(corners1[:, 0], corners1[:, 1], color='r', s=50)
    ax[0].set_title('Image 1')
    ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax[1].scatter(corners2[:, 0], corners2[:, 1], color='r', s=50)
    ax[1].set_title('Image 2')

    plt.show()

    return (q_img1, corners1), (q_img2, corners2)


def _test_similarity_metric(data1, data2):
    """
    Create a dataset, image pair. find corners, extract histograms, and create pairwise similarity matrix.
    Show a grid showing up to the first 10 of those corner windows on the top row, and below each, the
           highest & lowest N matching corners from the other image in descending order.
    :param data1: a tuple (QuantizedColorImage, corners) for the first image (and detected corners)
    :param data2: a tuple (QuantizedColorImage, corners) for the second image (and detected corners

    """
    n_color_bins = 5
    n_cols_max = 8

    # Number of examples to show (set to 1 to see the best and worst, etc)
    n_ex = 2
    qimg1, corners1 = data1
    qimg2, corners2 = data2
    img1, img2 = qimg1.rgb_img, qimg2.rgb_img

    window_size = 20
    # Extract histograms for each corner
    windows1 = _get_corner_windows(img1, corners1, window_size)
    windows2 = _get_corner_windows(img2, corners2, window_size)

    hist1 = np.array([get_window_histogram(window, n_bins=n_color_bins)
                     for window in windows1])
    hist2 = np.array([get_window_histogram(window, n_bins=n_color_bins)
                     for window in windows2])

    # Compare histograms
    similarity = np.array([[compare_histograms(h1, h2)
                          for h2 in hist2] for h1 in hist1])

    # show the first few corners of image1 and (up to) their top 8 matches in image2, in descending order of similarity
    n_cols = np.min([n_cols_max, len(corners1)])
    n_rows = 1 + 2 * n_ex
    fig, ax = plt.subplots(n_rows, n_cols)
    for i in range(n_cols):
        ax[0, i].imshow(cv2.cvtColor(windows1[i], cv2.COLOR_BGR2RGB))
        ax[0, i].axis('off')

        def _plot_match(j, idx):
            ax[j, i].imshow(cv2.cvtColor(windows2[idx], cv2.COLOR_BGR2RGB))
            ax[j, i].axis('off')
            ax[j, i].text(0, window_size, '%.2f' %
                          similarity[i, idx], color='k', fontsize=10, va='top')

        # show the best
        for j, idx in enumerate(np.argsort(similarity[i])[:n_ex]):
            _plot_match(j+1, idx)

        # show the worst
        for j, idx in enumerate(np.argsort(similarity[i])[-n_ex:]):
            _plot_match(j+n_ex+1, idx)

    # Annotate plots window with separation lines under first row and between best/worst rows.
    # get the y-coordinates between the first and second row in figure coordinates.
    _, y1 = ax[1, 0].transAxes.transform([0, 1.15])
    _, y2 = ax[n_ex, 0].transAxes.transform([0, -.20])
    y1 = fig.transFigure.inverted().transform([0, y1])[1]
    y2 = fig.transFigure.inverted().transform([0, y2])[1]
    # x limits are 5% and 95% of the figure width
    x1 = 0.05
    x2 = 0.90
    # annotate
    ax[1, 0].annotate('', xy=(x1, y1), xytext=(
        x2, y1), xycoords='figure fraction', arrowprops=dict(arrowstyle='-', color='k'))
    ax[n_ex, 0].annotate('', xy=(x1, y2), xytext=(
        x2, y2), xycoords='figure fraction', arrowprops=dict(arrowstyle='-', color='k'))

    # Add titles to the three row sections (off to the left)
    ax[0, 0].text(-0.1, 0.5, 'image1 corners', fontsize=12, color='r',
                  ha='right', va='center', transform=ax[0, 0].transAxes)
    ax[1, 0].text(-0.1, 0.5, 'best %i matches\nof image2' % (n_ex, ), fontsize=12,
                  color='r', ha='right', va='center', transform=ax[1, 0].transAxes)
    ax[n_ex+1, 0].text(-0.1, 0.5, 'worst %i matches\nof image2' % (n_ex, ), fontsize=12,
                       color='r', ha='right', va='center', transform=ax[n_ex+1, 0].transAxes)

    # remove most space between subplots
    plt.subplots_adjust(wspace=.1, hspace=.4)

    # show and return
    plt.show()


if __name__ == "__main__":
    plt.ion()  # do all windows on startup
    i1, i2 = _test_corner_detector()
    #_test_similarity_metric(i1, i2)
    plt.pause(0)
