import numpy as np
import cv2
import pprint
import matplotlib.pyplot as plt
from image_util import TestImage


def _test_similarity_metric(data1, data2, transf, plot=False):
    """
    Create a dataset, image pair. find corners, extract histograms, and create pairwise similarity matrix.
    Show a grid showing up to the first 10 of those corner windows on the top row, and below each, the
           highest & lowest N matching corners from the other image in descending order.
    :param data1: a tuple (QuantizedColorImage, corners) for the first image (and detected corners)
    :param data2: a tuple (QuantizedColorImage, corners) for the second image (and detected corners

    """
    qimg1, corners1 = data1
    qimg2, corners2 = data2
    img1, img2 = qimg1.rgb_img, qimg2.rgb_img

    window_size = 21  # size of the window around each corner to extract a descriptor (pixels)

    # Detect corners and extract descriptors for each corner
    hist1 = np.array([qimg1.get_patch_descriptor(x, y, window_size)
                      for x, y in corners1])
    hist2 = np.array([qimg2.get_patch_descriptor(x, y, window_size)
                      for x, y in corners2])

    # Compute the similarity between each pair of descriptors
    similarity = np.array([[TestImage.compare_descriptors(hist1[i], hist2[j])
                            for j in range(len(hist2))] for i in range(len(hist1))])

    # show the first few corners of image1 and (up to) their top 8 matches in image2, in descending order of similarity
    # Next to each image show the histogram.
    palette = qimg1.palette.astype(np.float32) / 255.

    def _plot_hist(ax, hist):
        """
        Bar graph using the color palette.
        :param hist:  The array of histogram counts (normalized).
        """
        ax.bar(range(len(hist)), hist, color=palette, width=7)
        ax.axis('off')

    def _plot_patch(ax, patch, score=None):
        """
        Show a patch with a score (if given)
        """
        print(patch.shape)  
        ax.imshow((patch))
        ax.axis('off')
        if score is not None:
            ax.text(0, window_size, '%.2f' % score, color='k', fontsize=10, va='top')

    # Plot a few corners from image1 and their best and worst matches in image2

    # Number of examples to show (set to 1 to see the best and worst, etc)
    n_corner_examples = 8
    n_match_examples = 4
    n_worst_examples = 2

    n_cols = np.min([n_corner_examples, len(corners1)]) * 2  # for image and histogram
    n_rows = 1 + n_worst_examples + n_match_examples  # for best and worst matches
    fig, ax = plt.subplots(n_rows, n_cols)

    for i in range(n_corner_examples):

        # show the corner at the top
        x, y = corners1[i]
        window = qimg1.get_patch(x, y, window_size, which='rgb')
        _plot_patch(ax[0, i*2], window)
        _plot_hist(ax[0, i*2+1], hist1[i])

        # show the best
        for j, idx in enumerate(np.argsort(similarity[i])[:n_match_examples]):
            x, y = corners2[idx]
            window = qimg2.get_patch(x, y, window_size, which='rgb')
            _plot_patch(ax[j+1, i*2], window, similarity[i][idx])
            _plot_hist(ax[j+1, i*2+1], hist2[idx])

        # show the worst
        for j, idx in enumerate(np.argsort(similarity[i])[-n_worst_examples:]):
            x, y = corners2[idx]
            window = qimg2.get_patch(x, y, window_size, which='rgb')
            _plot_patch(ax[j+1+n_match_examples, i*2], window, similarity[i][idx])
            _plot_hist(ax[j+1+n_match_examples, i*2+1], hist2[idx])

    # Annotate plots window with separation lines under first row and between best/worst rows.
    # get the y-coordinates between the first and second row in figure coordinates.
    _, y1 = ax[1, 0].transAxes.transform([0, 1.15])
    _, y2 = ax[n_match_examples, 0].transAxes.transform([0, -.20])
    y1 = fig.transFigure.inverted().transform([0, y1])[1]
    y2 = fig.transFigure.inverted().transform([0, y2])[1]
    # x limits are 5% and 95% of the figure width
    x1 = 0.05
    x2 = 0.90
    # annotate

    ax[1, 0].annotate('', xy=(x1, y1), xytext=(
        x2, y1), xycoords='figure fraction', arrowprops=dict(arrowstyle='-', color='k'))
    ax[n_match_examples, 0].annotate('', xy=(x1, y2), xytext=(
        x2, y2), xycoords='figure fraction', arrowprops=dict(arrowstyle='-', color='k'))

    # Add titles to the three row sections (off to the left)
    ax[0, 0].text(-0.1, 0.5, 'image1 corners', fontsize=12, color='r',
                  ha='right', va='center', transform=ax[0, 0].transAxes)
    ax[1, 0].text(-0.1, 0.5, 'best %i matches\nof image2' % (n_match_examples, ), fontsize=12,
                  color='r', ha='right', va='center', transform=ax[1, 0].transAxes)
    ax[n_match_examples+1, 0].text(-0.1, 0.5, 'worst %i matches\nof image2' % (n_match_examples, ), fontsize=12,
                                   color='r', ha='right', va='center', transform=ax[n_match_examples+1, 0].transAxes)

    # remove most space between subplots
    plt.subplots_adjust(wspace=.1, hspace=.4)
    plt.suptitle('Test Corner Matching', fontsize=16)

    # show and return
    plt.show()


if __name__ == "__main__":
    plt.ioff() 

    noise_frac = 0.1
    params = dict(blockSize=4,
                  ksize=3,
                  k=0.06)

    q_img1 = TestImage((400, 400))
    q_img2, transf = q_img1.transform(noise_frac=0.1)

    corners1 = q_img1.find_corners()
    corners2 = q_img2.find_corners()

    # plot both sets of detected corners
    fig, ax = plt.subplots(1, 2)
    q_img1.plot(ax[0], corners=corners1, title="Image 1 corners")
    q_img2.plot(ax[1], corners=corners2, title="Image 2 corners")
    plt.show()



    _test_similarity_metric((q_img1, corners1),
                            (q_img2, corners2),
                            transf=transf, plot=True)
    
    plt.pause(0)
    print("All tests passed.")
