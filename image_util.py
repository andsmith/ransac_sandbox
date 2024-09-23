import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_image_pair(size, noise_sigma=0.1, n_rects=10, n_circle_colors=3, n_rect_colors=3):
    """
    Generate an image that will be easy to analyze with a toy feature detector & matcher:
        * Draw 1000 circles of random sizes and colors
        * Draw N rectangles of random sizes and colors on top of the circles
        * Limit the number of colors to make it harder to match features

    Transform the image with a random rotation, translation, and scaling, and add noise.

    :param size: the size of the image (width, height)
    :param noise_sigma: the standard deviation of the noise to add to the image
    :param n_rects: the number of rectangles to draw on top of the circles
    """
    img = np.zeros((size[1], size[0], 3), dtype=np.float32)
    min_dim = np.min(size)

    # make image with circles and rectangles
    circle_rad_range = [min_dim//10, min_dim//5]
    circle_palette = np.random.rand(n_circle_colors, 3)
    circle_colors = circle_palette[np.random.randint(0, n_circle_colors, 1000)]
    for i in range(1000):
        center = np.random.rand(2) * size
        radius = int(np.random.uniform(*circle_rad_range))
        color = circle_colors[i]
        cv2.circle(img, tuple(center.astype(int)), radius,
                   color, -1, lineType=cv2.LINE_AA)

    rect_palette = np.random.rand(n_rect_colors, 3)
    rect_colors = rect_palette[np.random.randint(0, n_rect_colors, n_rects)]
    rect_size_range = [min_dim//15, min_dim//9]
    for i in range(n_rects):
        center = np.random.rand(2) * size
        width, height = np.random.uniform(*rect_size_range, 2)
        color = rect_colors[i]
        cv2.rectangle(img, tuple(center.astype(int)), tuple(
            (center + [width, height]).astype(int)), color, -1, lineType=cv2.LINE_AA)

    # Make an affine transformation
    transform_angle = np.pi/6
    transform_scale = 1.5 + np.random.rand() * 0.5
    transform_translation = np.random.randn(2) * 40
    M = cv2.getRotationMatrix2D(
        (size[0]//2, size[1]//2), transform_angle*180/np.pi, transform_scale)
    M[:, 2] += transform_translation
    img2 = cv2.warpAffine(img, M, size)
    img2 += np.random.randn(*img.shape) * noise_sigma

    # convert to uint8
    img = (255*np.clip(img, 0, 1)).astype(np.uint8)
    img2 = (255*np.clip(img2, 0, 1)).astype(np.uint8)

    return img, img2, {'M': M, 'angle': transform_angle, 'scale': transform_scale, 'translation': transform_translation}


def find_corners(img, margin=0):
    """
    Find corners in an image using the Harris corner detector.
    :param margin: don't return corners within this many pixels of the edge
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Detect corners
    dst = cv2.cornerHarris(gray, 3, 5, k=0.04) # increasing K decreases the number of corners

    # Dilate to mark the corners
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    #dst[dst > 0.01*dst.max()] = [0, 0, 255]

    # Get sub-pixel accuracy on the corners
    ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # Define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # Filter out corners near the edge
    corners = corners[(corners[:, 0] > margin) & (corners[:, 0] < img.shape[1]-margin) &
                      (corners[:, 1] > margin) & (corners[:, 1] < img.shape[0]-margin)] 
    return corners

def _get_corner_windows(img, corners, window_size):
    """
    Get the window around each corner.
    """
    window_size = window_size // 2
    corner_windows = []
    for corner in corners:
        x, y = corner
        x, y = int(x), int(y)
        window = img[y-window_size:y+window_size, x-window_size:x+window_size]
        corner_windows.append(window)

    return corner_windows

def get_window_histogram(window, n_bins = 3, smoothing=0.1):
    """
    Compute a histogram of the window using all three channels, the full range, and the specified number of bins.
    :param window: a 3-channel image window
    :param n_bins: the number of bins to use in each channel
    :param smoothing: a small value to add to each bin to avoid dividing by zero in the histogram comparison
    """
    hist = cv2.calcHist([window], [0, 1, 2], None, [n_bins, n_bins, n_bins], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten() + smoothing
    hist /= np.sum(hist)
    return hist


def compare_histograms(hist1, hist2):
    """
    Use the "symatrized Kullback-Leibler divergence" to compare two histograms.
    (i.e. as the distance between multinoimal distributions)
    """
    return 0.5 * np.sum(hist1 * np.log(hist1/hist2) + hist2 * np.log(hist2/hist1))

def _test_image_dataset():
    """
    Create a dataset, image pair. find corners, extract histograms, and create pairwise similarity matrix.
    Plot the image and its corners.

    Plot the upper right 5x5 corner of the similarity matrix, above each row show the corresponding corner patch, 
    and to the left of each column show the corresponding corner patch.
      
    """
    img, _, _ = make_image_pair((500, 500), noise_sigma=0.1)
    corners = find_corners(img, margin=10)
    n_bins = 3
    window_size = 10
    
    windows = _get_corner_windows(img, corners, window_size)
    
    histograms = np.array([get_window_histogram(window, n_bins=n_bins) for window in windows])
    

    similarity_matrix = np.array([[compare_histograms(hist1, hist2) for hist2 in histograms] for hist1 in histograms])

    # Plot the image and its corners
    plt.imshow(img)
    plt.scatter(corners[:, 0], corners[:, 1], c='r')
    plt.title("Image & Corners")


    # Plot the similarity matrix
    fig, ax = plt.subplots()
    ax.imshow(similarity_matrix, cmap='gray')
    ax.set_title("Similarity Matrix")
    ax.set_xlabel("Image 2 Corners")
    ax.set_ylabel("Image 1 Corners")

    # Plot the upper right 5x5 corner of the similarity matrix
    fig, ax = plt.subplots(5, 5)
    for i in range(5):
        for j in range(5):
            ax[i, j].imshow(windows[i])
            ax[i, j].set_title("Image 1 Corner %i" % i)
            ax[i, j].axis('off')
    for j in range(5):
        ax[0, j].set_title("Image 2 Corner %i" % j, pad=20)
        ax[0, j].imshow(windows[j])
        ax[0, j].axis('off')
    plt.show()


if __name__ == "__main__":
    _test_image_dataset()
