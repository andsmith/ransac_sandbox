import numpy as np
import matplotlib.pyplot as plt
import cv2

def make_image_pair(size, noise_sigma=0.1, n_rects = 10, n_circle_colors =3, n_rect_colors=3):
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
    min_dim = np.min(size)
    circle_rad_range = [min_dim//10, min_dim//5]
    rect_size_range = [min_dim//15, min_dim//9]

    # make image with circles and rectangles
    circle_palette = np.random.rand(n_circle_colors, 3)
    circle_colors = circle_palette[np.random.randint(0, n_circle_colors, 1000)]
    rect_palette = np.random.rand(n_rect_colors, 3)
    rect_colors = rect_palette[np.random.randint(0, n_rect_colors, n_rects)]
    img = np.zeros((size[1], size[0], 3), dtype=np.float32)
    for i in range(1000):
        center = np.random.rand(2) * size
        radius = int(np.random.uniform(*circle_rad_range))
        color = circle_colors[i]
        cv2.circle(img, tuple(center.astype(int)), radius, color, -1, lineType=cv2.LINE_AA)

    for i in range(n_rects):
        center = np.random.rand(2) * size
        width,height = np.random.uniform(*rect_size_range, 2)
        color = rect_colors[i]
        cv2.rectangle(img, tuple(center.astype(int)), tuple((center + [width, height]).astype(int)), color, -1, lineType=cv2.LINE_AA)

    # Make an affine transformation:
    
    transform_angle = np.pi/6
    transform_scale = 1.2
    transform_translation = np.random.randn(2) * 40
    M = cv2.getRotationMatrix2D((size[0]//2, size[1]//2), transform_angle*180/np.pi, transform_scale)
    M[:, 2] += transform_translation    
    img2 = cv2.warpAffine(img, M, size)
    img2 += np.random.randn(*img.shape) * noise_sigma

    return img, img2, {'M': M, 'angle': transform_angle, 'scale': transform_scale, 'translation': transform_translation}






def make_line_data(n_pts, n_outliers, noise_sigma=0.01):
    # make 2d points in a line, add noise & outlier points
    # make sure they are sufficiently far away, so the line is long enough for
    # the demo to be visualized clearly.
    min_sep = 0.6
    while True:
        p0 = np.random.rand(2).reshape(-1, 1)
        p1 = np.random.rand(2).reshape(-1, 1)
        if np.linalg.norm(p1 - p0) > min_sep:
            break
    t = np.random.rand(n_pts)
    line_pts = np.tile(p0, (1, n_pts)) * t + np.tile(p1, (1, n_pts)) * (1. - t)
    line_pts += np.random.randn(n_pts*2).reshape(2, -1)*noise_sigma
    error_pts = np.random.rand(n_outliers*2).reshape(-1, 2)
    params = fit_line(line_pts.T)
    return np.vstack((line_pts.T, error_pts)), params


def fit_line(data):
    """
    Fit a line to the set of points
    :param data: a Nx2 array of points
    :returns: the parameters a, b, c of the line a*x + b*y + c = 0
    """
    x, y = data[:, 0], data[:, 1]
    A = np.vstack((x, y, np.ones_like(x))).T
    return np.linalg.svd(A)[2][-1]


def point_line_distances(pts, a, b, c):
    """
    Return the perpendicular distance between points pts (Nx2) and the line defined
    by a*x + b*y + c = 0.
    """
    n = np.sqrt(a**2 + b**2)
    return np.abs(a*pts[:, 0] + b*pts[:, 1] + c) / n


def plot_line(a, b, c, ax=None, plt_args=(), plt_kwargs={}):
    """
    Add a line to the current plot with parameters a,b,c within current axis limits.
    """
    ax = ax if ax is not None else plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if np.isclose(b, 0):
        # vertical line, points are where y=0 and y=1
        x = [-c/a, -c/a]
        y = [ylim[0], ylim[1]]
    else:
        # otherwise, find points where x=0 and x=1.
        x = [xlim[0], xlim[1]]
        y = [-(c+a*x[0])/b, -(c+a*x[1])/b]
    ax.plot(x, y, *plt_args, **plt_kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax


if  __name__=="__main__":
    img1, img2, transform = make_image_pair((256, 256), noise_sigma=0.1, n_rects=10)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title("Image 1")
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title("Image 2")
    plt.show()
    print(transform )