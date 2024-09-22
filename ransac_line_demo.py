import numpy as np
import matplotlib.pyplot as plt

from util import make_line_data

def point_line_distances(pts, p0, p1):
    """
    Return the perpendicular distance between points pts (Nx2) and the line 
    through points p0 and p1 (each 1x2).
    """
    v = p1 - p0  # vector from p0 to p1
    u = v / np.linalg.norm(v)  # unit vector in the direction of the line
    n = np.array([-u[1], u[0]])  # perpendicular vector to the line
    w = pts - p0  # vector from p0 to pts
    return np.abs(np.dot(w, n))  # distance from pts to the line


def fit_line(pts):
    """
    Find the parameters a, b, c of the line a*x + b*y + c = 0 that best fits the points.
    (minimizing the sum of squared perpendicular distances)
    """
    x, y = pts[:, 0], pts[:, 1]
    A = np.vstack((x, y, np.ones_like(x))).T
    return np.linalg.svd(A)[2][-1]


def construct_line_endpoints(a, b, c):
    """
    Return two points on the line a*x + b*y + c = 0.
    """
    if np.isclose(b, 0):
        # vertical line, points are where y=0 and y=1
        return np.array([-c/a, 0]), np.array([-c/a, 1])
    # otherwise, find points where x=0 and x=1.
    return np.array([0, -c/b]), np.array([1, -(c+a)/b])


def plot_line(a, b, c, *args, **kwargs):
    """
    Add a line to the current plot with parameters a,b,c within current axis limits.
    """
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    if np.isclose(b, 0):
        # vertical line, points are where y=0 and y=1
        x = [-c/a, -c/a]
        y = [ylim[0], ylim[1]]
    else:
        # otherwise, find points where x=0 and x=1.
        x = [xlim[0], xlim[1]]
        y = [-(c+a*x[0])/b, -(c+a*x[1])/b]
    plt.plot(x, y, *args, **kwargs)


def plot_iteration(pts, ctrl_pts,line_params, inliers, iter, inliers_req, legend=True):
    """
    Plot the current iteration of RANSAC:
        All points, which are inliers/outliers, the line through the control points, and status.
    :param pts: N x 2 array of points
    :param ctrl_pts: points to draw as the control points, or (None, None)
    :param line_params: the parameters of the line to draw, or None
    :param inliers: boolean array indicating which points are inliers
    :param iter: string, the current iteration number, or "final", etc.
    :param inliers_req: the minimum number of inliers required for the fit condition
    """
    p0, p1 = ctrl_pts
    outliers = np.logical_not(inliers)
    fig, ax = plt.subplots()
    ax.plot(pts[outliers][:, 0], pts[outliers][:, 1], 'r.', label='outliers (%i)'%np.sum(outliers))
    ax.plot(pts[inliers][:, 0], pts[inliers][:, 1], 'g.', label='inliers (%i) '%np.sum(inliers))
    
    
    if ctrl_pts[0] is not None:
        # if there are control points, plot them and the line through themq
        draw_rad = np.max((np.max(pts[:, 0]) - np.min(pts[:, 0]),
                np.max(pts[:, 1]) - np.min(pts[:, 1]))) / 25.0
        ax.plot(p0[0], p0[1], 'bo', markersize=5, label="random sample points (2)")
        ax.plot(p1[0], p1[1], 'bo', markersize=5)
        ax.add_patch(plt.Circle(p0, radius=draw_rad, facecolor='none', edgecolor='b'))
        ax.add_patch(plt.Circle(p1, radius=draw_rad, facecolor='none', edgecolor='b'))

    if line_params is not None:
        plot_line(*line_params, "b--", label='fit line')


    ax.set_title('RANSAC iter %s found %i inliers (need >= %i)\nQ to continue...' % (iter,np.sum(inliers), inliers_req))
    if legend:
        ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    plt.show()

def find_line_ransac(pts, max_dist=0.01, min_inlier_count=10, max_iter=10000, animate=True):
    """
    Use RANSAC to fit a line of 2d points.
    Stops when min_inlier_frac of the points are inliers.

    :param pts:  N x 2 array of points
    :param max_dist: float, points farther away from the line are outliers
    :param min_inlier_count: minimum number of inliers required for a "fit"

    :returns: (a,b,c) the parameters of the line ax + by + c = 0,
              inliers: a boolean array indicating the consensus set, 
              error: the sum of squared perpendicular distances of all points to the line, and
              n_iter: number of iterations taken.
    """
    line_params = None
    success = False

    for i in range(max_iter):
        # pick two random points.  TODO: ensure a minimum separation between these points.
        idx = np.random.choice(pts.shape[0], 2, replace=False)
        p0, p1 = pts[idx]
        # find inliers
        dists = point_line_distances(pts, p0, p1)
        inliers = dists < max_dist
        inlier_sse = np.sum(dists[inliers]**2)


        if np.sum(inliers) >= min_inlier_count:
            # fit line to inliers, calculate error
            success = True
            line_params = fit_line(pts[inliers])
            line_points = construct_line_endpoints(*line_params)  # best fit line using all inliers
            line_dists = point_line_distances(pts[inliers], *line_points)
            error = np.sum(line_dists**2)
    
        if animate:
            # Draw the current iteration:

            p0p1_line_params = fit_line(np.array([p0, p1]))
            plot_iteration(pts, (p0, p1), p0p1_line_params, inliers, i, min_inlier_count)

        if success:
            return line_params, inliers, error, i

    return None, None, None, max_iter


def demo(n_line_pts, n_outliers, min_inlier_count, **kwargs):
    """
    Demonstrate RANSAC, finding a line among a set of 2d points.

    :param n_line_pts: number of points in the line
    :param n_outliers: number of outlier points
    :param kwargs: additional parameters for find_line_ransac()
    """
    pts = make_line_data(n_line_pts, n_outliers)
    line_params, inliers, sse, n_iter = find_line_ransac(pts, min_inlier_count=min_inlier_count, **kwargs)
    if line_params is None:
        print('RANSAC failed to find a line')
        return
    print('RANSAC found a solution on iteration %i:'% (n_iter, ))
    print('\tbest fit params: %.5f x + %.5f y + %3f = 0' % tuple(line_params))
    print('\tinlier rmse:  %.5f' % (np.sqrt(sse/(n_line_pts+n_outliers)),))
    plot_iteration(pts, (None, None), line_params, inliers, "(final)", min_inlier_count,legend=False)


if __name__ == '__main__':
    demo(n_line_pts=100,
         n_outliers=100,
         min_inlier_count=70,
         max_dist=0.03,
         max_iter=10000)
