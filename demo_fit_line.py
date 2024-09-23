from ransac import solve_ransac
from fit_line import RansacLineData, RansacLine
from util import make_line_data, plot_line
import matplotlib.pyplot as plt


def demo(n_line_pts, n_outliers, **ransac_args):
    """
    Demonstrate RANSAC, finding a line among a set of 2d points.

    :param n_line_pts: number of points in the line
    :param n_outliers: number of outlier points
    :param ransac_args: additional parameters for solve_ransac()
    """
    pts, params = make_line_data(n_line_pts, n_outliers)
    data = RansacLineData(pts)
    result = solve_ransac(data, RansacLine, **ransac_args)

    print('\n\nRANSAC found a solution on iteration %i:' %
          (result['best_iter'], ))
    print('\tInliers: %i of %i (%.1f %%)' % ((result['inliers'].sum()),
                                             len(result['inliers']),
                                             100*(result['inliers'].mean())))
    print("\nEstimated params of line:\n\t%.3f*x + %.3f*y + %.3f = 0" %
          tuple(result['best_model'].get_params()))
    print("\nTrue params of line:\n\t%.3f*x + %.3f*y + %.3f = 0" % tuple(params))

    if 'animate_pause_sec' in ransac_args and ransac_args['animate_pause_sec'] is not None:
        plt.ioff()
        _, ax = plt.subplots()
        # plot the final model & best model
        data.plot(inlier_mask=result['inliers'], ax=ax)
        result['best_model'].plot(ax=ax, plt_kwargs={'label': 'best model (iter %i)' % (result['best_iter'],),
                                                     'linewidth': 1},
                                  plt_args=['b-'], show_features_used=True)
        result['final_model'].plot(ax=ax, plt_kwargs={'label': 'final model'},
                                   plt_args=['g-'], show_features_used=False)
        plt.title('RANSAC result after %i iterations' %
                  (ransac_args['max_iter'],))
        least_squares_lines = RansacLine(result['features'])
        least_squares_lines.plot(ax=ax, plt_args=['k--'], plt_kwargs={'label': 'Fit to all data'}, show_features_used=False)
        ax.legend()
        plt.show()


if __name__ == '__main__':
    demo(n_line_pts=100,
         n_outliers=100,
         max_error=0.03,
         max_iter=20,
         animate_pause_sec=0.3  # 0 to pause between iterations, None to disable plotting
         )
