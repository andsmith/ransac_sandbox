import numpy as np
import cv2
from scipy.interpolate import interp2d


class Affine(object):
    """
    Utilities for daeling with affine transforms in 2d.
    """

    def __init__(self, size, rotation_rad,  scale_factor, translation_px=None):
        """
        Initialize an affine transform of the 2d plane. 
        Constrain scale to be the same in x and y (no shearing), and rotation to be about the center.

        Note: this constrained set is not closed under inversion, so affine.invert() will not break down cleanly this way.
          (THe matrices are invertable, so points can still be transformed back and forth, but some parameters will be None/invalid.)
        """

        self.size = np.array(size)
        self.rotation_rad = rotation_rad
        self.translation_px = np.array(translation_px) if translation_px is not None else None
        self.scale_factor = scale_factor

        self.M = cv2.getRotationMatrix2D(
            (int(size[0]//2), int(size[1]//2)), rotation_rad/np.pi*180, scale_factor)
        if self.translation_px is not None:
            self.M[:, 2] += self.translation_px
        self.M = np.vstack((self.M, [0, 0, 1]))  # for homogeneous coordinates

    def __eq__(self, other):
        """
        Return true if the two matrices are close enough to equal.
        """
        return np.allclose(self.M, other.M)

    def __str__(self):
        return 'Affine: rotation: {:.2f} (deg), translation: {}, scale: {:.2f}'.format(
            self.rotation_rad/np.pi*180, self.translation_px, self.scale_factor)

    @staticmethod
    def _decompose_transform(M):
        """
        Decompose an affine transform matrix into its components (those recoverable given this affine subset).
        (translation is not recoverable because rotation will be about a different point -- TODO: add this)
        :param M: 3x3 affine transform matrix
        :return: angle (radians), scale)
        """
        # extract the angle, scale, and translation from the matrix
        a = M[:, :2]
        scale = np.mean(np.linalg.norm(a, axis=0))  # x and y have the same scale
        angle = np.arctan2(a[0, 1], a[0, 0])  # "rotation" parts of the matrix
        print(M[:2,2    ])
        if angle<0:
            angle += 2*np.pi
        return angle, scale

    @staticmethod
    def from_random(size, restricted=False):

        size = np.array(size)
        if restricted:
            #  Good range for demo
            transform_angle = np.pi/3 + np.random.rand() * np.pi/3
            transform_scale = 1.4 + np.random.rand() * 0.3
            transform_translation = np.random.randn(2) * 40
        else:
            print("CONSTANT TESTING PARAMS")
            transform_angle = np.pi/6# np.random.rand() * 2 * np.pi
            transform_scale =  1.#0.5 + np.random.rand() * 2
            transform_translation = np.random.randn(2) * 0
        return Affine(size, transform_angle, transform_scale, transform_translation)

    @staticmethod
    def from_point_pairs(size, src_pts, dst_pts):
        """
        Fit an affine transform to the given 2d points.
        :param src_pts: Nx2 array of source points
        :param dst_pts: Nx2 array of destination points
        """
        # Add a column of ones to the src_pts
        src_pts = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))

        # Solve the least squares problem
        params = np.linalg.lstsq(src_pts, dst_pts, rcond=None)[0].T
        params_h = np.vstack((params, [0, 0, 1]))  # for homogeneous coordinates

        angle, scale = Affine._decompose_transform(params)
        r = Affine(size, angle, scale)
        r.M = params_h
        return r

    def invert(self):
        """
        Invert self.
        """
        M = np.linalg.inv(self.M)
        # estimate rotation from matrix, other params will be invalid
        angle, scale = Affine._decompose_transform(M)

        r = Affine(self.size, angle, scale)
        r.M = M
        return r

    def apply(self, pts):
        """
        Apply the affine transform to the given points.
        """
        pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
        pts_out = np.dot(self.M, pts_h.T).T
        pts_out = pts_out[:, :2] / pts_out[:, 2][:, None]
        return pts_out
    
    def warp_image(self, image):
        """
        Apply the affine transform to the given image.
        """
        #return cv2.warpAffine(img, self.M[:2], self.size)
        xg, yg = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]))
        fill = image[0,0]
        inv = np.linalg.inv(self.M)
        x = xg.flatten()
        y = yg.flatten()
        pts = np.vstack((x, y)).T
        pts_transformed = inv.apply(pts)
        pts_transformed = pts_transformed.reshape(self.size[1], self.size[0], 2)
        f = interp2d(x,y, image, kind='nearest', fill_value=fill)
        return f(pts_transformed[:,:,0], pts_transformed[:,:,1])



def test_affine(plot=True):
    size = (100, 100)
    aff = Affine.from_random(size)
    pts = np.random.rand(20, 2) * size
    pts_transformed = aff.apply(pts)

    # fit an affine transform the (point, transformed_point) pairs, and recover the original points.
    aff_recovered = Affine.from_point_pairs(size, pts, pts_transformed)
    pts_transformed_w_aff_rec = aff_recovered.apply(pts)

    # invert the original transform & move the points back.
    aff_inverted = aff.invert()
    pts_returned = aff_inverted.apply(pts_transformed)

    # finally invert the inverted transform and make sure it is the same as the original transform.
    aff_inverted_inverted = aff_inverted.invert()

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(pts[:, 0], pts[:, 1], 'o')
        plt.plot(pts_transformed[:, 0], pts_transformed[:, 1], 's')
        plt.plot(pts_returned[:, 0], pts_returned[:, 1], 'x')
        plt.legend(['Original', 'Transformed', 'Returned'])
        plt.show()

    # Print original, recovered and double-inverted transforms:
    print('Original:', aff)
    print('Recovered:', aff_recovered)
    print('Twice inverted:', aff_inverted_inverted)
    

    # make sure estimated affine transform is close to the original, and points were moved the same by both
    assert aff == aff_recovered, 'Affine transform failed to estimate correctly.'
    assert np.allclose(pts_transformed, pts_transformed_w_aff_rec), 'Affine transform failed to recover points correctly.'

    # make sure transformed points are mapped back to the original points.
    assert np.allclose(pts, pts_returned), 'Affine transform failed to invert correctly.'

    # make sure doubly inverted is the same as the original:
    assert aff == aff_inverted_inverted, 'Affine transform failed to invert twice and remain constant.'

    print('Affine transform passed.')


if __name__ == '__main__':
    test_affine()
    print('All tests passed.')
