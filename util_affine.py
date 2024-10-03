import numpy as np
from scipy.interpolate import interp2d


class Affine(object):
    """
    Utilities for daeling with affine transforms in 2d.
    """

    def __init__(self, rotate, scale, translate):
        """
        Initialize an affine transform of the 2d plane. 
        Constrain scale to be the same in x and y (no shearing), and rotation to be about the origin
        :param rotate: rotation angle in radians
        :param scale: scaling factor
        :param translate: translation vector in pixels
        """

        self.rotate = rotate
        self.translate = np.array(translate) if translate is not None else None
        self.scale = scale

    def __eq__(self, other):
        """
        Return true if the two matrices are close enough to equal.
        """
        return np.allclose([self.rotate, self.scale],
                            [other.rotate, other.scale]) and \
            np.allclose(self.translate, other.translate)

    def __str__(self):
        return 'Affine: rotation: {:.2f} (deg), translation: {}, scale: {:.2f}'.format(
            self.rotate, self.translate, self.scale)

    @staticmethod
    def from_random(restricted=False):
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
        return Affine(transform_angle, transform_scale, transform_translation)

    @staticmethod
    def from_point_pairs(src_pts, dst_pts):
        """
        Fit an affine transform to the given 2d points.
        :param src_pts: Nx2 array of source points
        :param dst_pts: Nx2 array of destination points
        """
        # Add a column of ones to the src_pts
        src_pts = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))

        # Solve the least squares problem
        m_inv = np.linalg.lstsq(src_pts, dst_pts, rcond=None)[0].T

        angle, scale, translate = Affine._decompose_transform(m_inv)
        return Affine(angle, scale, translate)
    @staticmethod
    def _decompose_transform(m):
        """
        Decompose the given 2x3 affine matrix into rotation, scale and translation.
        [x','y'].T = M [x,y,1].T 
        """
        # Extract scale, x and y are the same so use the average
        scale = np.mean([np.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2),
                            np.sqrt(m[1, 0] ** 2 + m[1, 1] ** 2)])

        # Extract the rotation
        angle = np.arctan2(m[1, 0], m[0, 0])

        # Extract the translation
        translate = m[:, 2]

        return angle, scale, translate
    
    def invert(self):
        """
        Invert self.
        """
        angle = - self.rotate
        scale = 1. / self.scale
        m = self.get_matrix()

        # Need to scale & rotate the translation vector to undo it properly:
        translate = -np.dot(m[:2, :2].T, self.translate)
        return Affine(angle, scale, translate)

    def get_matrix(self):
        """
        Return the 2x3 matrix representation of the affine transform.
        """
        m = np.array([[np.cos(self.rotate), -np.sin(self.rotate), self.translate[0]],
                      [np.sin(self.rotate), np.cos(self.rotate), self.translate[1]]])
        m[:2] *= self.scale
        return m

    def apply(self, pts):
        """
        Apply the affine transform to the given points.
        """
        m = self.get_matrix()
        pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
        pts_out = np.dot(m, pts_h.T).T
        return pts_out
    
    def warp_image(self, image):
        """
        Apply the affine transform to the given image.
        (find where coords in bounding box of new image originate in the old image and use the nearest pixel value)
        """
        #return cv2.warpAffine(img, self.M[:2], self.size)
        size = image.shape[1], image.shape[0]

        # original & new bounding box coords
        xg, yg = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
        x_centered, y_centered = xg.astype(float) - size[0] / 2., yg.astype(float) - size[1] / 2.

        fill = image[0,0]

        # invert transform, see where each new image pixel gets its value
        inv = self.invert()
        x = x_centered.flatten()
        y = y_centered.flatten()
        pts = np.vstack((x, y)).T
        pts_transformed = inv.apply(pts)

        # un-center
        pts_transformed[:, 0] += size[0] / 2.
        pts_transformed[:, 1] += size[1] / 2.

        # equivalent to nearest-neighbor interpolation
        pts_transformed= np.round(pts_transformed).astype(int).reshape(-1, 2)

        # in/out of bounds
        valid = (pts_transformed[:, 0] >= 0) & (pts_transformed[:, 0] < size[0]) & \
                (pts_transformed[:, 1] >= 0) & (pts_transformed[:, 1] < size[1])
        
        # fill in valid/invalid values
        if len(image.shape)==2:
            img_flat = np.zeros(size[0] * size[1], dtype=image.dtype)
            img_flat[valid] = image[pts_transformed[valid, 1], pts_transformed[valid, 0]]
            img_flat[~valid] = fill
            img = img_flat.reshape(size)
        else:
            img_flat = np.zeros((size[0] * size[1], image.shape[2]), dtype=image.dtype)
            for i in range(image.shape[2]):
                img_flat[valid, i] = image[pts_transformed[valid, 1], pts_transformed[valid, 0], i]
            img_flat[~valid] = fill
            img = img_flat.reshape(size[1], size[0], image.shape[2])
        
        return img


def test_affine(plot=True):
    size = (100, 100)   
    aff = Affine.from_random()
    pts = np.random.rand(20, 2) * size
    pts_transformed = aff.apply(pts)

    # fit an affine transform the (point, transformed_point) pairs, and recover the original points.
    aff_recovered = Affine.from_point_pairs(pts, pts_transformed)
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
    print(pts_transformed.shape, pts_transformed_w_aff_rec.shape)
    assert np.allclose(pts_transformed, pts_transformed_w_aff_rec), 'Affine transform failed to recover points correctly.'

    # make sure transformed points are mapped back to the original points.
    assert np.allclose(pts, pts_returned), 'Affine transform failed to invert correctly.'

    # make sure doubly inverted is the same as the original:
    assert aff == aff_inverted_inverted, 'Affine transform failed to invert twice and remain constant.'

    print('Affine transform passed.')


if __name__ == '__main__':
    test_affine()
    print('All tests passed.')
