import numpy as np

def pts_addone(points):
    """
    points: npoints x 3
    """
    points_shape = [item for item in points.shape[:-1]]
    ones = np.ones(points_shape + [1])
    return np.concatenate([points, ones], axis=-1)


