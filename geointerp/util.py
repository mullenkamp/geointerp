"""
Utility functions for coordinate conversion and grid operations.
"""
import numpy as np


def grid_coords_to_index_params(coord_arrays):
    """
    Given a tuple of 1D coordinate arrays defining a regular grid,
    compute the origin and spacing for each dimension.

    Parameters
    ----------
    coord_arrays : tuple of 1D ndarray
        Each element is a sorted 1D array of coordinates for one grid dimension.

    Returns
    -------
    origins : tuple of float
        Minimum value along each dimension.
    spacings : tuple of float
        Grid spacing along each dimension.
    """
    origins = tuple(float(arr.min()) for arr in coord_arrays)
    spacings = tuple(float(np.median(np.diff(arr))) for arr in coord_arrays)
    return origins, spacings


def coords_to_indices(points, origins, spacings):
    """
    Convert N-dimensional coordinates to fractional array indices.

    Parameters
    ----------
    points : ndarray of shape (M, ndim)
        Target points in coordinate space.
    origins : tuple of float
        Origin (minimum) for each dimension.
    spacings : tuple of float
        Spacing for each dimension.

    Returns
    -------
    indices : ndarray of shape (ndim, M)
        Fractional array indices suitable for scipy map_coordinates.
    """
    ndim = len(origins)
    indices = np.empty((ndim, points.shape[0]), dtype=float)
    for d in range(ndim):
        indices[d] = (points[:, d] - origins[d]) / spacings[d]
    return indices


def find_nearest(array, value):
    """
    Find the element in array nearest to value.

    Parameters
    ----------
    array : array-like
        Array to search.
    value : float
        Value to find nearest match for.

    Returns
    -------
    scalar
        The element of array closest to value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
