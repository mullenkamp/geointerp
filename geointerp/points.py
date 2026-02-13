"""
PointInterpolator for scattered/irregular point data.
"""
import numpy as np
from pyproj import CRS, Transformer
from scipy.spatial import Delaunay
from scipy.interpolate import (
    LinearNDInterpolator, NearestNDInterpolator, CloughTocher2DInterpolator,
)


class PointInterpolator:
    """
    Interpolator for scattered/irregular point data.

    Uses scipy.interpolate machinery (Delaunay triangulation, LinearNDInterpolator).
    Methods return precomputed callables for repeated application to time steps.

    Parameters
    ----------
    from_crs : int, str, or None
        CRS of source point coordinates.
    """

    def __init__(self, from_crs=None):
        self._from_crs = CRS.from_user_input(from_crs) if from_crs is not None else None

    def _make_transformer(self, to_crs):
        """
        Build pyproj Transformers between self._from_crs and to_crs.

        Returns
        -------
        fwd : Transformer or None
            Forward transform (from_crs -> to_crs).
        inv : Transformer or None
            Inverse transform (to_crs -> from_crs).
        out_is_geographic : bool
            Whether the output CRS is geographic (lat/lon).
        """
        if to_crs is None or self._from_crs is None:
            is_geo = self._from_crs.is_geographic if self._from_crs is not None else False
            return None, None, is_geo
        to_crs_obj = CRS.from_user_input(to_crs)
        fwd = Transformer.from_crs(self._from_crs, to_crs_obj, always_xy=True)
        inv = Transformer.from_crs(to_crs_obj, self._from_crs, always_xy=True)
        return fwd, inv, to_crs_obj.is_geographic

    def _transform_points_xy(self, points, transformer):
        """
        Apply a CRS transform to the x,y columns of a points array (z untouched).

        Parameters
        ----------
        points : ndarray of shape (N, ndim)
        transformer : pyproj Transformer

        Returns
        -------
        ndarray of shape (N, ndim) with transformed x,y.
        """
        pts = points.copy()
        tx, ty = transformer.transform(pts[:, 0], pts[:, 1])
        pts[:, 0] = np.asarray(tx)
        pts[:, 1] = np.asarray(ty)
        return pts

    def to_grid(self, source_points, grid_res, to_crs=None, bbox=None,
                method='linear', extrapolation='constant', fill_val=np.nan,
                min_val=None):
        """
        Precompute a points-to-grid interpolation function.

        Parameters
        ----------
        source_points : ndarray of shape (N, 2) or (N, 3)
            Source point locations in from_crs coordinates.
            Column order: (x, y) or (x, y, z).
        grid_res : float
            Output grid resolution in to_crs units.
        to_crs : int, str, or None
            Target CRS for the output grid.
        bbox : tuple of float or None
            (x_min, x_max, y_min, y_max) in to_crs coordinates.
            For 3D: (x_min, x_max, y_min, y_max, z_min, z_max).
        method : str
            'nearest', 'linear', or 'cubic'.
        extrapolation : str
            'constant' or 'nearest'.
        fill_val : float
            Fill value for 'constant' extrapolation.
        min_val : float or None
            Floor value.

        Returns
        -------
        callable
            func(values_1d) -> grid_nd
            values_1d: shape (N,) of data values at source_points.
            Output: (ny, nx) for 2D or (nz, ny, nx) for 3D.
        """
        ndim = source_points.shape[1]
        res = float(grid_res)

        # Transform source points to output CRS (only x,y)
        fwd, inv, _ = self._make_transformer(to_crs)
        if fwd is not None:
            pts = self._transform_points_xy(source_points, fwd)
        else:
            pts = source_points.copy()

        # Compute output bounds
        if bbox is not None:
            out_x_min, out_x_max = bbox[0], bbox[1]
            out_y_min, out_y_max = bbox[2], bbox[3]
        else:
            out_x_min, out_x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
            out_y_min, out_y_max = float(pts[:, 1].min()), float(pts[:, 1].max())

        new_x = np.arange(out_x_min, out_x_max + res * 0.5, res)
        new_y = np.arange(out_y_min, out_y_max + res * 0.5, res)

        if ndim == 3:
            if bbox is not None and len(bbox) >= 6:
                out_z_min, out_z_max = bbox[4], bbox[5]
            else:
                out_z_min, out_z_max = float(pts[:, 2].min()), float(pts[:, 2].max())
            new_z = np.arange(out_z_min, out_z_max + res * 0.5, res)

        # Build output query points and shape
        if ndim == 2:
            out_yy, out_xx = np.meshgrid(new_y, new_x, indexing='ij')
            output_query_points = np.column_stack([out_xx.ravel(), out_yy.ravel()])
            output_shape = (len(new_y), len(new_x))
        else:
            out_zz, out_yy, out_xx = np.meshgrid(new_z, new_y, new_x, indexing='ij')
            output_query_points = np.column_stack([
                out_xx.ravel(), out_yy.ravel(), out_zz.ravel()
            ])
            output_shape = (len(new_z), len(new_y), len(new_x))

        # Precompute Delaunay triangulation
        tri = Delaunay(pts)

        def _interpolate(values):
            if method == 'linear':
                interp = LinearNDInterpolator(tri, values, fill_value=fill_val)
            elif method == 'cubic':
                interp = CloughTocher2DInterpolator(tri, values, fill_value=fill_val)
            elif method == 'nearest':
                interp = NearestNDInterpolator(tri.points, values)
            else:
                raise ValueError(f"method must be 'nearest', 'linear', or 'cubic', got '{method}'")

            result = interp(output_query_points).reshape(output_shape)

            # Extrapolation: fill NaN outside convex hull with nearest neighbor
            if extrapolation == 'nearest' and method != 'nearest':
                nan_mask = np.isnan(result)
                if nan_mask.any():
                    nn = NearestNDInterpolator(tri.points, values)
                    result[nan_mask] = nn(output_query_points[nan_mask.ravel()])

            if min_val is not None:
                result = np.where(result < min_val, min_val, result)
            return result

        return _interpolate

    def to_points(self, source_points, target_points, to_crs=None,
                  method='linear', min_val=None):
        """
        Precompute a points-to-points interpolation function.

        Parameters
        ----------
        source_points : ndarray of shape (N, 2) or (N, 3)
            Source point locations in from_crs. Column order: (x, y) or (x, y, z).
        target_points : ndarray of shape (M, 2) or (M, 3)
            Target point locations in to_crs. Column order: (x, y) or (x, y, z).
        to_crs : int, str, or None
            CRS of target_points.
        method : str
            'nearest', 'linear', or 'cubic'.
        min_val : float or None
            Floor value.

        Returns
        -------
        callable
            func(values_1d) -> target_values of shape (M,)
        """
        # Transform source points to the target CRS so both are in the same space
        fwd, inv, _ = self._make_transformer(to_crs)
        if fwd is not None:
            src_pts = self._transform_points_xy(source_points, fwd)
        else:
            src_pts = source_points.copy()

        tgt_pts = target_points.copy()

        # Precompute Delaunay triangulation on transformed source points
        tri = Delaunay(src_pts)

        def _interpolate(values):
            if method == 'linear':
                interp = LinearNDInterpolator(tri, values, fill_value=np.nan)
            elif method == 'cubic':
                interp = CloughTocher2DInterpolator(tri, values, fill_value=np.nan)
            elif method == 'nearest':
                interp = NearestNDInterpolator(tri.points, values)
            else:
                raise ValueError(f"method must be 'nearest', 'linear', or 'cubic', got '{method}'")

            result = interp(tgt_pts)

            if min_val is not None:
                result = np.where(result < min_val, min_val, result)
            return result

        return _interpolate
