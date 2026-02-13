"""
GridInterpolator for regular grid data.
"""
import numpy as np
from pyproj import CRS, Transformer
from scipy.ndimage import map_coordinates
from scipy.interpolate import griddata

from geointerp.util import grid_coords_to_index_params, coords_to_indices


class GridInterpolator:
    """
    Interpolator for data on regular grids.

    Uses scipy.ndimage.map_coordinates for fast spline-based interpolation.
    Methods return precomputed callables for repeated application to time steps.

    Parameters
    ----------
    from_crs : int, str, or None
        The CRS of the input grid coordinates. Can be an EPSG integer,
        a proj4 string, or None (no projection).
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

    def _unpack_source_coords(self, source_coords):
        """
        Unpack user-facing (x, y) or (x, y, z) into array dimension order.

        Returns
        -------
        x_arr : ndarray
        y_arr : ndarray
        z_arr : ndarray or None
        dim_arrays : tuple
            Coordinate arrays in array dimension order: (y, x) or (z, y, x).
        """
        ndim = len(source_coords)
        if ndim == 2:
            x_arr, y_arr = source_coords
            return x_arr, y_arr, None, (y_arr, x_arr)
        elif ndim == 3:
            x_arr, y_arr, z_arr = source_coords
            return x_arr, y_arr, z_arr, (z_arr, y_arr, x_arr)
        else:
            raise ValueError('source_coords must have 2 or 3 elements')

    def _compute_output_bounds_xy(self, x_arr, y_arr, bbox, fwd):
        """
        Determine x/y output bounds from bbox, CRS transform, or source coords.

        Returns
        -------
        out_x_min, out_x_max, out_y_min, out_y_max : float
        """
        if bbox is not None:
            return bbox[0], bbox[1], bbox[2], bbox[3]
        if fwd is not None:
            xy_mesh = np.array(np.meshgrid(x_arr, y_arr)).reshape(2, -1)
            out_xy = np.array(fwd.transform(xy_mesh[0], xy_mesh[1]))
            return out_xy[0].min(), out_xy[0].max(), out_xy[1].min(), out_xy[1].max()
        return float(x_arr.min()), float(x_arr.max()), float(y_arr.min()), float(y_arr.max())

    def to_grid(self, source_coords, grid_res, to_crs=None, bbox=None,
                order=3, extrapolation='constant', fill_val=np.nan, min_val=None):
        """
        Precompute a grid-to-grid interpolation function.

        Parameters
        ----------
        source_coords : tuple of 1D ndarray
            Coordinate arrays defining the source grid axes.
            2D: (x_arr, y_arr) where data shape is (len(y), len(x)).
            3D: (x_arr, y_arr, z_arr) where data shape is (len(z), len(y), len(x)).
        grid_res : float or tuple of float
            Output grid resolution. Single value applies to all spatial dimensions.
            Tuple gives per-dimension resolution in (x, y) or (x, y, z) order.
        to_crs : int, str, or None
            Target CRS for the output grid.
        bbox : tuple of float or None
            Bounding box in to_crs coordinates.
            2D: (x_min, x_max, y_min, y_max)
            3D: (x_min, x_max, y_min, y_max, z_min, z_max)
        order : int
            Spline interpolation order (0-5). Default 3.
        extrapolation : str
            Mode for map_coordinates: 'constant', 'nearest', 'reflect', 'mirror', 'wrap'.
        fill_val : float
            Fill value for 'constant' extrapolation mode.
        min_val : float or None
            Floor value for results.

        Returns
        -------
        callable
            func(data_nd) -> grid_nd
            Input: (ny, nx) for 2D or (nz, ny, nx) for 3D.
            Output: (ny_out, nx_out) for 2D or (nz_out, ny_out, nx_out) for 3D.
        """
        x_arr, y_arr, z_arr, dim_arrays = self._unpack_source_coords(source_coords)
        ndim = len(dim_arrays)
        origins, spacings = grid_coords_to_index_params(dim_arrays)

        fwd, inv, _ = self._make_transformer(to_crs)

        # Parse grid_res per dimension (user order: x, y[, z])
        if isinstance(grid_res, (int, float)):
            res_x = res_y = float(grid_res)
            res_z = float(grid_res) if ndim == 3 else None
        else:
            res_x = float(grid_res[0])
            res_y = float(grid_res[1])
            res_z = float(grid_res[2]) if ndim == 3 and len(grid_res) > 2 else None

        # Output bounds for x/y
        out_x_min, out_x_max, out_y_min, out_y_max = self._compute_output_bounds_xy(
            x_arr, y_arr, bbox, fwd
        )

        # Add half-step to upper bound so np.arange includes the endpoint
        new_x = np.arange(out_x_min, out_x_max + res_x * 0.5, res_x)
        new_y = np.arange(out_y_min, out_y_max + res_y * 0.5, res_y)

        if ndim == 3:
            if bbox is not None and len(bbox) >= 6:
                out_z_min, out_z_max = bbox[4], bbox[5]
            else:
                out_z_min, out_z_max = float(z_arr.min()), float(z_arr.max())
            new_z = np.arange(out_z_min, out_z_max + res_z * 0.5, res_z)

        # Build output meshgrid and compute fractional array indices
        if ndim == 2:
            out_yy, out_xx = np.meshgrid(new_y, new_x, indexing='ij')

            if inv is not None:
                src_x, src_y = inv.transform(out_xx.ravel(), out_yy.ravel())
                src_x = np.asarray(src_x).reshape(out_xx.shape)
                src_y = np.asarray(src_y).reshape(out_yy.shape)
            else:
                src_x = out_xx
                src_y = out_yy

            # dim_arrays order is (y, x), so origins/spacings[0]=y, [1]=x
            idx_y = (src_y - origins[0]) / spacings[0]
            idx_x = (src_x - origins[1]) / spacings[1]
            precomputed_coords = np.array([idx_y.ravel(), idx_x.ravel()])
            output_shape = (len(new_y), len(new_x))
        else:
            out_zz, out_yy, out_xx = np.meshgrid(new_z, new_y, new_x, indexing='ij')

            if inv is not None:
                src_x, src_y = inv.transform(out_xx.ravel(), out_yy.ravel())
                src_x = np.asarray(src_x).reshape(out_xx.shape)
                src_y = np.asarray(src_y).reshape(out_yy.shape)
            else:
                src_x = out_xx
                src_y = out_yy
            src_z = out_zz  # CRS does not apply to z

            # dim_arrays order is (z, y, x), so origins/spacings[0]=z, [1]=y, [2]=x
            idx_z = (src_z - origins[0]) / spacings[0]
            idx_y = (src_y - origins[1]) / spacings[1]
            idx_x = (src_x - origins[2]) / spacings[2]
            precomputed_coords = np.array([idx_z.ravel(), idx_y.ravel(), idx_x.ravel()])
            output_shape = (len(new_z), len(new_y), len(new_x))

        def _interpolate(data):
            out = np.empty(precomputed_coords.shape[1], dtype=data.dtype)
            map_coordinates(data, precomputed_coords, out,
                            order=order, mode=extrapolation, cval=fill_val, prefilter=True)
            result = out.reshape(output_shape)
            if min_val is not None:
                result = np.where(result < min_val, min_val, result)
            return result

        return _interpolate

    def to_points(self, source_coords, target_points, to_crs=None,
                  order=3, min_val=None):
        """
        Precompute a grid-to-points interpolation function.

        Parameters
        ----------
        source_coords : tuple of 1D ndarray
            Source grid axes: (x_arr, y_arr) or (x_arr, y_arr, z_arr).
        target_points : ndarray of shape (M, 2) or (M, 3)
            Target point locations in to_crs coordinates. Column order: (x, y) or (x, y, z).
        to_crs : int, str, or None
            CRS of target_points if different from from_crs.
        order : int
            Spline interpolation order (0-5).
        min_val : float or None
            Floor value for results.

        Returns
        -------
        callable
            func(data_nd) -> values of shape (M,)
        """
        x_arr, y_arr, z_arr, dim_arrays = self._unpack_source_coords(source_coords)
        ndim = len(dim_arrays)
        origins, spacings = grid_coords_to_index_params(dim_arrays)

        target = target_points.copy()

        # Transform target x/y from to_crs back to from_crs (source grid space)
        if to_crs is not None and self._from_crs is not None:
            to_crs_obj = CRS.from_user_input(to_crs)
            inv = Transformer.from_crs(to_crs_obj, self._from_crs, always_xy=True)
            src_x, src_y = inv.transform(target[:, 0], target[:, 1])
            target[:, 0] = np.asarray(src_x)
            target[:, 1] = np.asarray(src_y)
            # z (column 2) stays unchanged if present

        # Reorder columns from user (x, y[, z]) to array dim order (y, x) or (z, y, x)
        if ndim == 2:
            dim_order_points = target[:, [1, 0]]  # (y, x)
        else:
            dim_order_points = target[:, [2, 1, 0]]  # (z, y, x)

        precomputed_coords = coords_to_indices(dim_order_points, origins, spacings)
        n_points = target.shape[0]

        def _interpolate(data):
            out = np.empty(n_points, dtype=data.dtype)
            map_coordinates(data, precomputed_coords, out,
                            order=order, cval=np.nan, prefilter=True)
            if min_val is not None:
                out = np.where(out < min_val, min_val, out)
            return out

        return _interpolate

    def interp_na(self, source_coords, method='linear', min_val=None):
        """
        Return a callable that fills NaN values in a grid slice.

        Parameters
        ----------
        source_coords : tuple of 1D ndarray
            Source grid axes: (x_arr, y_arr) or (x_arr, y_arr, z_arr).
        method : str
            Interpolation method: 'nearest', 'linear', 'cubic'.
        min_val : float or None
            Floor value.

        Returns
        -------
        callable
            func(data_nd) -> data_nd with NaN filled (new array, same shape).
        """
        # Build full coordinate meshgrid in array dimension order (y, x) or (z, y, x)
        _, _, _, dim_arrays = self._unpack_source_coords(source_coords)
        grids = np.meshgrid(*dim_arrays, indexing='ij')
        all_points = np.column_stack([g.ravel() for g in grids])

        def _fill_na(data):
            flat = data.ravel().copy()
            isnan = np.isnan(flat)
            if not isnan.any():
                return data.copy()
            flat[isnan] = griddata(
                all_points[~isnan], flat[~isnan], all_points[isnan],
                method=method, fill_value=np.nan
            )
            result = flat.reshape(data.shape)
            if min_val is not None:
                result = np.where(result < min_val, min_val, result)
            return result

        return _fill_na

    def regrid_levels(self, target_levels, axis=0, method='linear'):
        """
        Return a callable that regrids data from variable source levels to fixed target levels.

        For terrain-following coordinates where source levels vary per (y,x) location.
        Source levels must be passed to the returned function since they may change per
        time step.

        Parameters
        ----------
        target_levels : 1D ndarray
            Target level values (must be monotonically increasing).
        axis : int
            The axis in the input data that corresponds to the vertical/level dimension.
        method : str
            'linear' supported.

        Returns
        -------
        callable
            func(data_3d, source_levels_3d) -> data_3d
            source_levels_3d: same shape as data_3d, actual level values at each point.
            Source levels must be monotonically increasing along the given axis.
            Output has target_levels replacing the level axis.
        """
        target_levels = np.asarray(target_levels, dtype=float)
        n_tgt = len(target_levels)

        def _regrid(data, source_levels):
            # Move level axis to position 0 for consistent indexing
            data_moved = np.moveaxis(data, axis, 0)
            levels_moved = np.moveaxis(source_levels, axis, 0)

            spatial_shape = data_moved.shape[1:]
            n_src = data_moved.shape[0]

            # Flatten spatial dims for vectorized operations
            data_flat = data_moved.reshape(n_src, -1)
            levels_flat = levels_moved.reshape(n_src, -1)
            n_spatial = data_flat.shape[1]

            out = np.empty((n_tgt, n_spatial), dtype=data.dtype)
            cols = np.arange(n_spatial)

            for k in range(n_tgt):
                tgt = target_levels[k]

                # Find first source level >= target at each spatial point
                above_mask = levels_flat >= tgt
                above_idx = np.argmax(above_mask, axis=0)

                # Handle target above all source levels (no level >= tgt)
                no_above = ~above_mask.any(axis=0)
                above_idx[no_above] = n_src - 1

                below_idx = np.clip(above_idx - 1, 0, n_src - 1)

                lev_below = levels_flat[below_idx, cols]
                lev_above = levels_flat[above_idx, cols]
                val_below = data_flat[below_idx, cols]
                val_above = data_flat[above_idx, cols]

                denom = lev_above - lev_below
                safe_denom = np.where(denom == 0, 1.0, denom)
                weight = np.clip(
                    np.where(denom == 0, 0.0, (tgt - lev_below) / safe_denom),
                    0.0, 1.0
                )

                out[k] = val_below + weight * (val_above - val_below)

            result = out.reshape((n_tgt,) + spatial_shape)
            return np.moveaxis(result, 0, axis)

        return _regrid
