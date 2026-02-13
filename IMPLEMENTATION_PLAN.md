# Implementation Plan: Generalize geointerp to numpy-only with 2D/3D support

## Context

The current geointerp code (`core.py`, `interp2d.py`) is tightly coupled to xarray Datasets and pandas DataFrames, handles only 2D spatial interpolation, and recomputes coordinate transforms on every call. The goal is to:

- Remove xarray/pandas dependencies — numpy ndarrays only
- Support 2D and 3D spatial dimensions (x, y, optional z)
- Two classes: `GridInterpolator` (regular grid input) and `PointInterpolator` (scattered points)
- Classes take only `from_crs`; methods take config params and return a **callable** that accepts a single time-step array
- Precompute everything possible for speed
- Support both regular z and variable z per (x,y) for 3D grids
- Replace existing code entirely

---

## Step 1: Rewrite `geointerp/util.py`

Foundation module — everything depends on it. Clean up and generalize for N dimensions.

**Keep**: `find_nearest` (unchanged)

**Replace** `grid_xy_to_map_coords` with:

```python
def grid_coords_to_index_params(coord_arrays):
    """
    Given tuple of 1D coordinate arrays (one per grid dimension), return
    the origin and spacing for each dimension.

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
```

Simply computes `origin = arr.min()` and `spacing = median(diff(arr))` for each array.

**Replace** `point_xy_to_map_coords` with:

```python
def coords_to_indices(points, origins, spacings):
    """
    Convert (M, ndim) coordinate array to (ndim, M) fractional array indices.

    Parameters
    ----------
    points : ndarray of shape (M, ndim)
        Coordinates to convert.
    origins : tuple of float
        Origin for each dimension.
    spacings : tuple of float
        Spacing for each dimension.

    Returns
    -------
    indices : ndarray of shape (ndim, M)
        Fractional array indices suitable for scipy map_coordinates.
    """
```

For each dimension `d`: `indices[d] = (points[:, d] - origins[d]) / spacings[d]`

**Remove**: `map_coords_to_xy` (unused externally), all commented-out code (lines 6-24, 123-180).

---

## Step 2: Create `geointerp/grid.py` — GridInterpolator

Uses `scipy.ndimage.map_coordinates` for fast spline interpolation on regular grids.

### 2a. Constructor and CRS helper

```python
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
            Forward transform (from_crs → to_crs).
        inv : Transformer or None
            Inverse transform (to_crs → from_crs).
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
```

Uses `Transformer.from_crs(..., always_xy=True)` instead of the deprecated `Proj` class used in the current code. The `always_xy=True` flag ensures consistent x,y ordering regardless of CRS, fixing axis-order bugs present in the original code.

### 2b. `to_grid` method → returns `func(data_nd) → grid_nd`

```python
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
        Tuple gives per-dimension resolution matching source_coords order.
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
```

**Precomputation steps** (all done once inside `to_grid`, captured in returned closure):

1. **Validate and unpack source_coords.** Determine `ndim = len(source_coords)` (2 or 3). Map user-facing `(x, y)` or `(x, y, z)` to array dimension order `(y, x)` or `(z, y, x)`:

    ```python
    if ndim == 2:
        x_arr, y_arr = source_coords
        dim_arrays = (y_arr, x_arr)
    elif ndim == 3:
        x_arr, y_arr, z_arr = source_coords
        dim_arrays = (z_arr, y_arr, x_arr)
    ```

2. **Compute grid index params** from dim_arrays:

    ```python
    origins, spacings = grid_coords_to_index_params(dim_arrays)
    ```

3. **Build CRS transformer** (if to_crs specified):

    ```python
    fwd, inv, out_is_geo = self._make_transformer(to_crs)
    ```

4. **Determine output bounds.** If `bbox` is provided, use it directly. Otherwise, if CRS transform exists, transform source grid corners to output CRS and take min/max. If no CRS transform, use source coordinate min/max. For 3D, z bounds come directly from `z_arr` (CRS does not apply to z):

    ```python
    if bbox is not None:
        out_x_min, out_x_max, out_y_min, out_y_max = bbox[:4]
    elif fwd is not None:
        xy_mesh = np.array(np.meshgrid(x_arr, y_arr)).reshape(2, -1)
        out_xy = np.array(fwd.transform(xy_mesh[0], xy_mesh[1]))
        out_x_min, out_x_max = out_xy[0].min(), out_xy[0].max()
        out_y_min, out_y_max = out_xy[1].min(), out_xy[1].max()
    else:
        out_x_min, out_x_max = x_arr.min(), x_arr.max()
        out_y_min, out_y_max = y_arr.min(), y_arr.max()
    ```

5. **Build output coordinate arrays** using `np.arange` with `grid_res`.

6. **Build output meshgrid and compute fractional array indices.** Transform output points back to source CRS via inverse transformer, then convert to array indices. This is the key precomputation — these indices are reused on every call:

    ```python
    # 2D example
    out_yy, out_xx = np.meshgrid(new_y, new_x, indexing='ij')

    if inv is not None:
        src_x, src_y = inv.transform(out_xx.ravel(), out_yy.ravel())
        src_x = src_x.reshape(out_xx.shape)
        src_y = src_y.reshape(out_yy.shape)
    else:
        src_x, src_y = out_xx, out_yy

    idx_y = (src_y - origins[0]) / spacings[0]
    idx_x = (src_x - origins[1]) / spacings[1]
    precomputed_coords = np.array([idx_y.ravel(), idx_x.ravel()])
    output_shape = (len(new_y), len(new_x))
    ```

    For 3D, same pattern with 3 dimensions. CRS transform applies only to x/y; z indices are computed directly from z coordinates.

7. **Return closure:**

    ```python
    def _interpolate(data):
        out = np.empty(precomputed_coords.shape[1], dtype=data.dtype)
        map_coordinates(data, precomputed_coords, out,
                        order=order, mode=extrapolation, cval=fill_val, prefilter=True)
        result = out.reshape(output_shape)
        if min_val is not None:
            result = np.where(result < min_val, min_val, result)
        return result

    return _interpolate
    ```

    The output array is pre-allocated in the closure for each call. `map_coordinates` writes to it in-place (matching the pattern in the current `interp2d.py` line 142).

### 2c. `to_points` method → returns `func(data_nd) → values shape (M,)`

```python
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
```

**Precomputation:**
1. Compute grid index params from source_coords.
2. If CRS transform needed: transform `target_points` x,y columns from to_crs to from_crs (z passes through unchanged).
3. Convert transformed target points to fractional array indices → `precomputed_coords` of shape `(ndim, M)`.
4. Return closure that calls `map_coordinates(data, precomputed_coords, output, order=order, cval=np.nan)`.

Refactored from `grid_to_points` in `interp2d.py` lines 156-245. The key difference: coordinate transforms and index computation happen once at setup, not per time step.

### 2d. `interp_na` method → returns `func(data_nd) → data_nd`

```python
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
```

Cannot fully precompute because NaN locations vary per time step.

**Precomputation** (limited):
Build the full coordinate meshgrid from source_coords once:

```python
grids = np.meshgrid(*reversed(source_coords), indexing='ij')  # (y,x) or (z,y,x)
all_points = np.column_stack([g.ravel() for g in grids])
```

**Returned callable:**

```python
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
```

Refactored from `grid_interp_na` in `interp2d.py` lines 440-482.

### 2e. `regrid_levels` method → returns `func(data_3d, source_levels_3d) → data_3d`

New functionality for variable z (terrain-following coordinates). The source levels vary per (y,x) location and may change per time step, so `source_levels` is passed to the returned function.

```python
def regrid_levels(self, target_levels, axis=0, method='linear'):
    """
    Return a callable that regrids data from variable source levels to fixed target levels.

    For terrain-following coordinates where source levels vary per (y,x) location.

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
        Output: target_levels replaces the level axis.
    """
```

**Precomputation:**
Store `target_levels` (sorted 1D array), `axis`, and `n_target = len(target_levels)`.

**Algorithm in the returned callable** (vectorized, based on `metpy_custom2` pattern from `vertical_interp_test.py` lines 228-243):

```python
def _regrid(data, source_levels):
    # Move level axis to position 0 for consistent indexing
    data_moved = np.moveaxis(data, axis, 0)          # (n_src, ...)
    levels_moved = np.moveaxis(source_levels, axis, 0)  # (n_src, ...)

    spatial_shape = data_moved.shape[1:]
    n_src = data_moved.shape[0]
    n_tgt = len(target_levels)

    # Flatten spatial dims for vectorized searchsorted
    data_flat = data_moved.reshape(n_src, -1)         # (n_src, N_spatial)
    levels_flat = levels_moved.reshape(n_src, -1)      # (n_src, N_spatial)

    out = np.empty((n_tgt, data_flat.shape[1]), dtype=data.dtype)

    for k in range(n_tgt):
        tgt = target_levels[k]
        # For each spatial point, find where tgt falls in its level column
        # levels_flat[:, j] is the level profile at spatial point j
        # Use broadcasting: compare tgt against all levels
        above_mask = levels_flat >= tgt  # (n_src, N_spatial)
        # Index of first level >= tgt at each spatial point
        above_idx = np.argmax(above_mask, axis=0)  # (N_spatial,)
        # Handle case where tgt is above all levels
        no_above = ~above_mask.any(axis=0)
        above_idx[no_above] = n_src - 1
        below_idx = np.clip(above_idx - 1, 0, n_src - 1)

        cols = np.arange(data_flat.shape[1])
        lev_below = levels_flat[below_idx, cols]
        lev_above = levels_flat[above_idx, cols]
        val_below = data_flat[below_idx, cols]
        val_above = data_flat[above_idx, cols]

        denom = lev_above - lev_below
        safe_denom = np.where(denom == 0, 1.0, denom)
        weight = np.where(denom == 0, 0.0, (tgt - lev_below) / safe_denom)

        out[k] = val_below + weight * (val_above - val_below)

    result = out.reshape((n_tgt,) + spatial_shape)
    return np.moveaxis(result, 0, axis)

return _regrid
```

This avoids the per-column Python loop that `np.interp` would require, instead vectorizing across all spatial points for each target level.

---

## Step 3: Create `geointerp/points.py` — PointInterpolator

Uses `scipy.interpolate` machinery. Key optimization: precompute Delaunay triangulation on source points so it isn't recomputed every time step.

### 3a. Constructor

```python
import numpy as np
from pyproj import CRS, Transformer
from scipy.spatial import Delaunay
from scipy.interpolate import (
    LinearNDInterpolator, NearestNDInterpolator, CloughTocher2DInterpolator
)

from geointerp.util import find_nearest


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
```

Same `_make_transformer` helper as GridInterpolator (6 lines, duplicated rather than adding a mixin/base class).

### 3b. `to_grid` method → returns `func(values_1d) → grid_nd`

```python
def to_grid(self, source_points, grid_res, to_crs=None, bbox=None,
            method='linear', extrapolation='constant', fill_val=np.nan, min_val=None):
    """
    Precompute a points-to-grid interpolation function.

    Parameters
    ----------
    source_points : ndarray of shape (N, 2) or (N, 3)
        Source point locations in from_crs coordinates. Column order: (x, y) or (x, y, z).
    grid_res : float
        Output grid resolution in to_crs units.
    to_crs : int, str, or None
        Target CRS for the output grid.
    bbox : tuple of float or None
        (x_min, x_max, y_min, y_max) in to_crs coordinates.
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
        values_1d: shape (N,) data values at source_points.
        Output: (ny, nx) for 2D or (nz, ny, nx) for 3D.
    """
```

**Precomputation:**

1. **Transform source points** to output CRS if needed. CRS applies only to x,y columns; z passes through:

    ```python
    fwd, inv, out_is_geo = self._make_transformer(to_crs)
    pts = source_points.copy()
    if fwd is not None:
        tx, ty = fwd.transform(pts[:, 0], pts[:, 1])
        pts[:, 0], pts[:, 1] = tx, ty
    ```

2. **Compute output bounds** from transformed source points (or use `bbox`).

3. **Build output grid** coordinate arrays and meshgrid:

    ```python
    new_x = np.arange(out_x_min, out_x_max, grid_res)
    new_y = np.arange(out_y_min, out_y_max, grid_res)
    # For 2D:
    out_yy, out_xx = np.meshgrid(new_y, new_x, indexing='ij')
    output_query_points = np.column_stack([out_xx.ravel(), out_yy.ravel()])
    output_shape = (len(new_y), len(new_x))
    ```

4. **Precompute Delaunay triangulation:**

    ```python
    tri = Delaunay(pts)  # O(N log N) — done once
    ```

    This is the key optimization. The current code (`interp2d.py` line 330) calls `griddata` per time step, which internally recomputes the Delaunay triangulation every time. For `N` source points and `T` time steps, that's `O(T * N log N)` just for triangulation. With precomputed Delaunay, each call is `O(M log N)` where `M` = output points.

    `LinearNDInterpolator` and `CloughTocher2DInterpolator` both accept a `Delaunay` object as their first argument instead of raw points ([scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html)).

**Returned callable:**

```python
def _interpolate(values):
    if method == 'linear':
        interp = LinearNDInterpolator(tri, values, fill_value=fill_val)
    elif method == 'cubic':
        interp = CloughTocher2DInterpolator(tri, values, fill_value=fill_val)
    elif method == 'nearest':
        interp = NearestNDInterpolator(tri.points, values)

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
```

The `extrapolation='nearest'` fallback pattern comes from the current code at `interp2d.py` lines 331-336.

### 3c. `to_points` method → returns `func(values_1d) → values shape (M,)`

```python
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
```

**Precomputation:**
1. Transform source_points to a common CRS (to_crs, matching target_points).
2. Precompute `tri = Delaunay(source_points_transformed)`.
3. Store target_points for evaluation.

**Returned callable:** Same pattern as `to_grid` but evaluates at `target_points` instead of a meshgrid.

---

## Step 4: Update `geointerp/__init__.py`

```python
"""geointerp -- geospatial interpolations"""

from geointerp.grid import GridInterpolator
from geointerp.points import PointInterpolator

__all__ = ['GridInterpolator', 'PointInterpolator']
```

---

## Step 5: Update `pyproject.toml` dependencies

**Current:**
```toml
dependencies = [
  'hdf5tools>=0.3.1',
  'scipy~=1.0',
  'pyproj~=3.0',
  'h5netcdf~=1.1',
]
```

**New:**
```toml
dependencies = [
  'numpy>=1.20',
  'scipy>=1.7',
  'pyproj>=3.0',
]
```

Remove `hdf5tools` and `h5netcdf` entirely. Raise scipy floor to 1.7 for stable `LinearNDInterpolator` with precomputed Delaunay support.

---

## Step 6: Delete old files

- `geointerp/core.py`
- `geointerp/interp2d.py`
- `geointerp/tests/vertical_interp_test.py` (experimental scratch code, not real tests)

---

## Step 7: Create tests

### 7a. `geointerp/tests/conftest.py` — shared fixtures

```python
import numpy as np
import pytest


@pytest.fixture
def grid_2d_data():
    """Simple 2D grid with known analytic function."""
    x = np.linspace(0, 100, 11)
    y = np.linspace(0, 80, 9)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    data = np.sin(xx / 50) * np.cos(yy / 40)
    return x, y, data


@pytest.fixture
def grid_3d_data():
    """Simple 3D grid with known analytic function."""
    x = np.linspace(0, 100, 11)
    y = np.linspace(0, 80, 9)
    z = np.linspace(0, 500, 6)
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
    data = np.sin(xx / 50) * np.cos(yy / 40) * (zz / 500)
    return x, y, z, data


@pytest.fixture
def scattered_2d_points():
    """Scattered 2D test points with known function values."""
    rng = np.random.default_rng(42)
    N = 200
    points = rng.uniform(0, 100, (N, 2))
    values = np.sin(points[:, 0] / 50) * np.cos(points[:, 1] / 40)
    return points, values
```

### 7b. `geointerp/tests/test_grid.py`

Test classes and cases:

- **`TestGridToGrid2D`**:
  - `test_identity_no_crs` — same grid → same data (within tolerance)
  - `test_upscale` — finer resolution → larger output, constant field stays constant
  - `test_crs_transform` — EPSG:2193 to EPSG:4326
  - `test_bbox_clipping` — output respects bbox
  - `test_min_val` — values below min_val are clipped
  - `test_extrapolation_constant` — outside source domain gets fill_val
  - `test_extrapolation_nearest` — outside gets nearest edge value

- **`TestGridToGrid3D`**:
  - `test_3d_identity` — 3D identity interpolation
  - `test_3d_upscale` — 3D with finer resolution

- **`TestGridToPoints`**:
  - `test_at_grid_nodes` — values at grid nodes should be exact
  - `test_between_nodes` — linear interpolation correctness
  - `test_crs_transform` — points in different CRS

- **`TestGridInterpNa`**:
  - `test_fill_interior_nans` — surrounded NaN gets filled
  - `test_no_nans` — no-NaN data returned unchanged

- **`TestRegridLevels`**:
  - `test_uniform_levels` — uniform source/target levels
  - `test_terrain_following` — variable source levels per (y,x)
  - `test_extrapolation_clamp` — target outside source range clamps to boundary

### 7c. `geointerp/tests/test_points.py`

- **`TestPointsToGrid`**:
  - `test_known_function` — interpolate z = x + y, check interior grid points
  - `test_extrapolation_nearest` — fills outside convex hull
  - `test_crs_transform` — source and output in different CRS

- **`TestPointsToPoints`**:
  - `test_at_source_points` — values at source locations should be exact
  - `test_known_function` — known analytic function between point sets

- **`TestPrecomputationSpeed`**:
  - `test_repeated_calls_faster` — verify precomputed callable is faster than calling griddata per step

---

## Step 8: Update `CLAUDE.md`

Update architecture section to reflect the new two-class design, numpy-only API, and callable factory pattern.

---

## Design Decisions and Rationale

### User-facing coordinate order: (x, y) not (y, x)
The existing code mixes (y, x) and (x, y) conventions inconsistently. The new API consistently accepts `(x, y)` or `(x, y, z)` matching geographic convention (longitude=x, latitude=y). Internally, the code maps to array dimension order `(y, x)` or `(z, y, x)`.

### Closures for returned callables
A closure is simpler and more Pythonic than a class. It captures exactly the precomputed state needed. The captured state is effectively immutable.

### `always_xy=True` for pyproj
Eliminates axis-order ambiguity that caused real bugs in the original code (inconsistent y/x swapping between `points_to_points` and `grid_to_points`).

### No `digits` parameter
The existing `digits` parameter for rounding output is not an interpolation concern — the caller can round the result. Internal coordinate precision for detecting geographic vs projected CRS is handled automatically via `CRS.is_geographic`.

### Delaunay precomputation
In the current code, `griddata` recomputes Delaunay triangulation every call. By precomputing `tri = Delaunay(source_points)` and passing it to `LinearNDInterpolator(tri, values)`, we skip O(N log N) triangulation on each time step.

### `_make_transformer` duplicated in both classes
6 lines of code. A shared base class or mixin would add complexity disproportionate to the code saved.

---

## Implementation Order

1. `geointerp/util.py` — rewrite coordinate functions (no dependencies)
2. `geointerp/grid.py` — GridInterpolator (depends on util)
   - Constructor + `_make_transformer`
   - `to_grid` (2D first, then add 3D)
   - `to_points`
   - `interp_na`
   - `regrid_levels`
3. `geointerp/points.py` — PointInterpolator (depends on util)
   - Constructor + `_make_transformer`
   - `to_grid`
   - `to_points`
4. `geointerp/__init__.py` — update exports
5. `pyproject.toml` — update dependencies
6. Delete `core.py`, `interp2d.py`, `vertical_interp_test.py`
7. `geointerp/tests/conftest.py` — shared fixtures
8. `geointerp/tests/test_grid.py` — GridInterpolator tests
9. `geointerp/tests/test_points.py` — PointInterpolator tests
10. `CLAUDE.md` — update documentation

---

## Verification

```bash
# Run all tests
uv run pytest

# Run specific test class
uv run pytest geointerp/tests/test_grid.py::TestGridToGrid2D -v

# Verify no xarray/pandas imports remain
grep -r "import xarray\|import pandas\|from pandas\|from xarray" geointerp/

# Verify package imports work
uv run python -c "from geointerp import GridInterpolator, PointInterpolator; print('OK')"
```
