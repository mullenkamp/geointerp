# geointerp

<p align="center">
    <em>High-performance geospatial interpolations for 2D/3D time-series data.</em>
</p>

[![build](https://github.com/mullenkamp/geointerp/workflows/Build/badge.svg)](https://github.com/mullenkamp/geointerp/actions)
[![codecov](https://codecov.io/gh/mullenkamp/geointerp/branch/master/graph/badge.svg)](https://codecov.io/gh/mullenkamp/geointerp)
[![PyPI version](https://badge.fury.io/py/geointerp.svg)](https://badge.fury.io/py/geointerp)

---

**Source Code**: [https://github.com/mullenkamp/geointerp](https://github.com/mullenkamp/geointerp)

---

## Project Description

`geointerp` is a Python package designed for efficient 2D and 3D geospatial interpolation. It is specifically optimized for datasets with an additional time dimension (e.g., climate models, oceanographic data, or sensor networks).

The library follows a **callable factory pattern**: expensive setup operations—such as coordinate transformations, meshgrid generation, and Delaunay triangulation—are performed once. The result is a highly optimized interpolation function that can be rapidly applied to hundreds or thousands of time steps.

### Key Features

- **Regular Grid Interpolation**: Fast spline-based interpolation using `scipy.ndimage.map_coordinates`.
- **Scattered Point Interpolation**: Robust interpolation for irregular data using `scipy.interpolate.LinearNDInterpolator` with precomputed Delaunay triangulation.
- **Geospatial Aware**: Native support for CRS transformations via `pyproj`.
- **3D Support**: Handles vertical coordinates (z) alongside spatial (x, y).
- **Time-Series Optimized**: Drastically reduces overhead when interpolating multiple time steps over the same geometry.
- **Clean API**: Uses standard NumPy ndarrays for all inputs and outputs.

---

## Installation

### Using pip
```bash
pip install geointerp
```

### Using uv (Recommended)
```bash
uv add geointerp
```

---

## Quick Start

### 1. Regular Grid Interpolation

Interpolate from one grid to another (e.g., regridding a global model to a local projection).

```python
import numpy as np
from geointerp import GridInterpolator

# Define source grid coordinates (x, y)
x_src = np.linspace(170, 175, 51)
y_src = np.linspace(-45, -40, 41)
source_coords = (x_src, y_src)

# Create the interpolator (source is in WGS84 / EPSG:4326)
interp = GridInterpolator(from_crs="EPSG:4326")

# Precompute the interpolation function to a New Zealand Transverse Mercator grid
# This step handles the expensive CRS transforms and index mapping
regrid_func = interp.to_grid(
    source_coords, 
    grid_res=1000,          # 1km resolution
    to_crs="EPSG:2193"      # Target CRS
)

# Apply to time-series data
for time_step in model_data:
    # time_step is a (ny, nx) numpy array
    regridded_data = regrid_func(time_step)
```

### 2. Scattered Point Interpolation

Interpolate scattered sensor data onto a regular grid.

```python
from geointerp import PointInterpolator

# Scattered sensor locations (x, y) and values
points = np.array([[172.5, -43.5], [172.7, -43.6], [172.4, -43.4]])
values = np.array([15.2, 14.8, 16.1])

# Create interpolator
p_interp = PointInterpolator(from_crs="EPSG:4326")

# Precompute grid-to-points mapping (Delaunay triangulation happens here)
to_grid_func = p_interp.to_grid(
    points, 
    grid_res=0.01, 
    method='linear'
)

# Interpolate values
grid_result = to_grid_func(values)
```

### 3. Vertical Level Regridding (Terrain-Following to Fixed Levels)

This is particularly useful for ocean or atmospheric models where vertical layers follow the terrain (e.g., sigma or hybrid coordinates) and need to be interpolated to fixed depths or pressure levels.

```python
import numpy as np
from geointerp import GridInterpolator

# Suppose we have a 3D volume of ocean temperature: (depth_layers, lat, lon)
# Shape: (10 layers, 100 lat, 100 lon)
data_3d = np.random.random((10, 100, 100))

# In terrain-following coordinates, the "depth" of a layer varies by location.
# source_levels_3d must be the same shape as data_3d, containing the actual 
# vertical value (e.g., depth in meters) for every single grid cell.
source_levels_3d = np.zeros((10, 100, 100))
for z in range(10):
    # Depths increase with index, but vary spatially based on seabed/terrain
    source_levels_3d[z, :, :] = z * 10 + np.random.random((100, 100)) * 5

# Define the fixed depths we want to interpolate to
target_depths = np.array([0, 5, 10, 20, 50])

# Initialize interpolator
interp = GridInterpolator()

# Get the specialized regridding function (axis=0 is the vertical dimension)
regrid_z_func = interp.regrid_levels(target_depths, axis=0)

# Execute the interpolation
# This performs vectorized linear interpolation along the vertical axis for every (x, y) column
fixed_depth_data = regrid_z_func(data_3d, source_levels_3d)

# fixed_depth_data now has shape (5, 100, 100) corresponding to our 5 target depths
```

---

## Supported Methods & Extrapolation

### GridInterpolator
For regular grids, the library uses spline interpolation via [scipy.ndimage.map_coordinates](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html).

| Parameter | Options | Description |
| :--- | :--- | :--- |
| **`order`** | `0` to `5` | Spline order. `0`: nearest, `1`: linear, `3`: cubic (default). |
| **`extrapolation`** | `'constant'`, `'nearest'`, `'reflect'`, `'wrap'` | How to handle values outside the source grid. |
| **`fill_val`** | `float` or `np.nan` | Value used for `'constant'` extrapolation. |

### PointInterpolator
For scattered points, the library uses Delaunay-based interpolation classes from [scipy.interpolate](https://docs.scipy.org/doc/scipy/reference/interpolate.html):
- **linear**: [LinearNDInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html)
- **nearest**: [NearestNDInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NearestNDInterpolator.html)
- **cubic**: [CloughTocher2DInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CloughTocher2DInterpolator.html)

| Parameter | Options | Description |
| :--- | :--- | :--- |
| **`method`** | `'nearest'`, `'linear'`, `'cubic'` | Interpolation algorithm. |
| **`extrapolation`** | `'constant'`, `'nearest'` | `'nearest'` fills outside the convex hull with the closest value. |

---

## Development

If you want to contribute to `geointerp`, check out the [GEMINI.md](./GEMINI.md) and [CLAUDE.md](./CLAUDE.md) files for detailed architecture notes and development workflows.

### Running Tests
```bash
uv run pytest
```

---

## License

This project is licensed under the terms of the Apache Software License 2.0.
