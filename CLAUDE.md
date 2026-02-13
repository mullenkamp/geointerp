# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

geointerp is a Python package for 2D/3D geospatial interpolation. It wraps scipy's `map_coordinates` (regular grids) and `griddata`/`LinearNDInterpolator` (scattered points) with built-in CRS reprojection via pyproj. All inputs and outputs are numpy ndarrays.

## Build & Development

Uses **uv** for dependency management and **hatchling** as the build backend.

```bash
# Install in development mode
uv sync

# Run tests
uv run pytest

# Run a single test
uv run pytest geointerp/tests/test_name.py::test_function

# Lint
uv run ruff check .
uv run black --check .
uv run mypy --install-types --non-interactive geointerp/

# Format
uv run black .
uv run ruff check --fix .
```

pytest is configured in `pyproject.toml` with `--cov=geointerp/ --cov-report=term-missing`.

## Architecture

Two top-level classes, both following a **callable factory pattern**: methods precompute coordinate transforms and index mappings, then return a function that takes a single time-step array and returns the interpolated result. This avoids recomputing expensive setup on every call.

### `grid.py` — `GridInterpolator(from_crs=None)`
For data on regular grids. Uses `scipy.ndimage.map_coordinates` (fast spline interpolation).

- **`to_grid(source_coords, grid_res, ...)`** → `func(data_nd) → grid_nd`
  Precomputes output meshgrid, CRS transforms, and fractional array indices. Works for 2D `(ny, nx)` and 3D `(nz, ny, nx)`.
- **`to_points(source_coords, target_points, ...)`** → `func(data_nd) → values_1d`
  Precomputes target point locations as array indices.
- **`interp_na(source_coords, method)`** → `func(data_nd) → data_nd`
  Fills NaN using `griddata`. Coordinate meshgrid precomputed; NaN detection per-call.
- **`regrid_levels(target_levels, axis)`** → `func(data_3d, source_levels_3d) → data_3d`
  Regrids variable z-levels (terrain-following) to fixed levels. Vectorized linear interpolation with boundary clamping.

`source_coords` is always a tuple of 1D arrays in `(x, y)` or `(x, y, z)` order. Internally mapped to array dimension order `(y, x)` or `(z, y, x)`.

### `points.py` — `PointInterpolator(from_crs=None)`
For scattered/irregular point data. Uses `scipy.interpolate.LinearNDInterpolator` with precomputed `Delaunay` triangulation.

- **`to_grid(source_points, grid_res, ...)`** → `func(values_1d) → grid_nd`
  Precomputes Delaunay triangulation and output meshgrid. Supports `extrapolation='nearest'` fallback.
- **`to_points(source_points, target_points, ...)`** → `func(values_1d) → values_1d`
  Precomputes Delaunay on source points; evaluates at target locations.

### `util.py` — Coordinate conversion
- `grid_coords_to_index_params(coord_arrays)` → `(origins, spacings)` per dimension
- `coords_to_indices(points, origins, spacings)` → fractional array indices for `map_coordinates`
- `find_nearest(array, value)` → nearest element

### Key conventions
- User-facing coordinate order: `(x, y)` or `(x, y, z)` — geographic convention
- CRS applies only to x/y; z passes through unchanged
- Uses `Transformer.from_crs(..., always_xy=True)` for consistent axis ordering

## Key Dependencies

- **numpy**: Array containers, all inputs/outputs
- **scipy**: `map_coordinates` for grids, `Delaunay`/`LinearNDInterpolator` for points, `griddata` for NaN filling
- **pyproj**: CRS definitions and coordinate transformations
- **cupy** (optional): GPU acceleration
