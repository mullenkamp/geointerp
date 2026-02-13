# GEMINI.md

This file provides project-specific context and instructions for AI agents working on the `geointerp` codebase.

## Project Overview

`geointerp` is a Python package designed for efficient 2D and 3D geospatial interpolation, specifically optimized for data with a time dimension. It bridges the gap between raw interpolation algorithms and geospatial coordinate systems.

### Core Philosophy
- **NumPy-centric:** All inputs and outputs are NumPy `ndarrays`. No mandatory dependency on `xarray` or `pandas`.
- **Precomputation for Speed:** Uses a **callable factory pattern**. Methods (like `to_grid`) precompute coordinate transforms and index mappings, returning a optimized function (closure) that can be applied rapidly to multiple time steps.
- **Geospatial Aware:** Native support for Coordinate Reference Systems (CRS) via `pyproj`.
- **High Performance:** Leverages `scipy.ndimage.map_coordinates` for regular grids and precomputed `scipy.spatial.Delaunay` triangulations for scattered points.

### Architecture
- `geointerp/grid.py`: `GridInterpolator` class for regular grid data.
- `geointerp/points.py`: `PointInterpolator` class for scattered/irregular point data.
- `geointerp/util.py`: Shared utilities for coordinate-to-index mappings.
- `geointerp/tests/`: Comprehensive test suite using `pytest`.

## Building and Running

The project uses `uv` for dependency management and `hatchling` as the build backend.

### Environment Setup
```bash
# Install dependencies and create venv
uv sync
```

### Testing
```bash
# Run all tests with coverage
uv run pytest

# Run specific tests
uv run pytest geointerp/tests/test_grid.py
```

### Linting and Formatting
```bash
# Format code
uv run black .
uv run ruff check --fix .

# Type checking
uv run mypy geointerp/
```

### Documentation
```bash
# Serve documentation locally
hatch run docs-serve
```

## Development Conventions

### Coordinate Systems and Ordering
- **User-facing order:** Always use `(x, y)` or `(x, y, z)` order for coordinates, following geographic conventions (Longitude, Latitude, Elevation).
- **Internal array order:** Data arrays are typically `(ny, nx)` or `(nz, ny, nx)`. The library handles the mapping between geographic `(x, y)` and array `(i, j)` indices.
- **CRS Handling:** Always use `pyproj.Transformer.from_crs(..., always_xy=True)` to ensure consistent axis ordering regardless of the CRS definition.
- **Z-Dimension:** Vertical coordinates (z) are treated as CRS-independent and passed through transformations unchanged.

### Programming Patterns
- **Callable Factories:** Most interpolator methods should return a function. Example:
  ```python
  interpolator = GridInterpolator(from_crs="EPSG:4326")
  interp_func = interpolator.to_grid(source_coords, grid_res, to_crs="EPSG:2193")
  
  # Apply the precomputed function to many time steps
  for step in time_series:
      result = interp_func(step)
  ```
- **Error Handling:** Use `pyproj.CRS.from_user_input` for robust CRS parsing.
- **Performance:** Avoid expensive operations (like building a `Delaunay` triangulation or `Transformer`) inside the returned interpolation function. Move these to the factory method.

### Documentation & Standards
- Follow PEP 8 styles (enforced by `ruff` and `black`).
- Maintain type hints for all public APIs.
- Update `CLAUDE.md` if major architectural changes are made.
