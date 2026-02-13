import numpy as np
import pytest
from pyproj import CRS, Transformer

from geointerp import PointInterpolator


class TestPointsToGrid:

    def test_known_linear_function(self, scattered_2d_points):
        points, _ = scattered_2d_points
        # Use a simple linear function so interpolation is exact
        values = points[:, 0] + points[:, 1]

        pi = PointInterpolator()
        func = pi.to_grid(points, grid_res=5.0, method='linear')
        grid = func(values)

        # Check an interior grid point
        x_min, y_min = points.min(axis=0)
        mid_y, mid_x = grid.shape[0] // 2, grid.shape[1] // 2
        gx = x_min + mid_x * 5.0
        gy = y_min + mid_y * 5.0
        expected = gx + gy
        np.testing.assert_allclose(grid[mid_y, mid_x], expected, atol=0.5)

    def test_extrapolation_nearest(self):
        rng = np.random.default_rng(42)
        points = rng.uniform(20, 80, (100, 2))
        values = np.ones(100) * 7.0

        pi = PointInterpolator()
        func = pi.to_grid(points, grid_res=5.0, bbox=(0, 100, 0, 100),
                          method='linear', extrapolation='nearest')
        grid = func(values)
        assert not np.isnan(grid).any()
        np.testing.assert_allclose(grid, 7.0, atol=1e-10)

    def test_crs_transform(self):
        pi = PointInterpolator(from_crs=2193)
        source = np.array([
            [1750000, 5420000],
            [1755000, 5420000],
            [1750000, 5425000],
            [1755000, 5425000],
        ], dtype=float)
        values = np.array([10.0, 20.0, 30.0, 40.0])

        func = pi.to_grid(source, grid_res=0.01, to_crs=4326, method='linear')
        grid = func(values)
        assert grid.ndim == 2
        # Interior values should be within the range of input values
        valid = grid[~np.isnan(grid)]
        assert valid.min() >= 9.0
        assert valid.max() <= 41.0

    def test_3d_known_function(self):
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 100, (500, 3))
        values = points[:, 0] + points[:, 1] + points[:, 2]

        pi = PointInterpolator()
        func = pi.to_grid(points, grid_res=20.0, method='linear')
        grid = func(values)
        assert grid.ndim == 3

    def test_repeated_calls(self, scattered_2d_points):
        points, values = scattered_2d_points
        pi = PointInterpolator()
        func = pi.to_grid(points, grid_res=10.0, method='linear')
        r1 = func(values)
        r2 = func(values * 2)
        # Where both are valid, r2 should be 2x r1
        mask = ~np.isnan(r1) & ~np.isnan(r2) & (np.abs(r1) > 1e-10)
        np.testing.assert_allclose(r2[mask] / r1[mask], 2.0, atol=0.1)


class TestPointsToPoints:

    def test_at_source_points(self, scattered_2d_points):
        points, values = scattered_2d_points
        pi = PointInterpolator()
        func = pi.to_points(points, points[:10], method='linear')
        result = func(values)
        np.testing.assert_allclose(result, values[:10], atol=1e-10)

    def test_known_linear_function(self):
        rng = np.random.default_rng(42)
        source = rng.uniform(0, 100, (200, 2))
        values = source[:, 0] * 2 + source[:, 1] * 3

        target = np.array([[50.0, 50.0], [25.0, 75.0]])
        pi = PointInterpolator()
        func = pi.to_points(source, target, method='linear')
        result = func(values)
        expected = target[:, 0] * 2 + target[:, 1] * 3
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_crs_transform(self):
        pi = PointInterpolator(from_crs=2193)
        source = np.array([
            [1750000, 5420000],
            [1755000, 5420000],
            [1750000, 5425000],
            [1755000, 5425000],
            [1752500, 5422500],
        ], dtype=float)
        values = np.array([10.0, 20.0, 30.0, 40.0, 25.0])

        # Evaluate at source points (same CRS)
        func = pi.to_points(source, source[:3], to_crs=2193, method='linear')
        result = func(values)
        np.testing.assert_allclose(result, values[:3], atol=1e-10)

    def test_3d_points(self):
        rng = np.random.default_rng(42)
        source = rng.uniform(0, 100, (500, 3))
        values = source[:, 0] + source[:, 1] + source[:, 2]
        target = np.array([[50.0, 50.0, 50.0]])

        pi = PointInterpolator()
        func = pi.to_points(source, target, method='linear')
        result = func(values)
        np.testing.assert_allclose(result, [150.0], atol=1.0)

    def test_nearest_method(self, scattered_2d_points):
        points, values = scattered_2d_points
        pi = PointInterpolator()
        func = pi.to_points(points, points[:5], method='nearest')
        result = func(values)
        np.testing.assert_allclose(result, values[:5], atol=1e-10)
