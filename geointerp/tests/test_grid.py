import numpy as np
import pytest
from pyproj import CRS, Transformer

from geointerp import GridInterpolator


class TestGridToGrid2D:

    def test_identity_no_crs(self, grid_2d_data):
        x, y, data = grid_2d_data
        gi = GridInterpolator()
        spacing_x = x[1] - x[0]
        spacing_y = y[1] - y[0]
        func = gi.to_grid((x, y), grid_res=(spacing_x, spacing_y), order=1)
        result = func(data)
        assert result.shape == data.shape
        np.testing.assert_allclose(result, data, atol=1e-10)

    def test_upscale_constant(self, grid_2d_data):
        x, y, _ = grid_2d_data
        ones = np.ones((len(y), len(x)))
        gi = GridInterpolator()
        func = gi.to_grid((x, y), grid_res=5.0, order=1)
        result = func(ones)
        assert result.shape[0] >= len(y)
        assert result.shape[1] >= len(x)
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_crs_transform(self):
        gi = GridInterpolator(from_crs=2193)
        x = np.arange(1750000, 1760000, 1000.0)
        y = np.arange(5420000, 5430000, 1000.0)
        data = np.ones((len(y), len(x))) * 42.0

        func = gi.to_grid((x, y), grid_res=0.01, to_crs=4326, order=1)
        result = func(data)
        assert result.ndim == 2
        # Edge cells may be NaN due to CRS warping; check valid interior
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        np.testing.assert_allclose(valid, 42.0, atol=1e-6)

    def test_bbox_clipping(self, grid_2d_data):
        x, y, data = grid_2d_data
        gi = GridInterpolator()
        bbox = (20.0, 60.0, 10.0, 50.0)
        func = gi.to_grid((x, y), grid_res=10.0, bbox=bbox, order=1)
        result = func(data)
        expected_nx = len(np.arange(20.0, 60.0 + 5.0, 10.0))
        expected_ny = len(np.arange(10.0, 50.0 + 5.0, 10.0))
        assert result.shape == (expected_ny, expected_nx)

    def test_min_val(self, grid_2d_data):
        x, y, data = grid_2d_data
        gi = GridInterpolator()
        func = gi.to_grid((x, y), grid_res=10.0, order=1, min_val=0.0)
        result = func(data)
        assert result.min() >= 0.0

    def test_extrapolation_constant(self):
        gi = GridInterpolator()
        x = np.arange(10.0, 50.0, 10.0)
        y = np.arange(10.0, 50.0, 10.0)
        data = np.ones((len(y), len(x)))
        bbox = (0.0, 60.0, 0.0, 60.0)
        func = gi.to_grid((x, y), grid_res=10.0, bbox=bbox, order=1,
                          extrapolation='constant', fill_val=-999.0)
        result = func(data)
        assert np.any(result == -999.0)

    def test_extrapolation_nearest(self):
        gi = GridInterpolator()
        x = np.arange(10.0, 50.0, 10.0)
        y = np.arange(10.0, 50.0, 10.0)
        data = np.ones((len(y), len(x))) * 5.0
        bbox = (0.0, 60.0, 0.0, 60.0)
        func = gi.to_grid((x, y), grid_res=10.0, bbox=bbox, order=1,
                          extrapolation='nearest')
        result = func(data)
        np.testing.assert_allclose(result, 5.0, atol=1e-10)


class TestGridToGrid3D:

    def test_3d_identity(self, grid_3d_data):
        x, y, z, data = grid_3d_data
        gi = GridInterpolator()
        sx = x[1] - x[0]
        sy = y[1] - y[0]
        sz = z[1] - z[0]
        func = gi.to_grid((x, y, z), grid_res=(sx, sy, sz), order=1)
        result = func(data)
        assert result.shape == data.shape
        np.testing.assert_allclose(result, data, atol=1e-10)

    def test_3d_upscale_constant(self, grid_3d_data):
        x, y, z, _ = grid_3d_data
        ones = np.ones((len(z), len(y), len(x)))
        gi = GridInterpolator()
        func = gi.to_grid((x, y, z), grid_res=5.0, order=1)
        result = func(ones)
        assert result.ndim == 3
        np.testing.assert_allclose(result, 1.0, atol=1e-10)


class TestGridToPoints:

    def test_at_grid_nodes(self, grid_2d_data):
        x, y, data = grid_2d_data
        target = np.array([[x[2], y[3]], [x[5], y[1]]])
        gi = GridInterpolator()
        func = gi.to_points((x, y), target, order=1)
        result = func(data)
        expected = np.array([data[3, 2], data[1, 5]])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_between_nodes_linear(self):
        gi = GridInterpolator()
        x = np.arange(0, 30, 10.0)
        y = np.arange(0, 30, 10.0)
        yy, xx = np.meshgrid(y, x, indexing='ij')
        data = (xx + yy).astype(float)
        target = np.array([[5.0, 5.0], [15.0, 15.0]])
        func = gi.to_points((x, y), target, order=1)
        result = func(data)
        np.testing.assert_allclose(result, [10.0, 30.0], atol=1e-10)

    def test_crs_transform(self):
        gi = GridInterpolator(from_crs=2193)
        x = np.arange(1750000, 1760000, 1000.0)
        y = np.arange(5420000, 5430000, 1000.0)
        data = np.ones((len(y), len(x))) * 42.0

        t = Transformer.from_crs(
            CRS.from_user_input(2193), CRS.from_user_input(4326), always_xy=True
        )
        cx, cy = t.transform(1755000, 5425000)
        target = np.array([[cx, cy]])
        func = gi.to_points((x, y), target, to_crs=4326, order=1)
        result = func(data)
        np.testing.assert_allclose(result, 42.0, atol=1e-6)

    def test_3d_at_nodes(self, grid_3d_data):
        x, y, z, data = grid_3d_data
        target = np.array([[x[2], y[3], z[1]]])
        gi = GridInterpolator()
        func = gi.to_points((x, y, z), target, order=1)
        result = func(data)
        np.testing.assert_allclose(result, [data[1, 3, 2]], atol=1e-10)


class TestGridInterpNa:

    def test_fill_interior_nans(self):
        gi = GridInterpolator()
        x = np.arange(0, 50, 10.0)
        y = np.arange(0, 50, 10.0)
        yy, xx = np.meshgrid(y, x, indexing='ij')
        data = (xx + yy).astype(float)
        data[2, 2] = np.nan
        func = gi.interp_na((x, y), method='linear')
        result = func(data)
        assert not np.isnan(result).any()
        np.testing.assert_allclose(result[2, 2], 40.0, atol=1e-10)

    def test_no_nans(self, grid_2d_data):
        x, y, data = grid_2d_data
        gi = GridInterpolator()
        func = gi.interp_na((x, y))
        result = func(data)
        np.testing.assert_allclose(result, data, atol=1e-10)


class TestRegridLevels:

    def test_uniform_levels(self):
        gi = GridInterpolator()
        ny, nx = 4, 5
        source_z = np.array([0, 100, 200, 300, 400, 500], dtype=float)
        source_levels = np.broadcast_to(
            source_z[:, None, None], (len(source_z), ny, nx)
        ).copy()
        data = source_levels * 2.0 + 10.0

        target_z = np.array([50, 150, 250, 350, 450], dtype=float)
        func = gi.regrid_levels(target_z, axis=0)
        result = func(data, source_levels)
        expected = target_z[:, None, None] * 2.0 + 10.0
        np.testing.assert_allclose(result, np.broadcast_to(expected, result.shape), atol=1e-10)

    def test_terrain_following(self):
        gi = GridInterpolator()
        ny, nx, n_src = 3, 4, 8
        rng = np.random.default_rng(42)

        # Variable levels per (y, x) but all spanning 0-1000
        source_levels = np.zeros((n_src, ny, nx))
        for i in range(ny):
            for j in range(nx):
                inner = np.sort(rng.uniform(50, 950, n_src - 2))
                source_levels[:, i, j] = np.concatenate([[0], inner, [1000]])

        # Linear function of level: data = 2 * level
        data = source_levels * 2.0

        # Target levels well within the 0-1000 range of all columns
        target_z = np.array([100, 300, 500, 700, 900], dtype=float)
        func = gi.regrid_levels(target_z, axis=0)
        result = func(data, source_levels)
        assert result.shape == (len(target_z), ny, nx)
        expected = target_z[:, None, None] * 2.0
        np.testing.assert_allclose(result, np.broadcast_to(expected, result.shape), atol=1e-10)

    def test_extrapolation_clamp(self):
        gi = GridInterpolator()
        source_levels = np.array([100, 200, 300], dtype=float).reshape(3, 1, 1)
        source_levels = np.broadcast_to(source_levels, (3, 2, 2)).copy()
        data = source_levels * 1.0

        target_z = np.array([50, 250, 400], dtype=float)
        func = gi.regrid_levels(target_z, axis=0)
        result = func(data, source_levels)
        assert result.shape == (3, 2, 2)
        # At boundaries, clamps to edge values
        np.testing.assert_allclose(result[0, 0, 0], 100.0, atol=1e-10)
        np.testing.assert_allclose(result[2, 0, 0], 300.0, atol=1e-10)

    # def test_matches_np_interp(self):
    #     """Verify regrid_levels matches np.interp column-by-column."""
    #     gi = GridInterpolator()
    #     ny, nx, n_src = 5, 6, 10
    #     rng = np.random.default_rng(99)

    #     # Variable source levels per (y, x), all spanning 0-1000
    #     source_levels = np.zeros((n_src, ny, nx))
    #     for i in range(ny):
    #         for j in range(nx):
    #             inner = np.sort(rng.uniform(50, 950, n_src - 2))
    #             source_levels[:, i, j] = np.concatenate([[0], inner, [1000]])

    #     # Non-trivial data: nonlinear function of level
    #     data = np.sin(source_levels / 300.0) + source_levels ** 0.5

    #     # Target levels spanning beyond source range to test clamping too
    #     target_z = np.array([-50, 0, 100, 250, 500, 750, 999, 1000, 1100], dtype=float)
    #     func = gi.regrid_levels(target_z, axis=0)
    #     result = func(data, source_levels)

    #     # Build reference with np.interp column-by-column
    #     expected = np.empty((len(target_z), ny, nx))
    #     for i in range(ny):
    #         for j in range(nx):
    #             expected[:, i, j] = np.interp(
    #                 target_z, source_levels[:, i, j], data[:, i, j]
    #             )

    #     np.testing.assert_allclose(result, expected, atol=1e-10)
