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
