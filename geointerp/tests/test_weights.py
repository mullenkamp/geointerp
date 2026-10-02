"""
Tests for geointerp.weights, failure case first. The grid is the WRF d03 Lambert grid (sphere,
rotated about 37.5 degrees against NZTM); the units are rasters in NZTM.
"""
import numpy as np
import pytest
from pyproj import CRS, Transformer

from geointerp import weights
from geointerp.weights import area_weights, grid_anchor_residual

LCC = CRS.from_proj4('+proj=lcc +lat_1=-39.619 +lat_2=-39.619 +lat_0=-37.742 +lon_0=-129.917 '
                     '+x_0=0 +y_0=0 +R=6370000 +units=m +no_defs')
NZTM = 2193
D03_X = -5064939.9 + 3000.0 * np.arange(315)
D03_Y = -2719580.3 + 3000.0 * np.arange(534)
PIX = 32.0


def _subgrid(nzx, nzy, nx=16, ny=14):
    """A window of the d03 axes whose cell (ny // 2, nx // 2) contains the NZTM point."""
    gx, gy = Transformer.from_crs(NZTM, LCC, always_xy=True).transform(nzx, nzy)
    i = int(np.floor((gx - D03_X[0]) / 3000 + 0.5))
    j = int(np.floor((gy - D03_Y[0]) / 3000 + 0.5))
    return D03_X[i - nx // 2:i - nx // 2 + nx], D03_Y[j - ny // 2:j - ny // 2 + ny]


def _in_polygon(px, py, poly):
    """Crossing-number point-in-polygon, vectorised over points."""
    inside = np.zeros(px.shape, bool)
    xs, ys = poly[:, 0], poly[:, 1]
    for k in range(len(poly)):
        x1, y1, x2, y2 = xs[k - 1], ys[k - 1], xs[k], ys[k]
        cond = (y1 > py) != (y2 > py)
        xi = x1 + (py - y1) * (x2 - x1) / np.where(y2 != y1, y2 - y1, 1)
        inside ^= cond & (px < xi)
    return inside


def _rasterise(polys, x0, y1, ncols, nrows):
    """Label raster (north-up, pixel PIX) with label i+1 for pixels whose centre is in polys[i]."""
    cx = x0 + (np.arange(ncols) + 0.5) * PIX
    cy = y1 - (np.arange(nrows) + 0.5) * PIX
    px, py = np.meshgrid(cx, cy)
    lab = np.zeros((nrows, ncols), np.int32)
    for i, poly in enumerate(polys):
        lab[_in_polygon(px, py, np.asarray(poly))] = i + 1
    return lab, (x0, PIX, 0.0, y1, 0.0, -PIX)


def _cell_footprint_nztm(gx, gy, iy, ix, densify=20):
    """The polygon of grid cell (iy, ix) in NZTM, edges densified."""
    xc, yc = gx[ix], gy[iy]
    t = np.linspace(-1500, 1500, densify, endpoint=False)
    ring = np.r_[np.c_[t, np.full_like(t, -1500)], np.c_[np.full_like(t, 1500), t],
                 np.c_[-t, np.full_like(t, 1500)], np.c_[np.full_like(t, -1500), -t]]
    x, y = Transformer.from_crs(LCC, NZTM, always_xy=True).transform(xc + ring[:, 0], yc + ring[:, 1])
    return np.c_[x, y]


# --- basic properties ---------------------------------------------------------------------------


def test_fractions_sum_to_one_and_area_matches_polygon():
    gx, gy = _subgrid(1460000, 5140000)
    sq = np.array([[1450000, 5130000], [1470000, 5130000], [1470000, 5150000], [1450000, 5150000]])
    lab, gt = _rasterise([sq], 1445000, 5155000, 900, 900)
    w = area_weights(lab, gt, NZTM, gx, gy, LCC)
    assert w.units.tolist() == [1]
    assert w.fraction.sum() == pytest.approx(1.0)
    assert w.unit_area[0] == pytest.approx(400e6, rel=0.005)


def test_refuses_pixels_outside_the_grid():
    gx, gy = _subgrid(1460000, 5140000, nx=3, ny=3)
    sq = np.array([[1450000, 5130000], [1470000, 5130000], [1470000, 5150000], [1450000, 5150000]])
    lab, gt = _rasterise([sq], 1445000, 5155000, 900, 900)
    with pytest.raises(ValueError, match='outside the grid'):
        area_weights(lab, gt, NZTM, gx, gy, LCC)


def test_refuses_bad_inputs():
    gx, gy = _subgrid(1460000, 5140000)
    lab = np.zeros((10, 10), int)
    with pytest.raises(ValueError, match='no labelled'):
        area_weights(lab, (1460000, PIX, 0, 5140000, 0, -PIX), NZTM, gx, gy, LCC)
    lab[5, 5] = 1
    with pytest.raises(ValueError, match='rotated'):
        area_weights(lab, (1460000, PIX, 0.1, 5140000, 0, -PIX), NZTM, gx, gy, LCC)
    with pytest.raises(ValueError, match='regularly'):
        area_weights(lab, (1460000, PIX, 0, 5140000, 0, -PIX), NZTM, np.r_[gx[:-1], gx[-1] + 10], gy, LCC)


# --- orientation: the rotated grid ---------------------------------------------------------------


@pytest.mark.parametrize(('iy', 'ix'), [(3, 11), (9, 2)])
def test_orientation_one_cell_footprint_lands_on_that_cell(iy, ix):
    # Defects caught: x/y swapped, a flipped axis, a rotation ignored. Asymmetric cells so a swap or
    # flip lands somewhere else.
    gx, gy = _subgrid(1460000, 5140000)
    poly = _cell_footprint_nztm(gx, gy, iy, ix)
    x0, y1 = poly[:, 0].min() - 500, poly[:, 1].max() + 500
    ncols = int((poly[:, 0].max() - x0 + 500) / PIX)
    nrows = int((y1 - poly[:, 1].min() + 500) / PIX)
    lab, gt = _rasterise([poly], x0, y1, ncols, nrows)
    w = area_weights(lab, gt, NZTM, gx, gy, LCC)
    on = w.fraction[(w.iy == iy) & (w.ix == ix)].sum()
    assert on >= 0.99


def test_exact_intersection_second_route():
    shapely = pytest.importorskip('shapely')
    gx, gy = _subgrid(1460000, 5140000)
    poly = np.array([[1452000, 5131000], [1468000, 5129000], [1471000, 5142000], [1462000, 5151000],
                     [1455000, 5146000], [1458000, 5139000]])
    lab, gt = _rasterise([poly], 1448000, 5155000, 800, 850)
    w = area_weights(lab, gt, NZTM, gx, gy, LCC)
    p = shapely.Polygon(poly)
    exact = {}
    for iy in range(len(gy)):
        for ix in range(len(gx)):
            a = shapely.Polygon(_cell_footprint_nztm(gx, gy, iy, ix)).intersection(p).area
            if a > 0:
                exact[(iy, ix)] = a / p.area
    ours = {(int(a), int(b)): f for a, b, f in zip(w.iy, w.ix, w.fraction, strict=True)}
    keys = set(exact) | set(ours)
    assert sum(abs(exact.get(k, 0) - ours.get(k, 0)) for k in keys) < 0.01


# --- the anchor ---------------------------------------------------------------------------------


def _producer_lonlat(gx, gy):
    """Cell-centre lon/lat as a WRF-like producer writes them: the grid's own (spherical) geodetic."""
    xx, yy = np.meshgrid(gx, gy)
    return Transformer.from_crs(LCC, LCC.geodetic_crs, always_xy=True).transform(xx, yy)


def test_anchor_is_metres_for_the_correct_transform():
    gx, gy = _subgrid(1460000, 5140000)
    lon, lat = _producer_lonlat(gx, gy)
    r = grid_anchor_residual(gx, gy, LCC, lon, lat, NZTM)
    assert r.max() < 25.0


class _GeocentricRoute:
    """A plausible wrong transform: NZTM -> GRS80 geocentric -> spherical geocentric -> grid."""

    def __init__(self, to_crs):
        self.a = Transformer.from_crs(NZTM, CRS.from_proj4('+proj=geocent +ellps=GRS80'), always_xy=True)
        self.b = Transformer.from_crs(CRS.from_proj4('+proj=geocent +R=6370000'), to_crs, always_xy=True)

    def transform(self, x, y):
        X, Y, Z = self.a.transform(x, y, np.zeros_like(np.asarray(x, float)))
        gx, gy, _ = self.b.transform(X, Y, Z)
        return gx, gy


def test_anchor_detects_a_self_consistent_wrong_transform(monkeypatch):
    # The defect the round trip cannot see: the same wrong transform in both directions.
    gx, gy = _subgrid(1460000, 5140000)
    lon, lat = _producer_lonlat(gx, gy)
    real = weights._transformer

    def wrong(a, b):
        if CRS.from_user_input(a) == CRS.from_user_input(NZTM) and CRS.from_user_input(b) == LCC:
            return _GeocentricRoute(b)
        return real(a, b)

    monkeypatch.setattr(weights, '_transformer', wrong)
    assert grid_anchor_residual(gx, gy, LCC, lon, lat, NZTM).min() > 10_000


# --- applying the table -------------------------------------------------------------------------


def test_apply_flat_index_through_a_window():
    # Defect caught: a flat-index or window-offset mismatch between the table and the field read.
    gx, gy = _subgrid(1460000, 5140000)
    a = np.array([[1452000, 5131000], [1462000, 5131000], [1462000, 5141000], [1452000, 5141000]])
    b = np.array([[1462000, 5137000], [1470000, 5137000], [1470000, 5150000], [1462000, 5150000]])
    lab, gt = _rasterise([a, b], 1448000, 5155000, 800, 850)
    w = area_weights(lab, gt, NZTM, gx, gy, LCC)
    ny, nx = w.grid_shape
    field = np.arange(ny * nx, dtype=float).reshape(ny, nx)
    expect = [np.sum(w.fraction[w.unit == u] * (w.iy[w.unit == u] * nx + w.ix[w.unit == u])) for u in w.units]
    np.testing.assert_allclose(w(field), expect)
    sy, sx = w.window()
    np.testing.assert_allclose(w(field[sy, sx], offset=(sy.start, sx.start)), expect)
    with pytest.raises(ValueError, match='does not hold'):
        w(field[sy, sx][:-1, :-1], offset=(sy.start, sx.start))


def test_apply_leading_dims_and_nan():
    gx, gy = _subgrid(1460000, 5140000)
    sq = np.array([[1452000, 5131000], [1462000, 5131000], [1462000, 5141000], [1452000, 5141000]])
    lab, gt = _rasterise([sq], 1448000, 5155000, 800, 850)
    w = area_weights(lab, gt, NZTM, gx, gy, LCC)
    ny, nx = w.grid_shape
    f = np.ones((3, ny, nx))
    f[1] *= 2
    np.testing.assert_allclose(w(f)[:, 0], [1.0, 2.0, 1.0])
    f[2, w.iy[0], w.ix[0]] = np.nan
    assert np.isnan(w(f)[2, 0])


def test_rows_per_block_does_not_change_the_table():
    gx, gy = _subgrid(1460000, 5140000)
    sq = np.array([[1452000, 5131000], [1462000, 5131000], [1462000, 5141000], [1452000, 5141000]])
    lab, gt = _rasterise([sq], 1448000, 5155000, 800, 850)
    w1 = area_weights(lab, gt, NZTM, gx, gy, LCC)
    w2 = area_weights(lab, gt, NZTM, gx, gy, LCC, rows_per_block=7)
    for f in ('unit', 'iy', 'ix', 'fraction'):
        np.testing.assert_array_equal(getattr(w1, f), getattr(w2, f))


def test_weights_and_anchor_share_one_transform(monkeypatch):
    # Defect caught: area_weights building its own transformer, so the anchor would check a transform
    # the weights do not use. Under the same wrong transform the weights must move too.
    gx, gy = _subgrid(1460000, 5140000, nx=40, ny=40)
    iy, ix = 20, 20
    poly = _cell_footprint_nztm(gx, gy, iy, ix)
    x0, y1 = poly[:, 0].min() - 500, poly[:, 1].max() + 500
    ncols = int((poly[:, 0].max() - x0 + 500) / PIX)
    nrows = int((y1 - poly[:, 1].min() + 500) / PIX)
    lab, gt = _rasterise([poly], x0, y1, ncols, nrows)
    real = weights._transformer

    def wrong(a, b):
        if CRS.from_user_input(a) == CRS.from_user_input(NZTM) and CRS.from_user_input(b) == LCC:
            return _GeocentricRoute(b)
        return real(a, b)

    monkeypatch.setattr(weights, '_transformer', wrong)
    w = area_weights(lab, gt, NZTM, gx, gy, LCC)
    assert w.fraction[(w.iy == iy) & (w.ix == ix)].sum() < 0.5


def test_pixel_centre_not_corner_decides_the_cell():
    # Defect caught: using a pixel's corner instead of its centre (a half-pixel shift). One pixel
    # straddles a cell edge at x = 1500: its corner is left of the edge, its centre right of it.
    gx = np.arange(0.0, 9000.0, 3000.0)  # centres 0, 3000, 6000: edges at 1500, 4500
    gy = np.arange(0.0, 9000.0, 3000.0)
    lab = np.array([[1]])
    for gt, cell in [((1500 - 8, PIX, 0.0, 3000 + 16, 0.0, -PIX), 1), ((1500 - 24, PIX, 0.0, 3000 + 16, 0.0, -PIX), 0)]:
        w = area_weights(lab, gt, NZTM, gx, gy, NZTM)
        assert w.ix.tolist() == [cell]



# --- review thalweg-step0-code-1 ----------------------------------------------------------------


def _identity_grid():
    """A 3 km grid in NZTM itself, so expected fractions are plain geometry."""
    gx = 1_450_000 + 3000.0 * np.arange(10)
    gy = 5_130_000 + 3000.0 * np.arange(10)
    return gx, gy


def test_multi_unit_fractions_against_independent_geometry():
    # Defect caught: normalising over all units (or merging units) instead of per unit. Unit 1 is a
    # 4.5 x 3 km rectangle over cells ix 2 (1.5 km) and ix 3 (3 km) in row iy 3: fractions 1/3, 2/3.
    # Unit 2 is a 3 x 3 km square exactly on cell (iy 5, ix 6): fraction 1.
    gx, gy = _identity_grid()
    # cell (iy, ix) spans x [gx[ix]-1500, gx[ix]+1500], y [gy[iy]-1500, gy[iy]+1500]
    u1 = np.array([[gx[2], gy[3] - 1500], [gx[3] + 1500, gy[3] - 1500], [gx[3] + 1500, gy[3] + 1500],
                   [gx[2], gy[3] + 1500]])
    u2 = np.array([[gx[6] - 1500, gy[5] - 1500], [gx[6] + 1500, gy[5] - 1500], [gx[6] + 1500, gy[5] + 1500],
                   [gx[6] - 1500, gy[5] + 1500]])
    x0, y1 = gx[0] - 1500, gy[-1] + 1500
    lab, gt = _rasterise([u1, u2], x0, y1, int(30000 / PIX), int(30000 / PIX))
    w = area_weights(lab, gt, NZTM, gx, gy, NZTM)
    got = {(int(u), int(a), int(b)): f for u, a, b, f in zip(w.unit, w.iy, w.ix, w.fraction, strict=True)}
    assert got[(1, 3, 2)] == pytest.approx(1 / 3, abs=0.01)
    assert got[(1, 3, 3)] == pytest.approx(2 / 3, abs=0.01)
    assert got[(2, 5, 6)] == pytest.approx(1.0)
    assert w.unit_area[0] == pytest.approx(4500 * 3000, rel=0.01)
    assert w.unit_area[1] == pytest.approx(3000 * 3000, rel=0.01)


def test_pixel_centre_decides_the_cell_in_y_too():
    gx = np.arange(0.0, 9000.0, 3000.0)
    gy = np.arange(0.0, 9000.0, 3000.0)  # y edges at 1500, 4500
    lab = np.array([[1]])
    # A pixel spanning y 1508..1476 (top-left corner y0 = 1508, height -32): centre 1492 -> row 0; its
    # corner 1508 would say row 1.
    w = area_weights(lab, (3000 - 16, PIX, 0.0, 1500 + 8, 0.0, -PIX), NZTM, gx, gy, NZTM)
    assert w.iy.tolist() == [0]
    w = area_weights(lab, (3000 - 16, PIX, 0.0, 1500 + 24, 0.0, -PIX), NZTM, gx, gy, NZTM)
    assert w.iy.tolist() == [1]


def test_call_refuses_a_field_that_is_not_the_grid():
    gx, gy = _subgrid(1460000, 5140000)
    sq = np.array([[1452000, 5131000], [1462000, 5131000], [1462000, 5141000], [1452000, 5141000]])
    lab, gt = _rasterise([sq], 1448000, 5155000, 800, 850)
    w = area_weights(lab, gt, NZTM, gx, gy, LCC)
    ny, nx = w.grid_shape
    with pytest.raises(ValueError, match='not the grid shape'):
        w(np.ones((nx, ny)))  # transposed
    with pytest.raises(ValueError, match='not the grid shape'):
        w(np.ones((ny - 3, nx)))  # a window passed without its offset
    with pytest.raises(ValueError, match='does not fit'):
        sy, sx = w.window()
        w(np.ones((ny, nx)), offset=(sy.start, sx.start))  # window larger than its offset allows


def test_anchor_sees_an_error_in_y_alone():
    # Defect caught: an anchor that compares only x (or only y). Shift the grid's y axis by 2 km.
    gx, gy = _subgrid(1460000, 5140000)
    lon, lat = _producer_lonlat(gx, gy)
    assert grid_anchor_residual(gx, gy + 2000.0, LCC, lon, lat, NZTM).min() > 1500
    assert grid_anchor_residual(gx + 2000.0, gy, LCC, lon, lat, NZTM).min() > 1500
