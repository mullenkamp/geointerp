"""
Area weights from fine raster units to the cells of a coarse regular grid.

For each unit (a catchment, an elevation band, any labelled region on a fine raster) the table holds
the coarse cells it overlaps and the fraction of the unit's area in each. A unit's value is then the
weighted mean of its cells' values. Built by mapping every fine pixel centre into the coarse grid's
CRS and counting pixels per cell, which is exact to the fine pixel size (a 32 m raster against 3 km
cells) and handles any rotation or projection between the two.

Follows the package's callable-factory pattern: :func:`area_weights` does the expensive mapping
once and returns an :class:`AreaWeights` object that applies the table to any number of fields.

**WARNING:**
   **A round trip through one transformer cannot test that transformer.** A wrong but
   self-consistent CRS (a datum handled differently, a sphere treated as an ellipsoid) moves every
   unit by the same offset in both directions, and every internal check still passes. Anchor the
   mapping against coordinates the grid's producer wrote itself (e.g. WRF's ``XLAT``/``XLONG``)
   with :func:`grid_anchor_residual`, which goes through the same transform as the weights.
"""
from dataclasses import dataclass, replace

import numpy as np
from pyproj import CRS, Transformer


def _transformer(from_crs, to_crs) -> Transformer:
    """The one transform used for weights and for the anchor check, so the two cannot diverge."""
    return Transformer.from_crs(CRS.from_user_input(from_crs), CRS.from_user_input(to_crs), always_xy=True)


def _axis(c, name):
    c = np.asarray(c, float)
    if c.ndim != 1 or len(c) < 2:
        raise ValueError(f'{name} must be a 1-D array of at least two cell centres')
    d = np.diff(c)
    if not (d > 0).all():
        raise ValueError(f'{name} must be strictly ascending')
    if not np.allclose(d, d[0], rtol=1e-6, atol=0):
        raise ValueError(f'{name} must be regularly spaced')
    return c[0], d[0], len(c)


def _cell_index(v, origin, step, n, name):
    i = np.floor((v - origin) / step + 0.5).astype(np.int64)
    if (i < 0).any() or (i >= n).any():
        raise ValueError(f'{int(((i < 0) | (i >= n)).sum())} unit pixels fall outside the grid along {name}')
    return i


@dataclass(frozen=True)
class AreaWeights:
    """
    A unit-to-cell area-weight table. Call it on a field to get per-unit weighted means.

    Rows are sorted by unit, then by cell. Cell indices are into the full grid passed to
    :func:`area_weights`, as ``(iy, ix)``.
    """

    units: np.ndarray
    """The distinct unit labels, ascending."""
    unit: np.ndarray
    """Unit label of each row."""
    iy: np.ndarray
    """Grid row (y) index of each row."""
    ix: np.ndarray
    """Grid column (x) index of each row."""
    fraction: np.ndarray
    """Fraction of the unit's area in that cell; sums to 1 per unit."""
    unit_area: np.ndarray
    """Area of each unit in the units' CRS (pixel count times pixel area), aligned with ``units``."""
    grid_shape: tuple
    """(ny, nx) of the full grid."""

    def window(self) -> tuple:
        """``(slice_y, slice_x)`` of the smallest grid window holding every weighted cell."""
        return slice(int(self.iy.min()), int(self.iy.max()) + 1), slice(int(self.ix.min()), int(self.ix.max()) + 1)

    def __call__(self, field, offset=None) -> np.ndarray:
        """
        Per-unit weighted means of a field.

        Parameters
        ----------
        field : array-like
            Shape ``(..., ny, nx)``: the full grid when ``offset`` is None, or a window of it whose first
            cell is at grid index ``offset``.
        offset : (int, int) or None
            ``(iy0, ix0)`` of the window's first cell in the full grid. With None the field must be the
            full grid, exactly; a transposed or wrongly sized field is refused rather than indexed.

        Returns
        -------
        np.ndarray
            Shape ``(..., n_units)``, in the order of :attr:`units`. NaN in any weighted cell makes that
            unit NaN.
        """
        field = np.asarray(field)
        ny, nx = field.shape[-2:]
        if offset is None:
            if (ny, nx) != tuple(self.grid_shape):
                raise ValueError(f'field shape {(ny, nx)} is not the grid shape {tuple(self.grid_shape)}; '
                                 f'pass offset= for a window')
            offset = (0, 0)
        iy0, ix0 = (int(o) for o in offset)
        if iy0 < 0 or ix0 < 0 or iy0 + ny > self.grid_shape[0] or ix0 + nx > self.grid_shape[1]:
            raise ValueError(f'a {(ny, nx)} window at offset {(iy0, ix0)} does not fit the grid '
                             f'{tuple(self.grid_shape)}')
        iy = self.iy - iy0
        ix = self.ix - ix0
        if (iy < 0).any() or (ix < 0).any() or (iy >= ny).any() or (ix >= nx).any():
            raise ValueError(
                f'the field window (offset {offset}, shape {(ny, nx)}) does not hold every weighted cell; '
                f'needed rows {self.iy.min()}-{self.iy.max()}, columns {self.ix.min()}-{self.ix.max()}')
        vals = field[..., iy, ix] * self.fraction
        starts = np.flatnonzero(np.r_[True, self.unit[1:] != self.unit[:-1]])
        return np.add.reduceat(vals, starts, axis=-1)

    def without(self, drop, offset=None) -> 'AreaWeights':
        """
        The table without the cells flagged in ``drop``, each unit's fractions renormalised to sum to 1.

        For fields that are meaningless at some cells: a land-surface field at cells the grid's model
        treats as water, for example. The unit keeps its area (:attr:`unit_area` is unchanged); only the
        cells that represent it change. Use :meth:`kept_share` to record how much of each unit is left.

        Parameters
        ----------
        drop : array-like of bool, shape (ny, nx)
            True at cells to leave out: the full grid when ``offset`` is None, or a window of it whose
            first cell is at grid index ``offset``. It must cover every weighted cell.
        offset : (int, int) or None
            ``(iy0, ix0)`` of the window's first cell in the full grid.

        Returns
        -------
        AreaWeights

        Raises
        ------
        ValueError
            If the mask does not cover every weighted cell, or a unit would have no cell left (an empty
            mean is refused rather than returned as NaN).
        """
        drop = np.asarray(drop, bool)
        if drop.ndim != 2:
            raise ValueError(f'drop must be 2-D, got shape {drop.shape}')
        if offset is None:
            if drop.shape != tuple(self.grid_shape):
                raise ValueError(f'drop shape {drop.shape} is not the grid shape {tuple(self.grid_shape)}; '
                                 f'pass offset= for a window')
            offset = (0, 0)
        iy = self.iy - int(offset[0])
        ix = self.ix - int(offset[1])
        if (iy < 0).any() or (ix < 0).any() or (iy >= drop.shape[0]).any() or (ix >= drop.shape[1]).any():
            raise ValueError('the drop mask does not cover every weighted cell')
        keep = ~drop[iy, ix]
        gone = sorted(set(self.units.tolist()) - set(self.unit[keep].tolist()))
        if gone:
            raise ValueError(f'units {gone} would have no cell left')
        unit, frac = self.unit[keep], self.fraction[keep]
        pos = np.searchsorted(self.units, unit)
        total = np.bincount(pos, weights=frac, minlength=len(self.units))
        return replace(self, unit=unit, iy=self.iy[keep], ix=self.ix[keep], fraction=frac / total[pos])

    def kept_share(self, subset: 'AreaWeights') -> np.ndarray:
        """
        Share of each unit's area whose cells ``subset`` (from :meth:`without`) still holds, in the order
        of :attr:`units`.
        """
        if not np.array_equal(subset.units, self.units) or tuple(subset.grid_shape) != tuple(self.grid_shape):
            raise ValueError('subset is not a table of the same units on the same grid')
        ny, nx = self.grid_shape
        mine = self.unit.astype(np.int64) * ny * nx + self.iy.astype(np.int64) * nx + self.ix
        theirs = subset.unit.astype(np.int64) * ny * nx + subset.iy.astype(np.int64) * nx + subset.ix
        held = np.isin(mine, theirs)
        return np.bincount(np.searchsorted(self.units, self.unit[held]), weights=self.fraction[held],
                           minlength=len(self.units))


def area_weights(labels, geotransform, crs, grid_x, grid_y, grid_crs, nodata=0,
                 rows_per_block: int = 512) -> AreaWeights:
    """
    Build the area-weight table from a raster of unit labels to a regular coarse grid.

    Parameters
    ----------
    labels : array-like of int, shape (rows, cols)
        Unit label of each fine pixel; ``nodata`` marks pixels in no unit.
    geotransform : sequence of 6 floats
        GDAL geotransform of ``labels``: ``(x0, pixel_width, 0, y0, 0, pixel_height)``, with ``x0, y0``
        the outer corner of the first pixel. Rotated rasters are refused.
    crs : anything pyproj accepts
        CRS of ``labels`` (e.g. 2193 for NZTM).
    grid_x, grid_y : array-like
        Cell-centre coordinates of the coarse grid in ``grid_crs``, 1-D, ascending and regular. Take
        them, and ``grid_crs``, from the grid's own dataset rather than typing them by hand.
    grid_crs : anything pyproj accepts
        CRS of the coarse grid.
    nodata : int
        Label of pixels that belong to no unit.
    rows_per_block : int
        Raster rows transformed at once; bounds memory for large rasters.

    Returns
    -------
    AreaWeights

    Raises
    ------
    ValueError
        If no pixel is labelled, the raster is rotated, the grid axes are not regular, or any labelled
        pixel maps outside the grid.
    """
    lab = np.asarray(labels)
    if lab.ndim != 2:
        raise ValueError('labels must be 2-D')
    x0, pw, r1, y0, r2, ph = (float(v) for v in geotransform)
    if r1 != 0 or r2 != 0:
        raise ValueError('rotated rasters are not supported (geotransform terms 2 and 4 must be 0)')
    gx0, gdx, nx = _axis(grid_x, 'grid_x')
    gy0, gdy, ny = _axis(grid_y, 'grid_y')
    tr = _transformer(crs, grid_crs)

    keys = []
    for r0 in range(0, lab.shape[0], rows_per_block):
        block = lab[r0:r0 + rows_per_block]
        rr, cc = np.nonzero(block != nodata)
        if rr.size == 0:
            continue
        px = x0 + (cc + 0.5) * pw
        py = y0 + (r0 + rr + 0.5) * ph
        gx, gy = tr.transform(px, py)
        ix = _cell_index(np.asarray(gx), gx0, gdx, nx, 'x')
        iy = _cell_index(np.asarray(gy), gy0, gdy, ny, 'y')
        k = np.stack([block[rr, cc].astype(np.int64), iy, ix], axis=1)
        u, c = np.unique(k, axis=0, return_counts=True)
        keys.append(np.column_stack([u, c]))
    if not keys:
        raise ValueError('no labelled pixels')
    allk = np.concatenate(keys)
    u, inv = np.unique(allk[:, :3], axis=0, return_inverse=True)
    counts = np.bincount(inv.ravel(), weights=allk[:, 3]).astype(np.int64)
    unit, iy, ix = u[:, 0], u[:, 1], u[:, 2]
    units, uinv = np.unique(unit, return_inverse=True)
    totals = np.bincount(uinv, weights=counts)
    return AreaWeights(
        units=units,
        unit=unit,
        iy=iy,
        ix=ix,
        fraction=counts / totals[uinv],
        unit_area=totals * abs(pw * ph),
        grid_shape=(ny, nx),
    )


def grid_anchor_residual(grid_x, grid_y, grid_crs, lon, lat, via_crs, iy=None, ix=None,
                         lonlat_crs='EPSG:4326') -> np.ndarray:
    """
    Distance between the grid's cell centres and the producer's own lon/lat for those cells, measured
    through the same transform the weights use.

    Each ``(lon, lat)`` is taken to ``via_crs`` (the units' CRS) by a standard transform, then to
    ``grid_crs`` by exactly the transform :func:`area_weights` uses, and compared with the cell
    centre. A healthy mapping gives metres; a datum or projection mistake in the weights' transform
    gives kilometres.

    Parameters
    ----------
    grid_x, grid_y : array-like
        Cell-centre axes of the grid, as passed to :func:`area_weights`.
    grid_crs : anything pyproj accepts
        The grid's CRS.
    lon, lat : array-like, shape (ny, nx)
        The producer's cell-centre longitude and latitude (e.g. WRF ``XLONG``, ``XLAT``).
    via_crs : anything pyproj accepts
        The CRS of the units' raster.
    iy, ix : array-like of int, optional
        Cells to check (default: every cell).
    lonlat_crs : anything pyproj accepts
        CRS of ``lon``/``lat``.

    Returns
    -------
    np.ndarray
        Residual distance in grid-CRS units (metres for a projected grid), one per checked cell.
    """
    gx0, gdx, nx = _axis(grid_x, 'grid_x')
    gy0, gdy, ny = _axis(grid_y, 'grid_y')
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    if lon.shape != (ny, nx) or lat.shape != (ny, nx):
        raise ValueError(f'lon and lat must have the grid shape {(ny, nx)}')
    if iy is None:
        iy, ix = np.indices((ny, nx))
        iy, ix = iy.ravel(), ix.ravel()
    iy = np.asarray(iy)
    ix = np.asarray(ix)
    vx, vy = _transformer(lonlat_crs, via_crs).transform(lon[iy, ix], lat[iy, ix])
    gx, gy = _transformer(via_crs, grid_crs).transform(vx, vy)
    return np.hypot(np.asarray(gx) - (gx0 + ix * gdx), np.asarray(gy) - (gy0 + iy * gdy))
