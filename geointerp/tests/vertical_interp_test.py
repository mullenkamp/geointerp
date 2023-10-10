#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:01:37 2023

@author: mike
"""
import numpy as np
import xarray as xr
import salem
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d, LinearNDInterpolator
from timeit import timeit
from metpy.interpolate import interpolate_1d, interpolate_to_isosurface
from metpy.calc import find_bounding_indices
# from numba import jit
# import cupy as cp
# from cupyx.profiler import benchmark

########################################################
### Parameters

wrf_file_path = '/media/data01/data/UC/domain04/wrfout_d04_2020-01-04_00_00_00'


new_levels = [10, 20, 30, 80, 150, 200, 350, 500, 750, 1000, 1300, 1600, 2000, 2500, 3000, 4000, 5000, 7000, 10000]

########################################################
### Initial data loading

x1 = salem.open_wrf_dataset(wrf_file_path)

model_alt = x1['Z'].copy().load()
model_height = model_alt - model_alt.isel(bottom_top=0)

p = x1['P'].copy().load()
p_height = p - p.isel(bottom_top=0)

temp = x1['T'].copy().load()

dims = temp.dims
var = temp.values
source_levels = model_height.values

# cp_var = cp.asarray(var)
# cp_source_levels = cp.asarray(source_levels)


########################################################
### Functions


def to_fixed_heights(xr_data, variable):
    """

    """
    var = xr_data[variable].copy().load()
    level_index = var.dims.index('bottom_top')

    model_alt = x1['Z'].copy().load()
    model_height = (model_alt - model_alt.isel(bottom_top=0)).values

    new_heights = model_height.mean(axis=tuple(s for s in range(len(var.dims)) if s != level_index))

    new_shape = [s for i, s in enumerate(var.shape) if i != level_index]
    new_shape.append(len(new_heights))
    out = np.zeros(new_shape)

    it = np.ndindex(tuple(new_shape[:3]))

    var2 = np.moveaxis(var.values, level_index, -1)
    source_levels2 = np.moveaxis(model_height, level_index, -1)

    for i in it:
        levels_var = np.interp(new_heights, source_levels2[i], var2[i])
        out[i] = levels_var

    da1 = xr.DataArray(np.moveaxis(out, -1, level_index), dims=tuple(dim if dim != 'bottom_top' else 'height' for dim in var.dims),
                       coords={'time': xr_data['time'],
                               'height': new_heights,
                               'south_north': xr_data['south_north'],
                               'west_east': xr_data['west_east']
                               }
                       )

    return da1



def np_interp(dims, var, source_levels, new_levels):
    """

    """
    level_index = dims.index('bottom_top')

    # n_levels = var.shape[level_index]

    new_shape = [s for i, s in enumerate(var.shape) if i != level_index]
    new_shape.append(len(new_levels))
    out = np.zeros(new_shape)

    it = np.ndindex(tuple(new_shape[:3]))

    var2 = np.moveaxis(var, level_index, -1)
    source_levels2 = np.moveaxis(source_levels, level_index, -1)

    # n_levels_arr = np.asarray(range(n_levels))
    new_levels_arr = np.asarray(new_levels)

    for i in it:
        # levels_index = np.interp(new_levels_arr, source_levels2[i], n_levels_arr)
        # levels_var = np.interp(levels_index, n_levels_arr, var2[i])
        levels_var = np.interp(new_levels_arr, source_levels2[i], var2[i])
        out[i] = levels_var

    return out


def scipy_interp1d(dims, var, source_levels, new_levels):
    """

    """
    level_index = dims.index('bottom_top')

    new_shape = [s for i, s in enumerate(var.shape) if i != level_index]
    new_shape.append(len(new_levels))
    out = np.zeros(new_shape)

    it = np.ndindex(tuple(new_shape[:3]))

    var2 = np.moveaxis(var, level_index, -1)
    source_levels2 = np.moveaxis(source_levels, level_index, -1)

    new_levels_arr = np.asarray(new_levels)

    for i in it:
        # levels_index = interp1d(source_levels2[i], n_levels_arr)(new_levels_arr)
        # levels_var = interp1d(n_levels_arr, var2[i])(levels_index)
        levels_var = interp1d(source_levels2[i], var2[i])(new_levels_arr)
        out[i] = levels_var

    return out


def metpy_interpolate1d(dims, var, source_levels, new_levels):
    """

    """
    level_index = dims.index('bottom_top')

    new_shape = [s for i, s in enumerate(var.shape) if i != level_index]
    new_shape.append(len(new_levels))
    out = np.zeros(new_shape)

    it = np.ndindex(tuple(new_shape[:3]))

    var2 = np.moveaxis(var, level_index, -1)
    source_levels2 = np.moveaxis(source_levels, level_index, -1)

    new_levels_arr = np.asarray(new_levels)

    for i in it:
        # levels_index = interpolate_1d(new_levels_arr, source_levels2[i], n_levels_arr)
        # levels_var = interpolate_1d(levels_index, n_levels_arr, var2[i])
        levels_var = interpolate_1d(new_levels_arr, source_levels2[i], var2[i])
        out[i] = levels_var

    return out


def metpy_interpolate_to_isosurface(dims, var, source_levels, new_levels):
    """

    """
    time_index = dims.index('time')
    level_index = dims.index('bottom_top')

    n_times = var.shape[time_index]

    new_shape = [s if i != level_index else len(new_levels) for i, s in enumerate(var.shape)]
    out = np.zeros(new_shape)

    new_levels_arr = np.asarray(new_levels)

    for t in range(n_times):
        source_levels_3d = source_levels[t]
        var_3d = var[t]
        for i, level in enumerate(new_levels_arr):
            levels_var = interpolate_to_isosurface(source_levels_3d, var_3d, level)
            out[t, i] = levels_var

    return out


def metpy_custom1(dims, var, source_levels, new_levels):
    """

    """
    time_index = dims.index('time')
    level_index = dims.index('bottom_top')
    x_index = dims.index('west_east')
    y_index = dims.index('south_north')

    n_times = var.shape[time_index]
    n_xs = var.shape[x_index]
    n_ys = var.shape[y_index]
    n_levels = len(new_levels)

    new_shape = (n_times, n_levels, n_ys, n_xs)
    out = np.zeros(new_shape)

    new_levels_arr = np.asarray(new_levels)

    for t in range(n_times):
        source_levels_3d = source_levels[t]
        var_3d = var[t]

        above, below, good = find_bounding_indices(source_levels_3d, new_levels_arr, axis=level_index - 1, from_below=True)

        tile_levels = np.repeat(new_levels_arr, n_xs*n_ys).reshape(n_levels, n_ys, n_xs)
        interp_level = (((tile_levels - source_levels_3d[above]) / (source_levels_3d[below] - source_levels_3d[above])) * (var_3d[below] - var_3d[above])) + var_3d[above]

        out[t] = interp_level

    return out


def metpy_custom2(dims, var, source_levels, new_levels):
    """

    """
    level_index = dims.index('bottom_top')

    new_shape = tuple(s if i != level_index else len(new_levels) for i, s in enumerate(var.shape))

    new_levels_arr = np.asarray(new_levels)

    above, below, good = find_bounding_indices(source_levels, new_levels_arr, axis=level_index, from_below=True)

    tile_levels = np.repeat(new_levels_arr, np.prod(tuple(s for i, s in enumerate(var.shape) if i != level_index))).reshape(new_shape)

    interp_level = (((tile_levels - source_levels[above]) / (source_levels[below] - source_levels[above])) * (var[below] - var[above])) + var[above]

    return interp_level



# def cupy_interp(dims, cp_var, cp_source_levels, new_levels):
#     """

#     """
#     # time_index = dims.index('time')
#     level_index = dims.index('bottom_top')
#     # x_index = dims.index('west_east')
#     # y_index = dims.index('south_north')

#     n_levels = cp_var.shape[level_index]

#     new_shape = [s for i, s in enumerate(var.shape) if i != level_index]
#     new_shape.append(len(new_levels))
#     out = cp.zeros(new_shape)

#     it = cp.ndindex(tuple(new_shape[:3]))

#     var2 = cp.moveaxis(cp_var, level_index, -1)
#     source_levels2 = cp.moveaxis(cp_source_levels, level_index, -1)

#     n_levels_arr = cp.asarray(range(n_levels))
#     new_levels_arr = cp.asarray(new_levels)

#     for i in it:
#         levels_index = cp.interp(new_levels_arr, source_levels2[i], n_levels_arr)
#         levels_var = cp.interp(levels_index, n_levels_arr, var2[i])
#         out[i] = levels_var

#     return out


# print(benchmark(cupy_interp, (dims, cp_var, cp_source_levels, new_levels), n_repeat=5))


level_var = source_levels_3d
interp_var = var_3d

above, below, good = find_bounding_indices(level_var, new_levels_arr, axis=0, from_below=True)

tile_levels = np.repeat(new_levels_arr, 102*102).reshape(19, 102, 102)

interp_level_3d = (((tile_levels - level_var[above]) / (level_var[below] - level_var[above])) * (interp_var[below] - interp_var[above])) + interp_var[above]



level_var = source_levels
interp_var = var

above, below, good = find_bounding_indices(level_var, new_levels_arr, axis=1, from_below=True)

tile_levels = np.repeat(new_levels_arr, 193*102*102).reshape(193, 19, 102, 102)

interp_level = (((tile_levels - level_var[above]) / (level_var[below] - level_var[above])) * (interp_var[below] - interp_var[above])) + interp_var[above]


###################################################
### map_coordinates testing

time_index = dims.index('time')
level_index = dims.index('bottom_top')
x_index = dims.index('west_east')
y_index = dims.index('south_north')

n_times = var.shape[time_index]
n_xs = var.shape[x_index]
n_ys = var.shape[y_index]
n_zs = var.shape[level_index]
n_levels = len(new_levels)

var2 = np.moveaxis(var[100], 0, -1)

rng = np.random.default_rng()

xs = rng.random(30) * n_xs
ys = rng.random(30) * n_ys
zs = rng.random(30) * n_zs

output = map_coordinates(var2, [ys, xs, zs], order=1)

ygrid = np.tile(range(1, n_ys+1), n_xs*n_zs)
xgrid = np.tile(range(1, n_xs+1), n_ys*n_zs)
zgrid = np.tile(range(1, n_zs+1), n_xs*n_ys)

points = list(zip(ygrid, xgrid, zgrid))

interp = LinearNDInterpolator(points, var2.ravel())


x = rng.random(10) - 0.5
y = rng.random(10) - 0.5
z = np.hypot(x, y)
X = np.linspace(min(x), max(x))
Y = np.linspace(min(y), max(y))
X, Y = np.meshgrid(X, Y)

list(zip(x, y))






































