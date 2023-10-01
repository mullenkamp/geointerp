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
from scipy.interpolate import interp1d
from timeit import timeit
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

temp = x1['T'].copy().load()

dims = temp.dims
var = temp.values
source_levels = model_height.values

# cp_var = cp.asarray(var)
# cp_source_levels = cp.asarray(source_levels)


########################################################
### Functions


def np_interp(dims, var, source_levels, new_levels):
    """

    """
    level_index = dims.index('bottom_top')

    n_levels = var.shape[level_index]

    new_shape = [s for i, s in enumerate(var.shape) if i != level_index]
    new_shape.append(len(new_levels))
    out = np.zeros(new_shape)

    it = np.ndindex(tuple(new_shape[:3]))

    var2 = np.moveaxis(var, level_index, -1)
    source_levels2 = np.moveaxis(source_levels, level_index, -1)

    n_levels_arr = np.asarray(range(n_levels))
    new_levels_arr = np.asarray(new_levels)

    for i in it:
        levels_index = np.interp(new_levels_arr, source_levels2[i], n_levels_arr)
        levels_var = np.interp(levels_index, n_levels_arr, var2[i])
        out[i] = levels_var

    return out


def scipy_interp1d(dims, var, source_levels, new_levels):
    """

    """
    level_index = dims.index('bottom_top')

    n_levels = var.shape[level_index]

    new_shape = [s for i, s in enumerate(var.shape) if i != level_index]
    new_shape.append(len(new_levels))
    out = np.zeros(new_shape)

    it = np.ndindex(tuple(new_shape[:3]))

    var2 = np.moveaxis(var, level_index, -1)
    source_levels2 = np.moveaxis(source_levels, level_index, -1)

    n_levels_arr = np.asarray(range(n_levels))
    new_levels_arr = np.asarray(new_levels)

    for i in it:
        levels_index = interp1d(source_levels2[i], n_levels_arr)(new_levels_arr)
        levels_var = interp1d(n_levels_arr, var2[i])(levels_index)
        out[i] = levels_var

    return out


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




































































