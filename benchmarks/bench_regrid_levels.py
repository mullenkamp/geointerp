"""
Benchmark: regrid_levels (np.interp) vs vectorized argmax implementation.

The current regrid_levels uses np.interp per spatial column. This benchmark
compares it against an alternative that loops over target levels and uses
vectorized argmax/fancy-indexing across all spatial columns at once.
"""
import csv
import os
import time

import numpy as np

from geointerp import GridInterpolator


def _regrid_vectorized_argmax(data, source_levels, target_levels, axis=0):
    """
    Alternative regrid implementation using vectorized argmax.

    Loops over target levels (small dimension) and processes all spatial
    columns at once via fancy indexing. Faster at small grid sizes but
    slower at large sizes due to memory access patterns.
    """
    target_levels = np.asarray(target_levels, dtype=float)
    n_tgt = len(target_levels)

    data_moved = np.moveaxis(data, axis, 0)
    levels_moved = np.moveaxis(source_levels, axis, 0)

    spatial_shape = data_moved.shape[1:]
    n_src = data_moved.shape[0]

    data_flat = data_moved.reshape(n_src, -1)
    levels_flat = levels_moved.reshape(n_src, -1)
    n_spatial = data_flat.shape[1]

    out = np.empty((n_tgt, n_spatial), dtype=data.dtype)
    cols = np.arange(n_spatial)

    for k in range(n_tgt):
        tgt = target_levels[k]

        above_mask = levels_flat >= tgt
        above_idx = np.argmax(above_mask, axis=0)

        no_above = ~above_mask.any(axis=0)
        above_idx[no_above] = n_src - 1

        below_idx = np.clip(above_idx - 1, 0, n_src - 1)

        lev_below = levels_flat[below_idx, cols]
        lev_above = levels_flat[above_idx, cols]
        val_below = data_flat[below_idx, cols]
        val_above = data_flat[above_idx, cols]

        denom = lev_above - lev_below
        safe_denom = np.where(denom == 0, 1.0, denom)
        weight = np.clip(
            np.where(denom == 0, 0.0, (tgt - lev_below) / safe_denom),
            0.0, 1.0
        )

        out[k] = val_below + weight * (val_above - val_below)

    result = out.reshape((n_tgt,) + spatial_shape)
    return np.moveaxis(result, 0, axis)


def make_test_data(ny, nx, n_src, n_tgt, seed=42):
    """Generate test data for regrid_levels benchmarks."""
    rng = np.random.default_rng(seed)

    source_levels = np.zeros((n_src, ny, nx))
    for i in range(ny):
        for j in range(nx):
            inner = np.sort(rng.uniform(50, 950, n_src - 2))
            source_levels[:, i, j] = np.concatenate([[0], inner, [1000]])

    data = np.sin(source_levels / 300.0) + source_levels ** 0.5
    target_levels = np.linspace(0, 1000, n_tgt)

    return data, source_levels, target_levels


def bench_regrid_levels(rounds, ny, nx, n_src, n_tgt):
    data, source_levels, target_levels = make_test_data(ny, nx, n_src, n_tgt)

    gi = GridInterpolator()
    func = gi.regrid_levels(target_levels, axis=0)

    # Warmup
    func(data, source_levels)
    _regrid_vectorized_argmax(data, source_levels, target_levels)

    t0 = time.perf_counter()
    for _ in range(rounds):
        func(data, source_levels)
    t_np_interp = (time.perf_counter() - t0) / rounds

    t0 = time.perf_counter()
    for _ in range(rounds):
        _regrid_vectorized_argmax(data, source_levels, target_levels)
    t_argmax = (time.perf_counter() - t0) / rounds

    return t_np_interp, t_argmax


CONFIGS = [
    {"label": "Small  (20x30, 10 src, 20 tgt)", "ny": 20, "nx": 30, "n_src": 10, "n_tgt": 20},
    {"label": "Medium (50x80, 20 src, 40 tgt)", "ny": 50, "nx": 80, "n_src": 20, "n_tgt": 40},
    {"label": "Large  (100x200, 30 src, 50 tgt)", "ny": 100, "nx": 200, "n_src": 30, "n_tgt": 50},
    {"label": "XLarge (200x400, 40 src, 80 tgt)", "ny": 200, "nx": 400, "n_src": 40, "n_tgt": 80},
]


if __name__ == "__main__":
    results = []

    print(f"{'Config':<45} {'np.interp':>12} {'argmax':>12} {'Speedup':>10}")
    print("-" * 82)

    for cfg in CONFIGS:
        rounds = 50 if cfg["ny"] * cfg["nx"] < 10000 else 10
        t_interp, t_argmax = bench_regrid_levels(
            rounds, cfg["ny"], cfg["nx"], cfg["n_src"], cfg["n_tgt"]
        )
        speedup = t_argmax / t_interp
        print(f"{cfg['label']:<45} {t_interp*1000:>9.2f} ms {t_argmax*1000:>9.2f} ms {speedup:>9.1f}x")
        results.append({
            "config": cfg["label"],
            "ny": cfg["ny"],
            "nx": cfg["nx"],
            "n_src": cfg["n_src"],
            "n_tgt": cfg["n_tgt"],
            "np_interp_ms": round(t_interp * 1000, 3),
            "argmax_ms": round(t_argmax * 1000, 3),
            "speedup": round(speedup, 2),
        })

    out_path = os.path.join(os.path.dirname(__file__), "results_regrid_levels.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to {out_path}")
