from src.signals import Signal
import numpy as np
import math


def sign_signal(signal: Signal):
    return _sign_samples(signal.vals, 40)


def _sign_samples(samples, dim_y):
    samples_2d = _transform_2d(samples, dim_y)
    lph_vals = _iter_lph_vals(samples_2d)
    hist = np.bincount(lph_vals, minlength=256)
    return hist / hist.sum()


def _iter_lph_vals(samples_2d):
    dim_x, dim_y = samples_2d.shape
    return [_lph_val(samples_2d, row, col)
            for row in range(1, dim_x - 2)
            for col in range(1, dim_y - 2)]


lph_order = [(0, -1), (1, -1), (1, 0), (1, 1),
             (0, 1), (-1, 1), (-1, 0), (-1, -1)]


def _lph_val(samples_2d, row, col):
    ctr_val = samples_2d[row][col]
    pattern = 0
    for n, (i, j) in enumerate(lph_order):
        neighbor_val = samples_2d[row + i][col + j]
        bit_val = 1 if neighbor_val - ctr_val >= 0 else 0
        pattern += bit_val << n
    return pattern


def _transform_2d(samples, dim_y):
    dim_x = len(samples) // dim_y
    return samples[:dim_x * dim_y].reshape(dim_x, dim_y)


def _transform_square(samples):
    dim = int(math.sqrt(len(samples)))
    return samples[:dim * dim].reshape(dim, dim)
