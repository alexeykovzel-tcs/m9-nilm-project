import numpy as np
import math


lph_order = [(0, -1), (1, -1), (1, 0), (1, 1),
             (0, 1), (-1, 1), (-1, 0), (-1, -1)]


def lph_histogram(samples):
    samples_2d = transform_2d(samples)
    vals = lph_vals(samples_2d)
    hist = np.bincount(vals, minlength=256)
    return hist / hist.sum()


def lph_vals(samples_2d):
    dim = len(samples_2d)
    return [lph_val(samples_2d, row, col)
            for row in range(1, dim - 2)
            for col in range(1, dim - 2)]


def lph_val(samples_2d, row, col):
    ctr_val = samples_2d[row][col]
    value = 0
    for n, (i, j) in enumerate(lph_order):
        neighbor_val = samples_2d[row + i][col + j]
        bit_val = 1 if neighbor_val - ctr_val >= 0 else 0
        value += bit_val << n
    return value


def transform_2d(samples):
    dim = int(math.sqrt(len(samples)))
    return samples[:dim * dim].reshape(dim, dim)
