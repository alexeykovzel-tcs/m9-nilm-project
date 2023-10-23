import matplotlib.pyplot as plt
import numpy as np
import math


lph_order = [(0, -1), (1, -1), (1, 0), (1, 1),
             (0, 1), (-1, 1), (-1, 0), (-1, -1)]

dim_y = 100


def plot(samples):
    l1_hist = histogram(samples)
    plt.bar(np.arange(256), l1_hist)
    plt.show()


def histogram(samples):
    samples_2d = transform_2d(samples)
    vals = lph_vals(samples_2d)
    hist = np.bincount(vals, minlength=256)
    return hist / hist.sum()


def lph_vals(samples_2d):
    dim_x, dim_y = samples_2d.shape
    return [lph_val(samples_2d, row, col)
            for row in range(1, dim_x - 2)
            for col in range(1, dim_y - 2)]


def lph_val(samples_2d, row, col):
    ctr_val = samples_2d[row][col]
    value = 0
    for n, (i, j) in enumerate(lph_order):
        neighbor_val = samples_2d[row + i][col + j]
        bit_val = 1 if neighbor_val - ctr_val >= 0 else 0
        value += bit_val << n
    return value


def transform_2d(samples):
    dim_x = len(samples) // dim_y
    return samples[:dim_x * dim_y].reshape(dim_x, dim_y)


def transform_2d_full(samples):
    dim = int(math.sqrt(len(samples)))
    return samples[:dim * dim].reshape(dim, dim)
