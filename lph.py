import matplotlib.pyplot as plt
import numpy as np
import math


lph_order = [(0, -1), (1, -1), (1, 0), (1, 1),
             (0, 1), (-1, 1), (-1, 0), (-1, -1)]


def plot_hists(titles, hists, cols=2):
    rows = len(hists) // cols + 1
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))

    for i, hist in enumerate(hists):
        ax = axes[i // cols][i % cols]
        ax.bar(np.arange(256), hist)
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()


def histogram(samples, dim_y):
    samples_2d = transform_2d(samples, 20)
    vals = lph_vals(samples_2d)
    hist = np.bincount(vals, minlength=256)
    return hist


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


def transform_2d(samples, dim_y):
    dim_x = len(samples) // dim_y
    return samples[:dim_x * dim_y].reshape(dim_x, dim_y)


def transform_2d_full(samples):
    dim = int(math.sqrt(len(samples)))
    return samples[:dim * dim].reshape(dim, dim)
