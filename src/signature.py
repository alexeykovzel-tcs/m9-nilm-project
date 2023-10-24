import matplotlib.pyplot as plt
import numpy as np
import math


lph_order = [(0, -1), (1, -1), (1, 0), (1, 1),
             (0, 1), (-1, 1), (-1, 0), (-1, -1)]


def closest_idx(arr, el):
    return min(range(len(arr)), key=lambda i: abs(arr[i] - el))


def sign_signal(signal):
    hist = histogram(signal, 40)
    return hist / hist.sum()


def trunc_tag_pf(data, tag):
    _, _, start, end = tag
    start_idx = closest_idx(data['TimeTicks'], start)
    end_idx = closest_idx(data['TimeTicks'], end)
    return data['Pf'][start_idx:end_idx]


def sign_tag(data, tag):
    signal = trunc_tag_pf(data, tag)
    return sign_signal(signal)


def sign_and_plot_tags(data):
    tags = data['TaggingInfo']
    tag_names = [tag[1] for tag in tags]
    hists = [sign_tag(data, tag) for tag in tags]
    plot_hists(tag_names, hists)


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
    samples_2d = _transform_2d(samples, dim_y)
    vals = _lph_vals(samples_2d)
    hist = np.bincount(vals, minlength=256)
    return hist


def _lph_vals(samples_2d):
    dim_x, dim_y = samples_2d.shape
    return [_lph_cell(samples_2d, row, col)
            for row in range(1, dim_x - 2)
            for col in range(1, dim_y - 2)]


def _lph_cell(samples_2d, row, col):
    ctr_val = samples_2d[row][col]
    value = 0
    for n, (i, j) in enumerate(lph_order):
        neighbor_val = samples_2d[row + i][col + j]
        bit_val = 1 if neighbor_val - ctr_val >= 0 else 0
        value += bit_val << n
    return value


def _transform_2d(samples, dim_y):
    dim_x = len(samples) // dim_y
    return samples[:dim_x * dim_y].reshape(dim_x, dim_y)


def _transform_square(samples):
    dim = int(math.sqrt(len(samples)))
    return samples[:dim * dim].reshape(dim, dim)
