import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime
import numpy as np
import copy


l_keys = ['Datetimes', 'TimeTicks', 'Real', 'Reactive', 'Apparent', 'Pf']
hf_keys = ['HF_Datetimes', 'HF_TimeTicks', 'HF']


def plot_data(data, has_labels=False, time_range=None):
    _, axs = plt.subplots(4, 1, figsize=(10, 8))

    if time_range is not None:
        data = trunc_data(data, time_range)

    plot_real_power(data, axs[0])
    plot_reactive_power(data, axs[1])
    plot_power_factor(data, axs[2])
    plot_hf_noise(data, axs[3])

    for ax in axs:
        if has_labels and 'TaggingInfo' in data:
            add_device_tags(data, ax)

    plt.tight_layout(h_pad=2)
    plt.show()


def plot_tagged_data(data, has_labels=True):
    if 'TaggingInfo' not in data:
        print('there is no tagged data...')
    else:
        min_ts = min(x[2] for x in data['TaggingInfo'])
        max_ts = max(x[3] for x in data['TaggingInfo'])
        plot_data(data, time_range=(min_ts, max_ts), has_labels=has_labels)


def plot_real_power(data, ax=None, ts_range=None):
    plot_l(data, ax, ts_range, 'Real', 'Real Power (W)')


def plot_reactive_power(data, ax=None, ts_range=None):
    plot_l(data, ax, ts_range, 'Reactive', 'Reactive power (VAR)')


def plot_power_factor(data, ax=None, ts_range=None):
    plot_l(data, ax, ts_range, 'Pf', 'Power Factor')


def plot_l(data, ax, ts_range, field, title):
    data = prepare_plot(data, ax, ts_range)
    ax.plot(data['Datetimes'], data[field], 'c')
    ax.set_title(title)


def plot_hf_noise(data, ax=None, ts_range=None):
    data = prepare_plot(data, ax, ts_range)

    x = data['HF_Datetimes']
    ax.imshow(np.transpose(data['HF']), aspect='auto', origin='lower',
              extent=[x[0], x[-1], 0, 1e6])

    freqs = np.linspace(0, 1e6, 6)
    ax.set_yticks(freqs)
    ax.set_yticklabels([f"{int(f / 1e3)}K" for f in freqs])
    ax.set_title('High Frequency Noise')
    ax.set_ylabel('Frequency KHz')


def prepare_plot(data, ax, ts_range):
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax.grid(True)

    if ts_range is not None:
        trunc_data(data, ts_range)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    return data


def add_device_tags(data, ax):
    y_steps = np.arange(0.2, 0.8, 0.2)
    y_idx = 0
    for tag in data['TaggingInfo']:
        add_device_tag(ax, tag, y_steps[y_idx])
        y_idx = (y_idx + 1) % len(y_steps)


def add_device_tag(ax, tag, y_step):
    name = tag[1]
    add_line(ax, f'ON-{name}', tag[2], 'g', y_step)
    add_line(ax, f'OFF-{name}', tag[3], 'r', y_step + 0.1)


def add_line(ax, name, time, color, y_step):
    xlims, ylims = ax.get_xlim(), ax.get_ylim()
    line_x = datetime.fromtimestamp(time)
    text_x = datetime.fromtimestamp(time + (xlims[1] - xlims[0]) * 500)
    text_y = (ylims[1] - ylims[0]) * y_step + ylims[0]
    ax.axvline(x=line_x, color=color, linestyle='--')
    ax.text(text_x, text_y, name, fontsize='xx-small')


def trunc_data(data, ts_range):
    data = copy.deepcopy(data)
    trunc_range(data, ts_range, 'TimeTicks', l_keys)
    trunc_range(data, ts_range, 'HF_TimeTicks', hf_keys)
    return data


def trunc_range(data, ts_range, time_key, keys):
    start = closest_idx(data[time_key], ts_range[0])
    stop = closest_idx(data[time_key], ts_range[1])
    idx_offset = int((stop - start) * 0.05)
    start = max(start - idx_offset, 0)
    stop = min(stop + idx_offset, len(data[time_key]) - 1)
    for key in keys:
        data[key] = data[key][start:stop]


def closest_idx(arr, el):
    return min(range(len(arr)), key=lambda i: abs(arr[i] - el))
