import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime
import numpy as np
import copy


l1_keys = ['L1_TimeTicks', 'L1_Real', 'L1_App', 'L1_Imag', 'L1_Pf']
l2_keys = ['L2_TimeTicks', 'L2_Real', 'L2_App', 'L2_Imag', 'L2_Pf']
hf_keys = ['HF_TimeTicks', 'HF']


def to_datetimes(data, keys):
    for key in keys:
        data[key] = np.array([datetime.fromtimestamp(t) for t in data[key]])


def closest_idx(arr, el):
    return min(range(len(arr)), key=lambda i: abs(arr[i]-el))


def trunc_range(data, ts_range, time_key, keys):
    start = closest_idx(data[time_key], ts_range[0])
    stop = closest_idx(data[time_key], ts_range[1])
    idx_offset = int((stop - start) * 0.05)
    start = max(start - idx_offset, 0)
    stop = min(stop + idx_offset, len(data[time_key]) - 1)
    for key in keys:
        data[key] = data[key][start:stop]


def plot_real_power(ax, data):
    ax.plot(data['L1_TimeTicks'], data['L1_Real'])
    ax.plot(data['L2_TimeTicks'], data['L2_Real'], 'c')
    ax.set_title('Real Power (W) and ON/OFF Device Category IDs')


def plot_reactive_power(ax, data):
    ax.plot(data['L1_TimeTicks'], data['L1_Imag'])
    ax.plot(data['L2_TimeTicks'], data['L2_Imag'], 'c')
    ax.set_title('Imaginary/Reactive power (VAR)')


def plot_power_factor(ax, data):
    ax.plot(data['L1_TimeTicks'], data['L1_Pf'])
    ax.plot(data['L2_TimeTicks'], data['L2_Pf'], 'c')
    ax.set_title('Power Factor')


def plot_hf_noise(ax, data):
    time_ticks = np.transpose(data['HF_TimeTicks'])
    ax.imshow(np.transpose(data['HF']), aspect='auto', origin='lower',
              extent=[time_ticks[0], time_ticks[-1], 0, 1e6])

    freqs = np.linspace(0, 1e6, 6)
    ax.set_yticks(freqs)
    ax.set_yticklabels([f"{int(f/1e3)}K" for f in freqs])
    ax.set_ylabel('Frequency KHz')
    ax.set_title('High Frequency Noise')


def add_line(ax, name, time, color, y_step):
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    x_dim = xlims[1] - xlims[0]
    y_dim = ylims[1] - ylims[0]
    line_x = datetime.fromtimestamp(time)
    text_x = datetime.fromtimestamp(time + x_dim * 500)
    text_y = y_dim * y_step + ylims[0]
    ax.axvline(x=line_x, color=color, linestyle='--')
    ax.text(text_x, text_y, name, fontsize='xx-small')


def add_device_tag(ax, tag, y_step):
    name = tag[1]
    add_line(ax, f'ON-{name}', tag[2], 'g', y_step)
    add_line(ax, f'OFF-{name}', tag[3], 'r', y_step + 0.1)


def plot_data(data, has_labels=False, ts_range=None):
    _, axs = plt.subplots(4, 1, figsize=(10, 8))
    data = copy.deepcopy(data)

    # Truncate data by range
    if ts_range != None:
        trunc_range(data, ts_range, 'L1_TimeTicks', l1_keys)
        trunc_range(data, ts_range, 'L2_TimeTicks', l2_keys)
        trunc_range(data, ts_range, 'HF_TimeTicks', hf_keys)

    # Convert time ticks to the datetime format for better visualization
    to_datetimes(data, ['L1_TimeTicks', 'L2_TimeTicks', 'HF_TimeTicks'])

    plot_real_power(axs[0], data)
    plot_reactive_power(axs[1], data)
    plot_power_factor(axs[2], data)
    plot_hf_noise(axs[3], data)

    for ax in axs:
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        ax.grid(True)

        if has_labels and 'TaggingInfo' in data:
            y_steps = np.arange(0.2, 0.8, 0.2)
            y_idx = 0
            for tag in data['TaggingInfo']:
                add_device_tag(ax, tag, y_steps[y_idx])
                y_idx = (y_idx + 1) % len(y_steps)

    plt.tight_layout(h_pad=2)
    plt.show()


def plot_data_tagged(data, has_labels=True):
    if 'TaggingInfo' not in data:
        print('There is no tagged data')
        return
    
    min_ts = min(x[2] for x in data['TaggingInfo'])
    max_ts = max(x[3] for x in data['TaggingInfo'])
    plot_data(data, ts_range=(min_ts, max_ts), has_labels=has_labels)