import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime
import numpy as np


def plot_data(data, has_labels=False, start=None, stop=None):
    _, axs = plt.subplots(4, 1, figsize=(10, 8))

    if start is not None or stop is not None:
        data = data.truncate(data, start, stop)

    times = np.array([datetime.fromtimestamp(t) for t in data.times])

    _plot_line(axs[0], times, data.real_power(), 'Real Power (W)')
    _plot_line(axs[1], times, data.reactive_power(), 'Reactive power (VAR)')
    _plot_line(axs[2], times, data.pf, 'Power Factor')
    _plot_hf(axs[3], times, data.hf)

    for ax in axs:
        if has_labels and data.tags is not None:
            _add_device_tags(data, ax)

    plt.tight_layout(h_pad=2)
    plt.show()


def plot_tagged_data(data, has_labels=True):
    if data.tags is None:
        print('there is no tagged belkin...')
    else:
        start = min(x[2] for x in data.tags)
        stop = max(x[3] for x in data.tags)
        plot_data(data, has_labels=has_labels, start=start, stop=stop)


def _plot_hf(ax, times, values):
    _prepare_plot(ax)

    ax.imshow(np.transpose(values),
              aspect='auto', origin='lower',
              extent=[times[0], times[-1], 0, 1e6])

    freqs = np.linspace(0, 1e6, 6)
    ax.set_yticks(freqs)
    ax.set_yticklabels([f"{int(f / 1e3)}K" for f in freqs])
    ax.set_title('High Frequency Noise')
    ax.set_ylabel('Frequency KHz')


def _plot_line(ax, times, values, title):
    _prepare_plot(ax)
    ax.plot(times, values, 'c')
    ax.set_title(title)


def _prepare_plot(ax):
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax.grid(True)


def _add_device_tags(data, ax):
    y_steps = np.arange(0.2, 0.8, 0.2)
    y_idx = 0
    for tag in data.tags:
        _add_device_tag(ax, tag, y_steps[y_idx])
        y_idx = (y_idx + 1) % len(y_steps)


def _add_device_tag(ax, tag, y_step):
    name = tag[1]
    _add_line(ax, f'ON-{name}', tag[2], 'g', y_step)
    _add_line(ax, f'OFF-{name}', tag[3], 'r', y_step + 0.1)


def _add_line(ax, name, time, color, y_step):
    xlims, ylims = ax.get_xlim(), ax.get_ylim()
    line_x = datetime.fromtimestamp(time)
    text_x = datetime.fromtimestamp(time + (xlims[1] - xlims[0]) * 500)
    text_y = (ylims[1] - ylims[0]) * y_step + ylims[0]
    ax.axvline(x=line_x, color=color, linestyle='--')
    ax.text(text_x, text_y, name, fontsize='xx-small')
