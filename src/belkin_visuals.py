import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime
import numpy as np


def plot(data, show_tags=False):
    _, axs = plt.subplots(4, 1, figsize=(10, 8))

    plot_real_power(data, axs[0])
    plot_reactive_power(data, axs[1])
    plot_power_factor(data, axs[2])
    plot_hf(data, axs[3])

    for ax in axs:
        if show_tags and data.tags is not None:
            _add_device_tags(data, ax)

    plt.tight_layout(h_pad=2)
    plt.show()


def plot_hf(data, ax=None):
    ax = _prepare_plot(ax)

    times = data.hf.datetimes()
    ax.imshow(np.transpose(data.hf.vals),
              aspect='auto', origin='lower',
              extent=[times[0], times[-1], 0, 1e6])

    freqs = np.linspace(0, 1e6, 6)
    ax.set_yticks(freqs)
    ax.set_yticklabels([f"{int(f / 1e3)}K" for f in freqs])
    ax.set_title('High Frequency Noise')
    ax.set_ylabel('Frequency KHz')


def plot_real_power(data, ax=None):
    _plot_power(data, ax, lambda p: p.net().real(), 'Real Power (W)')


def plot_reactive_power(data, ax=None):
    _plot_power(data, ax, lambda p: p.net().reactive(), 'Reactive power (VAR)')


def plot_power_factor(data, ax=None):
    _plot_power(data, ax, lambda p: p.factor(), 'Power factor')


def _plot_power(data, ax, consumer, title):
    ax = _prepare_plot(ax)
    ax.plot(data.l1.datetimes(), consumer(data.l1), 'c')
    ax.plot(data.l2.datetimes(), consumer(data.l2))
    ax.set_title(title)


def _add_device_tags(data, ax):
    y_steps, y_idx = np.arange(0.2, 0.8, 0.2), 0
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


def _prepare_plot(ax):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax.grid(True)
    return ax
