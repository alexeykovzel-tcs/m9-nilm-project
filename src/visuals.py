from src.meter_data import MeterData
from matplotlib.dates import DateFormatter
from datetime import datetime
from src.signals import Signal
import matplotlib.pyplot as plt
import numpy as np


def plot_powers_pie(powers, labels):
    sizes = [sum(power) for power in powers]
    plt.pie(sizes, labels=labels)
    plt.show()


def plot_powers_stack(times, powers, labels):
    formatted_times = [datetime.fromtimestamp(time) for time in times]
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.stackplot(formatted_times, powers, labels=labels, alpha=0.6)
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax.grid(True)

    ax.set_xlabel('Time')
    ax.set_ylabel('Energy Consumption (Wh)')
    ax.set_title('Energy Disaggregation')

    plt.tight_layout()
    plt.show()


def plot_signal(signal: Signal):
    plt.plot(signal.format_times(), signal.vals)
    plt.figure(figsize=(12, 3))
    plt.show()


def plot(data: MeterData, show_tags=True):
    _, axs = plt.subplots(3, 1, figsize=(10, 8))

    _plot_power(data, axs[0], lambda p: p.real(), 'Real Power (W)')
    _plot_power(data, axs[1], lambda p: p.reactive(), 'Reactive power (VAR)')
    _plot_power(data, axs[2], lambda p: p.factor(), 'Power factor')
    # _plot_hf(data, axs[3])

    for ax in axs:
        if show_tags and data.tags is not None:
            _add_device_tags(data, ax)

    plt.tight_layout(h_pad=2)
    plt.show()


def _plot_hf(data: MeterData, ax):
    times = data.hf.format_times()
    ax.imshow(np.transpose(data.hf.vals),
              aspect='auto', origin='lower',
              extent=[times[0], times[-1], 0, 1e6])

    freqs = np.linspace(0, 1e6, 6)
    ax.set_yticks(freqs)
    ax.set_yticklabels([f"{int(f / 1e3)}K" for f in freqs])
    ax.set_ylabel('Frequency KHz')

    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax.set_title('High Frequency Noise')
    ax.grid(True)


def _plot_power(data: MeterData, ax, consumer, title):
    ax.plot(data.l1.format_times(), consumer(data.l1), 'c')
    ax.plot(data.l2.format_times(), consumer(data.l2))

    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax.set_title(title)
    ax.grid(True)


def _add_device_tags(data: MeterData, ax):
    y_steps, y_idx = np.arange(0.2, 0.8, 0.2), 0

    for tag in data.tags:
        _add_device_line(ax, tag.label, tag.on, tag.off, y_steps[y_idx])
        y_idx = (y_idx + 1) % len(y_steps)


def _add_line(ax, name, time, color, y_step):
    xlims, ylims = ax.get_xlim(), ax.get_ylim()
    line_x = datetime.fromtimestamp(time)
    text_x = datetime.fromtimestamp(time + (xlims[1] - xlims[0]) * 500)
    text_y = (ylims[1] - ylims[0]) * y_step + ylims[0]
    ax.axvline(x=line_x, color=color, linestyle='--')
    ax.text(text_x, text_y, name, fontsize='xx-small')


def _add_device_line(ax, name, on_time, off_time, y_step):
    _add_line(ax, f'ON-{name}', on_time, 'g', y_step)
    _add_line(ax, f'OFF-{name}', off_time, 'r', y_step + 0.1)
