from src.signals import *
from src.event_detector import detect_cycles
from matplotlib.dates import DateFormatter
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os


class Tag:
    def __init__(self, idx, label, on, off):
        self.idx, self.label = idx, label
        self.on, self.off = on, off

    def find_cycle(self, powers: [Power]):
        """
        finds a power cycle that matches the tag cycle.
        (with the highest overlapping score)
        """
        all_cycles = [(power, cycle) for power in powers for cycle in detect_cycles(power)]
        if not all_cycles: return None
        power, cycle = max(all_cycles, key=lambda _, c: self.overlap_score(c))
        return power.truncate(cycle)

    def overlap_score(self, cycle):
        """
        calculate how the given cycle overlaps with the tag cycle, where:
        0 - don't overlap, 1 - fully overlap.
        """
        on, off = cycle
        overlap_len = max(0, min(self.off, off) - max(self.on, on))
        combined_len = (self.off - self.on) + (off - on) - overlap_len
        return overlap_len / combined_len if combined_len > 0 else 0

    def adjust_times(self):
        """
        shifts the tag cycle, as appliances have transient states
        and don't turn ON and OFF immediately.
        """
        size = self.off - self.on
        new_on = self.on + size * 0.2
        new_off = self.off + size * 0.5
        return Tag(self.idx, self.label, new_on, new_off)


class MeterData:
    def __init__(self, l1: Power, l2: Power, hf: FreqNoise, tags: [Tag]):
        self.hf, self.tags = hf, tags
        self.l1, self.l2 = l1, l2
        self.total_power = l1 + l2
        self.powers = [l1, l2]

    def tagged_cycles(self):
        adjusted_tags = [tag.adjust_times() for tag in self.tags]
        return [
            (tag, cycle) for tag in adjusted_tags
            for cycle in [tag.find_cycle(self.powers)] if cycle is not None
        ]

    def truncate_tagged(self):
        start = min(x.on for x in self.tags)
        stop = max(x.off for x in self.tags)
        return self.truncate((start, stop))

    def truncate(self, cycle=None):
        return MeterData(
            self.l1.truncate(cycle),
            self.l2.truncate(cycle),
            self.hf.truncate(cycle),
            self.tags
        )

    def plot(self, show_tags=True):
        """
        plots the real power, reactive power, power factor of the 1st component, and high frequency noise.
        if show_tags is true, then ON and OFF labels of appliances are also added.
        """
        _, axs = plt.subplots(4, 1, figsize=(10, 8))

        self._plot_power(axs[0], lambda p: p.real(), 'Real Power (W)')
        self._plot_power(axs[1], lambda p: p.reactive(), 'Reactive power (VAR)')
        self._plot_power(axs[2], lambda p: p.factor(), 'Power factor')
        self._plot_hf(axs[3])

        for ax in axs:
            if show_tags and self.tags is not None:
                self._add_device_tags(ax)

        plt.tight_layout(h_pad=2)
        plt.show()

    def _plot_hf(self, ax):
        times = self.hf.format_times()
        ax.imshow(np.transpose(self.hf.vals),
                  aspect='auto', origin='lower',
                  extent=[times[0], times[-1], 0, 1e6])

        freqs = np.linspace(0, 1e6, 6)
        ax.set_yticks(freqs)
        ax.set_yticklabels([f"{int(f / 1e3)}K" for f in freqs])
        ax.set_ylabel('Frequency KHz')

        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        ax.set_title('High Frequency Noise')
        ax.grid(True)

    def _plot_power(self, ax, consumer, title):
        ax.plot(self.l1.format_times(), consumer(self.l1), 'c')
        ax.plot(self.l2.format_times(), consumer(self.l2))

        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        ax.set_title(title)
        ax.grid(True)

    def _add_device_tags(self, ax):
        y_steps, y_idx = np.arange(0.2, 0.8, 0.2), 0

        def _add_line(name, time, color, y_step):
            xlims, ylims = ax.get_xlim(), ax.get_ylim()
            line_x = datetime.fromtimestamp(time)
            text_x = datetime.fromtimestamp(time + (xlims[1] - xlims[0]) * 500)
            text_y = (ylims[1] - ylims[0]) * y_step + ylims[0]
            ax.axvline(x=line_x, color=color, linestyle='--')
            ax.text(text_x, text_y, name, fontsize='xx-small')

        def _add_device_line(name, on_time, off_time, y_step):
            _add_line(f'ON-{name}', on_time, 'g', y_step)
            _add_line(f'OFF-{name}', off_time, 'r', y_step + 0.1)

        for tag in self.tags:
            _add_device_line(tag.label, tag.on, tag.off, y_steps[y_idx])
            y_idx = (y_idx + 1) % len(y_steps)


train_files = {}
test_files = {}

data_dir = os.path.abspath('data')

for h in os.listdir(data_dir):
    files = os.listdir(f'{data_dir}/{h}')
    train_files[h] = [f for f in files if f.startswith('Tagged')]
    test_files[h] = [f for f in files if f.startswith('Testing')]


def load_train(h_dir, idx) -> MeterData:
    """ loads a train file by index """
    return load(h_dir, train_files[h_dir][idx])


def load_test(h_dir, idx) -> MeterData:
    """ loads a test file by index """
    return load(h_dir, test_files[h_dir][idx])


def load(h_dir, name) -> MeterData:
    """ loads a file by name """
    data = loadmat(f'{data_dir}/{h_dir}/{name}')
    return _process_raw_data(data)


def _process_raw_data(data) -> MeterData:
    buffer = data['Buffer']

    # 1st phase power
    l1_vals = buffer['LF1V'][0][0] * np.conj(buffer['LF1I'][0][0])
    l1_times = buffer['TimeTicks1'][0][0][:, 0]
    l1 = Power(l1_vals, l1_times)

    # 2nd phase power
    l2_vals = buffer['LF2V'][0][0] * np.conj(buffer['LF2I'][0][0])
    l2_times = buffer['TimeTicks2'][0][0][:, 0]
    l2 = Power(l2_vals, l2_times)

    # align L1 & L2 times
    if len(l1.vals) < len(l2.vals):
        l1 = l1.align_times(l2.times)
    else:
        l2 = l2.align_times(l1.times)

    # high-frequency noise
    hf_vals = np.transpose(buffer['HF'][0][0])
    hf_times = buffer['TimeTicksHF'][0][0][:, 0]
    hf = FreqNoise(hf_vals, hf_times)

    # appliance tags
    tags = None
    if 'TaggingInfo' in buffer.dtype.names:
        tags = [[x[0][0] for x in y] for y in buffer['TaggingInfo'][0][0]]
        tags = [Tag(x[0], x[1][0], x[2], x[3]) for x in tags]

    return MeterData(l1, l2, hf, tags)
