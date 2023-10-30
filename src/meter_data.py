from src.signals import *
from src.cycle_detector import detect_cycles


"""

This file contains the MeterData class, that contains:

- powers of 2 phases (L1 & L2)
- high frequency noise
- tagging info (for training)

"""


class Tag:
    def __init__(self, idx, label, on, off):
        self.idx, self.label = idx, label
        self.on, self.off = on, off

    def shift_cycle(self):
        """ shifts the tag cycle, as appliances don't turn ON and OFF immediately. """
        size = self.off - self.on
        new_on = self.on + size * 0.2
        new_off = self.off + size * 0.4
        return Tag(self.idx, self.label, new_on, new_off)

    def find_power_cycle(self, power_cycles):
        """ finds a power cycle that matches the tag cycle. """
        power, cycle = max(power_cycles, key=lambda x: self._overlap_score(x[1][0], x[1][1]))
        return power.truncate(cycle)

    def _overlap_score(self, on, off):
        """ calculates how the given cycle overlaps with the tag cycle. """
        overlap_len = max(0, min(self.off, off) - max(self.on, on))
        combined_len = (self.off - self.on) + (off - on) - overlap_len
        return overlap_len / combined_len if combined_len > 0 else 0


class MeterData:
    def __init__(self, l1: Power, l2: Power, hf: FreqNoise, tags: [Tag]):
        self.hf, self.tags = hf, tags
        self.l1, self.l2 = l1, l2
        self.total_power = l1 + l2
        self.power_cycles = [
            (power, cycle) for power in [l1, l2]
            for cycle in detect_cycles(power)
        ]

    def tagged_cycles(self):
        return [] if not self.power_cycles else [
            (tag, cycle) for tag in [tag.shift_cycle() for tag in self.tags]
            for cycle in [tag.find_power_cycle(self.power_cycles)] if cycle is not None
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
