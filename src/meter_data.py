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

    def find_cycle(self, cycles):
        """
        Finds a power cycle that matches the tag cycle.
        (with the highest overlapping score)
        """
        if not cycles: return None
        power, cycle = max(cycles, key=lambda x: self.overlap_score(x[1]))
        return power.truncate(cycle)

    def overlap_score(self, cycle):
        """
        Calculate how the given cycle overlaps with the tag cycle, where:
        0 - don't overlap, 1 - fully overlap.
        """
        on, off = cycle
        overlap_len = max(0, min(self.off, off) - max(self.on, on))
        combined_len = (self.off - self.on) + (off - on) - overlap_len
        return overlap_len / combined_len if combined_len > 0 else 0

    def adjust_times(self):
        """
        Shifts the tag cycle, as appliances don't turn ON and OFF immediately.
        """
        size = self.off - self.on
        new_on = self.on + size * 0.2
        new_off = self.off + size * 0.5
        return Tag(self.idx, self.label, new_on, new_off)


class MeterData:
    def __init__(self, l1: Power, l2: Power, hf: FreqNoise, tags: [Tag]):
        self.l1, self.l2, self.total_power = l1, l2, l1 + l2
        self.hf, self.tags = hf, tags
        self.cycles = [(power, cycle) for power in [l1, l2] for cycle in detect_cycles(power)]

    def tagged_cycles(self):
        adjusted_tags = [tag.adjust_times() for tag in self.tags]
        return [
            (tag, cycle) for tag in adjusted_tags
            for cycle in [tag.find_cycle(self.cycles)] if cycle is not None
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
