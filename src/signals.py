from datetime import datetime
from src.feature_extractor import stats
import numpy as np


class Signal:
    def __init__(self, vals, times):
        self.vals, self.times = vals, times

    def __add__(self, other):
        if len(self.vals) != len(self.vals):
            raise Exception('signal lengths should be equal')

        return self.__class__(self.vals + other.vals, self.times)

    def format_times(self):
        return [datetime.fromtimestamp(t) for t in self.times]

    def truncate(self, cycle):
        start, stop = cycle
        i1 = self.time_idx(start) if (start is not None) else 0
        i2 = self.time_idx(stop) if (stop is not None) else len(self.times) - 1
        return self.__class__(self.vals[i1:i2], self.times[i1:i2])

    def time_idx(self, time):
        return min(range(len(self.times)), key=lambda i: abs(self.times[i] - time))

    def align_times(self, times):
        result = []
        i, max_i = 0, len(self.vals) - 1

        for t in times:
            if self.times[i] < t and i < max_i: i += 1
            result.append(self.vals[i])

        return self.__class__(np.array(result), times)

    def len(self):
        return self.times[-1] - self.times[0]


class Power(Signal):
    def __init__(self, vals, times):
        super().__init__(vals, times)
        self.net = vals.sum(axis=1)

    def factor(self, axis=0):
        return np.cos(np.angle(self.vals[:, axis]))

    def real(self):
        return np.real(self.net)

    def reactive(self):
        return np.imag(self.net)

    def apparent(self):
        return np.abs(self.net)

    def features(self):
        return np.concatenate([
            stats(self.real()),
            stats(self.reactive()),
            stats(self.factor()),
            [self.len()]
        ])


class FreqNoise(Signal):
    def __init__(self, vals, times):
        super().__init__(vals, times)

    def avg(self):
        vals = [np.average(val) for val in self.vals]
        return Signal(vals, self.times)
