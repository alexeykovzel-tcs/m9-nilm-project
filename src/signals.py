import numpy as np
from datetime import datetime


class Signal:
    def __init__(self, vals, times):
        self.vals, self.times = vals, times

    def datetimes(self):
        return [datetime.fromtimestamp(t) for t in self.times]

    def truncate(self, start=None, stop=None):
        i1 = self.time_idx(start) if (start is not None) else 0
        i2 = self.time_idx(stop) if (stop is not None) else len(self.times) - 1
        self.vals = self.vals[i1:i2]
        self.times = self.times[i1:i2]
        return self

    def time_idx(self, time):
        return min(range(len(self.times)),
                   key=lambda i: abs(self.times[i] - time))


class Power(Signal):
    def __init__(self, vals, times):
        super().__init__(vals, times)

    def real(self):
        return np.real(self.vals)

    def reactive(self):
        return np.imag(self.vals)

    def apparent(self):
        return np.abs(self.vals)

    def copy(self):
        return Power(self.vals.copy(), self.times.copy())


class ComplexPower(Signal):
    def __init__(self, vals, times):
        super().__init__(vals, times)

    def net(self):
        return Power(self.vals.sum(axis=1), self.times)

    def factor(self):
        return np.cos(np.angle(self.vals[:, 0]))

    def copy(self):
        return ComplexPower(self.vals.copy(), self.times.copy())


class FreqNoise(Signal):
    def __init__(self, vals, times):
        super().__init__(vals, times)

    def avg(self) -> Signal:
        vals = [np.average(val) for val in self.vals]
        return Signal(vals, self.times)

    def copy(self):
        return FreqNoise(self.vals.copy(), self.times.copy())
