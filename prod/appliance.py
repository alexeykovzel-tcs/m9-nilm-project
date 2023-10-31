from prod.feature_extractor import power_features
import numpy as np


class Appliance:
    def __init__(self, idx, label):
        self.idx, self.label, self.cycles = idx, label, []

    def features(self):
        cycle_features = [power_features(cycle) for cycle in self.cycles]
        return np.mean(cycle_features, axis=0)

    @staticmethod
    def base_power(times, powers):
        result = np.zeros(len(times))

        for power in powers:
            start, stop = power.range
            length = len(power.times)

            times_start = np.where(times == start)[0][0]
            result[times_start:times_start + length] = power.real().vals

        return result
