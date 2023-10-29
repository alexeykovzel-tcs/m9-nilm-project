import numpy as np


class Appliance:
    def __init__(self, idx, label):
        self.idx, self.label, self.cycles = idx, label, []

    def features(self):
        return np.mean([cycle.features for cycle in self.cycles], axis=0)

    def base_power(self, length, cycle):
        result = np.zeros(length)

        for i in range(cycle[0], cycle[1]):
            result[i] = 20

        return result
