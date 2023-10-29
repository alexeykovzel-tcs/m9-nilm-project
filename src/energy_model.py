from src.meter_data import MeterData
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


class Appliance:
    def __init__(self, idx, label):
        self.idx, self.label, self.cycles = idx, label, []

    def features(self):
        return np.mean([cycle.features() for cycle in self.cycles], axis=0)


class EnergyModel:
    def __init__(self):
        self.appliances = {}

    def add_data(self, data_per_day: [MeterData]):
        for data in data_per_day:
            for tag, cycle in data.tagged_cycles():
                appliance = self.appliances.setdefault(tag.idx, Appliance(tag.idx, tag.label))
                appliance.cycles.append(cycle)

    def disaggregate_and_plot(self, data: MeterData):
        result = self.disaggregate(data)
        if not result: return None

        times, powers, labels = result
        plt.figure(figsize=(10, 4))
        plt.stackplot(times, powers, labels=labels, alpha=0.6)
        plt.legend(loc='upper left')
        plt.xlabel('Time')
        plt.ylabel('Energy Consumption (Wh)')
        plt.title('Energy Disaggregation')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def disaggregate(self, data: MeterData):
        times = data.total_power.times

        # assign appliances to power cycles
        guesses = self._guess_appliances(data)
        if not guesses:
            print('no cycles detected.')
            return None

        powers = [_base_power(times, cycles) for appliance, cycles in guesses.items()]
        labels = [appliance.label for appliance in guesses.keys()]

        # handle remaining power
        other = data.total_power.real() - sum(powers)
        powers = [other] + powers
        labels = ['Other'] + labels

        # handle when power < 0 at some timestamps
        for i, val in enumerate(other):
            if val < 0:
                other[i] = 0
                # TODO: Update powers.

        return times, powers, labels

    def _guess_appliances(self, data: MeterData):
        if not data.cycles: return None
        appliances = list(self.appliances.values())

        # extract cycle & appliance features
        f_cycles = [power.truncate(cycle).features() for power, cycle in data.cycles]
        f_appliances = [appliance.features() for appliance in appliances]

        # normalize extracted features
        scaler = StandardScaler()
        scaler.fit_transform(f_cycles + f_appliances)
        f_cycles = scaler.transform(f_cycles)
        f_appliances = scaler.transform(f_appliances)

        # guess appliances per cycle (ignore if low CI)
        guesses = [(cycle, _similar_ints_idx(f, f_appliances)) for cycle, f in zip(data.cycles, f_cycles)]
        guesses = [(cycle, appliances[idx]) for cycle, (idx, ci) in guesses if ci >= 0.0]

        # group guesses with the same appliance
        grouped_guesses = defaultdict(list)
        for cycle, appliance in guesses:
            grouped_guesses[appliance].append(cycle)

        return grouped_guesses


def _base_power(times, power_cycles):
    result = np.zeros(len(times))

    for power, cycle in power_cycles:
        times_start = np.where(times == cycle[0])[0][0]
        power_start = np.where(power.times == cycle[0])[0][0]
        power_stop = np.where(power.times == cycle[1])[0][0]

        power_len = power_stop - power_start
        result[times_start:times_start + power_len] \
            = power.real()[power_start:power_start + power_len]

    return result


def _similar_ints_idx(f: [int], vs: [[int]]):
    compares = [(idx, _compare_ints(f, v)) for idx, v in enumerate(vs)]
    return max(compares, key=lambda v: v[1])


def _compare_ints(v1: [int], v2: [int]):
    v1, v2 = v1.reshape(1, -1), v2.reshape(1, -1)
    return cosine_similarity(v1, v2)[0][0]
