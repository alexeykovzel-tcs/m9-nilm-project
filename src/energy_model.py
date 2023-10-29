from src.meter_data import MeterData
from src.event_detector import detect_cycles
from src.appliance import Appliance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


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
        plt.stackplot(times, powers, labels, alpha=0.6)
        plt.legend(loc='upper left')
        plt.xlabel('Time')
        plt.ylabel('Energy Consumption (Wh)')
        plt.title('Energy Disaggregation')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # TODO: Handle when power < 0 at some timestamps.
    def disaggregate(self, data: MeterData):
        times = data.total_power.times

        # assign appliances to power cycles
        appliance_cycles = self.detect_appliance_cycles(data)
        if not appliance_cycles:
            print('no cycles detected :/')
            return None

        powers = [appliance.base_power(len(times), cycle) for cycle, appliance in appliance_cycles]
        labels = [appliance.label for _, appliance in appliance_cycles]

        # handle remaining power
        powers.append(data.total_power.real() - sum(powers))
        labels.append('Other')

        return times, powers, labels

    def detect_appliance_cycles(self, data: MeterData):
        # detect cycles for each power phase
        cycles = [power.truncate(cycle) for power in data.powers for cycle in detect_cycles(power)]
        if not cycles: return None

        # extract cycle & appliance features
        cycle_features = [cycle.features() for cycle in cycles]
        appliance_features = [appliance.features() for appliance in self.appliances.values()]

        # normalize extracted features
        scaler = StandardScaler()
        scaler.fit_transform(cycle_features + appliance_features)
        appliance_features = scaler.transform(appliance_features)
        cycle_features = scaler.transform(cycle_features)

        # guess an appliance for each detected cycle
        guesses = [_similar_ints_idx(f, appliance_features) for f in cycle_features]
        return [(cycle, self.appliances[idx]) for cycle, (idx, ci) in zip(cycles, guesses) if ci > 0.2]


def _similar_ints_idx(f: [int], vs: [[int]]):
    compares = [(idx, _compare_ints(f, v)) for idx, v in enumerate(vs)]
    return max(compares, key=lambda v: v[1])


def _compare_ints(v1: [int], v2: [int]):
    v1, v2 = v1.reshape(1, -1), v2.reshape(1, -1)
    return cosine_similarity(v1, v2)[0][0]
