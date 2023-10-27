from src.meter_data import MeterData
from src.appliance_profiler import profile_appliances
from src.cycle_detector import detect_cycles
from src.feature_extractor import extract_power_features
from src.meter_data import Power
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


class EnergyModel:
    def __init__(self):
        self.appliances = []
        self.feature_map = {}

    def train(self, data_per_day: [MeterData]):
        # profile and add tagged appliances in the meter data to existing appliances
        self.appliances = [profile_appliances(data) for data in data_per_day] + self.appliances

        # group identical appliances by ID
        unique_idx = {x.idx for x in self.appliances}
        appliances_by_id = [[x for x in self.appliances if x.idx == idx] for idx in unique_idx]

        # create a scaler to normalize features
        scaler = StandardScaler()
        scaler.fit([x.features for x in self.appliances])

        # combine appliances by index with averaged and normalized features
        feature_data = [sum(x.features for x in appliances) / len(appliances)
                        for appliances in appliances_by_id]

        feature_data = scaler.transform(feature_data)
        self.feature_map = {idx: features for idx, features in zip(unique_idx, feature_data)}

    def disaggregate_and_plot(self, data: MeterData):
        times, powers, labels = self.disaggregate(data)

        plt.figure(figsize=(10, 4))
        plt.stackplot(times, powers, labels, alpha=0.6)
        plt.legend(loc='upper left')
        plt.xlabel('Time')
        plt.ylabel('Energy Consumption (Wh)')
        plt.title('Energy Disaggregation')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def disaggregate(self, data: MeterData):
        total_power = data.total_power()
        times = total_power.times

        # disaggregate power for each cycle of each power phase
        powers, labels = [], []
        for l in [data.l1, data.l2]:
            for cycle in detect_cycles(l):
                appliance = self.match_appliance(l.truncate(cycle))
                if appliance is not None:
                    powers.append(appliance.base_power(cycle, times))
                    labels.append(appliance.label)

        # calculate the remaining power that couldn't be assigned an appliance
        powers.append(total_power.real() - sum(powers))
        labels.append('Other')

        # TODO: Handle when power < 0 at some timestamps.
        return times, powers, labels
    #
    # def match_appliance(self, power: Power):
    #     features = extract_power_features(power)
    #
    #     for idx, idx_features in self.feature_map.items():
    #         eucl_distance = np.linalg.norm(features - idx_features)
    #         eucl_similarity = 1 / (1 + eucl_distance)
    #
    #     matches = [(x, x.match(data, cycle)) for x in self.appliances]
    #
    #     appliance, match = max(matches, key=lambda x: x[1])
    #     return None if match < 0.2 else appliance
