from src.meter_data import MeterData
from src.appliance_profiler import profile_appliances
from src.cycle_detector import detect_cycles
from src.appliance import Appliance
import matplotlib.pyplot as plt


class EnergyModel:
    def __init__(self):
        self.appliances = []

    def train(self, data_per_day: [MeterData]):
        appliances = [profile_appliances(data) for data in data_per_day] + self.appliances
        appliances = [[x for x in appliances if x.idx == idx] for idx in {x.idx for x in appliances}]
        self.appliances = [Appliance.merge(x) for x in appliances]

    def match_appliance(self, data: MeterData, cycle):
        matches = [(x, x.match(data, cycle)) for x in self.appliances]
        appliance, match = max(matches, key=lambda x: x[1])
        return None if match < 0.2 else appliance

    def disaggregate(self, data: MeterData):
        total_power = data.total_power()
        times = total_power.times

        powers = []
        appliances = []
        for l in [data.l1, data.l2]:
            for cycle in detect_cycles(l, data.hf):
                appliance = self.match_appliance(data, cycle)
                powers.append(appliance.base_power(cycle, times))
                appliances.append(appliance)

        return times, powers, appliances

    def disaggregate_and_plot(self, data: MeterData):
        times, powers, appliances = self.disaggregate(data)
        labels = [x.label for x in appliances]

        plt.figure(figsize=(10, 4))
        plt.stackplot(times, powers, labels, alpha=0.6)
        plt.legend(loc='upper left')
        plt.xlabel('Time')
        plt.ylabel('Energy Consumption (Wh)')
        plt.title('Energy Disaggregation')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
