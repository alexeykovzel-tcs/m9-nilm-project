import numpy as np
from src.meter_data import MeterData


class Appliance:
    def __init__(self, label, idx):
        self.label, self.idx = label, idx

    def base_power(self, cycle, times):
        # TODO: Implement this.
        return np.zeros(len(times))

    def match(self, data: MeterData, cycle):
        # TODO: Implement this.
        return 0.2

    @staticmethod
    def profile(data: MeterData, tag, cycle, phase):
        # TODO: Implement this.
        return Appliance('test', -1)

    @staticmethod
    def merge(appliances):
        # TODO: Implement this.
        return appliances[0]
