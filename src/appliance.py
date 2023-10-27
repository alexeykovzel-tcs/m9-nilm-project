import numpy as np
from src.meter_data import MeterData


class Appliance:
    def __init__(self, idx, label, features):
        self.idx, self.label, self.features = idx, label, features

    def base_power(self, cycle, times):
        # TODO: Implement this.
        return np.zeros(len(times))

    def match(self, data: MeterData, cycle):
        # TODO: Implement this.
        return 0.2
