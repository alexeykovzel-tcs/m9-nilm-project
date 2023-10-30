from src.appliance_guesser import ApplianceGuesser
from src.meter_data import MeterData
from src.appliance import Appliance
from collections import defaultdict


class EnergyModel:
    def __init__(self, data_per_day: [MeterData]):
        self.guesser = self._init_guesser(data_per_day)

    def disaggregate(self, data: MeterData):
        guesses = self._guess_appliances(data.power_cycles)
        if not guesses: return [], [], []

        times = data.total_power.times
        labels = [appliance.label for appliance in guesses.keys()]
        powers = [appliance.base_power(times, powers) for appliance, powers in guesses.items()]

        other_power = data.total_power.real() - sum(powers)
        powers = [other_power] + powers
        labels = ['Other'] + labels

        return times, powers, labels

    def _guess_appliances(self, power_cycles):
        guesses = defaultdict(list)
        for power_phase, cycle in power_cycles:
            power = power_phase.truncate(cycle)
            appliance = self.guesser.guess_by_power(power)
            guesses[appliance].append(power)

        return guesses

    @staticmethod
    def _init_guesser(data_per_day: [MeterData]):
        appliances = {}
        for data in data_per_day:
            for tag, cycle in data.tagged_cycles():
                appliance = appliances.setdefault(tag.idx, Appliance(tag.idx, tag.label))
                appliance.cycles.append(cycle)

        appliances = list(appliances.values())
        return ApplianceGuesser(appliances)
