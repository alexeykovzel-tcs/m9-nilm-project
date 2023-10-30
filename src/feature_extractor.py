from scipy.stats import skew, kurtosis
from src.signals import Power
import numpy as np


def appliance_features(cycles: [Power]):
    return np.mean([power_features(cycle) for cycle in cycles], axis=0)


def power_features(power: Power):
    return np.concatenate([
        stats(power.real()),
        stats(power.reactive()),
        stats(power.factor()),
        [power.len()]
    ])


def stats(vals):
    return np.array([
        np.std(vals),
        np.mean(vals),
        kurtosis(vals),
        skew(vals)
    ])
