from src.meter_data import Power
from scipy.stats import skew, kurtosis
import numpy as np


def extract_power_features(power: Power):
    params = np.array([power.times[-1] - power.times[0]])
    params += _extract_features(power.real())
    params += _extract_features(power.reactive())
    params += _extract_features(power.factor())
    return params


def _extract_features(vals):
    return np.array([np.std(vals), np.mean(vals), kurtosis(vals), skew(vals)])
