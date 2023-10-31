from scipy.stats import skew, kurtosis
from prod.signals import Power
from datetime import datetime
import numpy as np


def power_features(power: Power):
    return np.concatenate([
        _basic_stats(power.real().vals),
        _basic_stats(power.reactive().vals),
        _basic_stats(power.factor().vals),
        [power.duration, _day_time(power.times)],
    ])


def _basic_stats(vals):
    return np.array([
        np.std(vals),
        np.mean(vals),
        kurtosis(vals),
        skew(vals),
        (np.argmax(vals) + 1) / len(vals)
    ])


def _day_time(times):
    time = (times[-1] - times[0]) / 2
    time = datetime.utcfromtimestamp(time).time()
    day_fraction = time.hour / 24.0 + time.minute / 1440.0 + time.second / 86400.0
    return day_fraction
