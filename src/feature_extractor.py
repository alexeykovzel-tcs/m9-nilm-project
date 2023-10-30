from scipy.stats import skew, kurtosis
from src.signals import Power
from datetime import datetime
import numpy as np


def power_features(power: Power):
    real = power.real()
    reactive = power.reactive()
    factor = power.factor()

    return np.concatenate([
        _basic_stats(real),
        _basic_stats(reactive),
        _basic_stats(factor),
        [
            power.len(),
            _day_time(power.times),
            _peak_time(reactive),
            _peak_time(real),
            _peak_time(factor)
        ],
    ])


def _basic_stats(vals):
    stats = [np.std(vals), np.mean(vals), kurtosis(vals), skew(vals)]
    return np.array(stats)


def _peak_time(vals):
    return (np.argmax(vals) + 1) / len(vals)


def _day_time(times):
    time = (times[-1] - times[0]) / 2
    time = datetime.utcfromtimestamp(time).time()
    day_fraction = time.hour / 24.0 + time.minute / 1440.0 + time.second / 86400.0
    return day_fraction
