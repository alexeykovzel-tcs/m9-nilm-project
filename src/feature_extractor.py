from scipy.stats import skew, kurtosis
from src.signals import Power
from datetime import datetime
import numpy as np


def power_features(power: Power):
    times, factor = power.times, power.factor()
    real, reactive = power.real(), power.reactive()

    return np.concatenate([
        _basic_stats(real),
        _basic_stats(reactive),
        _basic_stats(factor),
        [
            power.len(),
            _day_time(times),
            _peak_time(times, reactive),
            _peak_time(times, real),
        ],
    ])


def _basic_stats(vals):
    stats = [np.std(vals), np.mean(vals), kurtosis(vals), skew(vals)]
    return np.array(stats)


def _peak_time(times, vals):
    time = times[np.argmax(vals)]
    return (time - times[0]) / (times[-1] - times[0])


def _day_time(times):
    time = (times[-1] - times[0]) / 2
    time = datetime.utcfromtimestamp(time).time()
    day_fraction = time.hour / 24.0 + time.minute / 1440.0 + time.second / 86400.0
    return 2 * day_fraction - 1
