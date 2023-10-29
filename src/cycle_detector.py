import numpy as np

from src.cluster_storage import ClusteredData
from src.signals import *
from sklearn.cluster import DBSCAN


def detect_cycles(signal: Power, eps=100, k=1, min_time=30):
    clusters = _group(signal, eps, k)
    bounds = clusters.boundaries()
    bounds = sorted(bounds, key=lambda x: x[0])
    cycles = _clean_cycles(bounds, min_time)
    return cycles


def _group(signal: Power, eps, k):
    x_reactive = _extract_trues(signal, 'reactive', k)
    x_real = _extract_trues(signal, 'real', k)
    x = np.concatenate((x_reactive.times, x_real.times))
    x = np.array(x).reshape(-1, 1)

    model = DBSCAN(eps=eps)
    labels = model.fit_predict(x)

    clusters = ClusteredData(labels, x)
    clusters.compose()

    return clusters


def _find_threshold(signal_vals, k):
    mean = np.mean(signal_vals)
    std_dev = np.std(signal_vals)
    return mean + k * std_dev


def _clean_cycles(bounds, min_time):
    diffs = [x[1] - x[0] for x in bounds]
    cleaned = []

    for i in range(len(diffs)):
        if diffs[i] >= min_time:
            cleaned.append(bounds[i])

    return cleaned


def _extract_trues(signal: Power, power, k):
    t_vals = []
    t_times = []

    if power == 'reactive':
        data_signal = signal.reactive()
    elif power == 'real':
        data_signal = signal.real()
    else:
        raise Exception("Available power types: 'reactive', 'real'.")

    threshold = _find_threshold(data_signal, k)

    for i in range(len(data_signal.real)):
        if abs(data_signal.real[i]) > threshold:
            t_vals.append([data_signal.real[i]])
            t_times.append(signal.times[i])

    return Signal(t_vals, t_times)
