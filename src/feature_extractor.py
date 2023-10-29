from scipy.stats import skew, kurtosis
import numpy as np


# TODO: Add more feature extractor options.

def stats(vals):
    return np.array([np.std(vals), np.mean(vals), kurtosis(vals), skew(vals)])
