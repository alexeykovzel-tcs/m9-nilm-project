from src.signals import *
from scipy.io import loadmat
import numpy as np
import os


data_dir = os.path.abspath('data')

train_files = {}
test_files = {}

for h in ['h1', 'h2', 'h3', 'h4']:
    files = os.listdir(f'{data_dir}/{h}')
    train_files[h] = [f for f in files if f.startswith('Tagged')]
    test_files[h] = [f for f in files if f.startswith('Testing')]


class MeterData:
    def __init__(self, l1: Power, l2: Power, hf: FreqNoise, tags):
        self.l1, self.l2, self.hf, self.tags = l1, l2, hf, tags

    def total_power(self):
        return self.l1 + self.l2

    def tagged(self):
        start = min(x[2] for x in self.tags)
        stop = max(x[3] for x in self.tags)
        return self.truncate(start, stop)

    def truncate(self, start=None, stop=None):
        return MeterData(
            self.l1.truncate(start, stop),
            self.l2.truncate(start, stop),
            self.hf.truncate(start, stop),
            self.tags
        )


def load_train(h_dir, idx):
    return load(h_dir, train_files[h_dir][idx])


def load_test(h_dir, idx):
    return load(h_dir, test_files[h_dir][idx])


def load(h_dir, name) -> MeterData:
    data = loadmat(f'{data_dir}/{h_dir}/{name}')
    return _process_raw_data(data)


def _process_raw_data(data):
    buffer = data['Buffer']

    # 1st phase power
    l1_vals = buffer['LF1V'][0][0] * np.conj(buffer['LF1I'][0][0])
    l1_times = buffer['TimeTicks1'][0][0][:, 0]
    l1 = Power(l1_vals, l1_times)

    # 2nd phase power
    l2_vals = buffer['LF2V'][0][0] * np.conj(buffer['LF2I'][0][0])
    l2_times = buffer['TimeTicks2'][0][0][:, 0]
    l2 = Power(l2_vals, l2_times)

    # align L1 & L2 times
    if len(l1.vals) < len(l2.vals):
        l1 = l1.align_times(l2.times)
    else:
        l2 = l2.align_times(l1.times)

    # high-frequency noise
    hf_vals = np.transpose(buffer['HF'][0][0])
    hf_times = buffer['TimeTicksHF'][0][0][:, 0]
    hf = FreqNoise(hf_vals, hf_times)

    # appliance tags
    tags = None
    if 'TaggingInfo' in buffer.dtype.names:
        tags = [[x[0][0] for x in y] for y in buffer['TaggingInfo'][0][0]]
        tags = [[x[0], x[1][0], x[2], x[3]] for x in tags]

    return MeterData(l1, l2, hf, tags)
