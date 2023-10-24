from scipy.io import loadmat
from src.signals import *
import numpy as np
import os


resources_dir = 'resources/belkin'

files = {
    'h1': [
        'Tagged_Training_04_13_1334300401.mat',
        'Tagged_Training_10_22_1350889201.mat',
        'Tagged_Training_10_23_1350975601.mat',
        'Tagged_Training_10_24_1351062001.mat',
        'Tagged_Training_10_25_1351148401.mat',
        'Tagged_Training_12_27_1356595201.mat',
        'Testing_07_09_1341817201.mat',
        'Testing_07_11_1341990001.mat',
        'Testing_07_12_1342076401.mat',
        'Testing_07_16_1342422001.mat'
    ],
    'h2': [],
    'h3': [],
    'h4': []
}


class BelkinData:
    def __init__(self, l1: ComplexPower, l2: ComplexPower, hf: FreqNoise, tags):
        self.l1, self.l2, self.hf, self.tags = l1, l2, hf, tags

    def tagged(self):
        start = min(x[2] for x in self.tags)
        stop = max(x[3] for x in self.tags)
        return self.truncate(start, stop)

    def truncate(self, start=None, stop=None):
        return BelkinData(
            self.l1.truncate(start, stop),
            self.l2.truncate(start, stop),
            self.hf.truncate(start, stop),
            self.tags
        )


def load_sample(file_idx=0, h_dir='h1') -> BelkinData:
    path = os.path.abspath(resources_dir)
    data = loadmat(f'{path}/{h_dir}/{files[h_dir][file_idx]}')
    return _process_raw_data(data)


def _process_raw_data(data):
    buffer = data['Buffer']

    # 1st phase power
    l1_vals = buffer['LF1V'][0][0] * np.conj(buffer['LF1I'][0][0])
    l1_times = buffer['TimeTicks1'][0][0][:, 0]
    l1 = ComplexPower(l1_vals, l1_times)

    # 2nd phase power
    l2_vals = buffer['LF2V'][0][0] * np.conj(buffer['LF2I'][0][0])
    l2_times = buffer['TimeTicks2'][0][0][:, 0]
    l2 = ComplexPower(l2_vals, l2_times)

    # high-frequency noise
    hf_vals = np.transpose(buffer['HF'][0][0])
    hf_times = buffer['TimeTicksHF'][0][0][:, 0]
    hf = FreqNoise(hf_vals, hf_times)

    # appliance tags
    tags = None
    if 'TaggingInfo' in buffer.dtype.names:
        tags = [[x[0][0] for x in y] for y in buffer['TaggingInfo'][0][0]]
        tags = [[x[0], x[1][0], x[2], x[3]] for x in tags]

    return BelkinData(l1, l2, hf, tags)
