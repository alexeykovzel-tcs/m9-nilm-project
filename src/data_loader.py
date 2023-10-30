from src.signals import *
from src.meter_data import MeterData, Tag
from scipy.io import loadmat
import numpy as np
import os


"""

This file is for loading and processing the raw data into the MeterData class.
(see meter_data.py for more information)

"""

tagged_files = {}
test_files = {}

data_dir = os.path.abspath('data')

for h in os.listdir(data_dir):
    files = os.listdir(f'{data_dir}/{h}')
    tagged_files[h] = sorted([f for f in files if f.startswith('Tagged')])
    test_files[h] = sorted([f for f in files if f.startswith('Testing')])


def load_all_tagged(h_dir) -> [MeterData]:
    return [load(h_dir, f) for f in tagged_files[h_dir]]


def load_tagged(h_dir, idx) -> MeterData:
    """ loads a train file by index """
    return load(h_dir, tagged_files[h_dir][idx])


def load_test(h_dir, idx) -> MeterData:
    """ loads a test file by index """
    return load(h_dir, test_files[h_dir][idx])


def load(h_dir, name) -> MeterData:
    """ loads a file by name """
    data = loadmat(f'{data_dir}/{h_dir}/{name}')
    return _process_raw_data(data)


def _process_raw_data(data) -> MeterData:
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
        tags = [Tag(x[0], x[1][0], x[2], x[3]) for x in tags]

    return MeterData(l1, l2, hf, tags)
