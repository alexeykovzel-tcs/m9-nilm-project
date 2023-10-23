from scipy.io import loadmat
import numpy as np
import os


resources_dir = os.path.abspath('resources/belkin') + '/'

files = {
    1: [
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
    ]
}


class BelkinData:
    def __init__(self, power, hf, pf, times, tags):
        self.power, self.hf, self.pf, self.times, self.tags = power, hf, pf, times, tags

    def real_power(self):
        return np.real(self.power)

    def reactive_power(self):
        return np.imag(self.power)

    def apparent_power(self):
        return np.abs(self.power)

    def truncate(self, start=None, stop=None, offset=0.0):
        max_i = len(self.times) - 1
        i1 = _closest_idx(self.times, start) if start is not None else 0
        i2 = _closest_idx(self.times, stop) if stop is not None else max_i

        if offset != 0:
            i_offset = int((i2 - i1) * offset)
            i1 = max(i1 - i_offset, 0)
            i2 = min(i2 + i_offset, max_i)

        return BelkinData(
            self.power[i1:i2],
            self.hf[i1:i2],
            self.pf[i1:i2],
            self.times[i1:i2],
            self.tags
        )


def load_sample(file_idx=0, home_idx=1) -> BelkinData:
    path = f'h{home_idx}/{files[home_idx][file_idx]}'
    return load_data(path)


def load_data(path) -> BelkinData:
    data = loadmat(resources_dir + path)
    return _process_raw_data(data)


def _process_raw_data(data):
    buffer = data['Buffer']

    # set L2 times as default
    times = buffer['TimeTicks2'][0][0][:, 0]

    # retrieve L1 & L2 powers
    l1_powers = buffer['LF1V'][0][0] * np.conj(buffer['LF1I'][0][0])
    l2_powers = buffer['LF2V'][0][0] * np.conj(buffer['LF2I'][0][0])

    l1_times = buffer['TimeTicks1'][0][0][:, 0]
    l1_powers = _align_times(l1_powers, l1_times, times)

    powers = l1_powers + l2_powers
    net_power = powers.sum(axis=1)

    # retrieve high-frequency noise
    hf_times = buffer['TimeTicksHF'][0][0][:, 0]
    hf = np.transpose(buffer['HF'][0][0])
    hf = _align_times(hf, hf_times, times)

    # retrieve power factor for the 1st 60Hz component
    pf = np.cos(np.angle(powers[:, 0]))

    # retrieve appliance tags
    tags = None
    if 'TaggingInfo' in buffer.dtype.names:
        tags = [[x[0][0] for x in y] for y in buffer['TaggingInfo'][0][0]]
        tags = [[x[0], x[1][0], x[2], x[3]] for x in tags]

    return BelkinData(net_power, hf, pf, times, tags)


def _align_times(vals, times, result_times):
    result = []
    i, max_i = 0, len(vals) - 1

    for t in result_times:
        if times[i] < t and i < max_i: i += 1
        result.append(vals[i])

    return np.array(result)


def _closest_idx(arr, el):
    return min(range(len(arr)), key=lambda i: abs(arr[i] - el))
