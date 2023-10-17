import numpy as np
from scipy.io import loadmat
from datetime import datetime


root_path = 'resources/'

train_files = {
    'h1': [
        'Tagged_Training_04_13_1334300401.mat',
        'Tagged_Training_10_22_1350889201.mat',
        'Tagged_Training_10_23_1350975601.mat',
        'Tagged_Training_10_24_1351062001.mat',
        'Tagged_Training_10_25_1351148401.mat',
        'Tagged_Training_12_27_1356595201.mat'
    ]
}

test_files = {
    'h1': [
        'Testing_07_09_1341817201.mat',
        'Testing_07_11_1341990001.mat',
        'Testing_07_12_1342076401.mat',
        'Testing_07_16_1342422001.mat'
    ]
}


def to_datetimes(time_ticks):
    return np.array([datetime.fromtimestamp(t) for t in time_ticks])


def process_raw_data(buffer):
    result = {}

    l1_p = buffer['LF1V'][0][0] * np.conj(buffer['LF1I'][0][0])
    l2_p = buffer['LF2V'][0][0] * np.conj(buffer['LF2I'][0][0])

    # Compute net Complex power
    l1 = l1_p.sum(axis=1)
    l2 = l2_p.sum(axis=1)

    # Real, Reactive, Apparent powers
    result['L1_Real'] = np.real(l1)
    result['L1_Imag'] = np.imag(l1)
    result['L1_App'] = np.abs(l1)

    result['L2_Real'] = np.real(l2)
    result['L2_Imag'] = np.imag(l2)
    result['L2_App'] = np.abs(l2)

    # Compute Power Factor, we only consider the first 60Hz component
    result['L1_Pf'] = np.cos(np.angle(l1_p[:, 0]))
    result['L2_Pf'] = np.cos(np.angle(l2_p[:, 0]))

    # Copy Time ticks to our processed structure
    result['L1_TimeTicks'] = buffer['TimeTicks1'][0][0][:, 0]
    result['L2_TimeTicks'] = buffer['TimeTicks2'][0][0][:, 0]
    result['L1_Datetimes'] = to_datetimes(result['L1_TimeTicks'])
    result['L2_Datetimes'] = to_datetimes(result['L2_TimeTicks'])

    # Move over HF Noise and Device label (tagging) data to our final structure as well
    result['HF'] = np.transpose(buffer['HF'][0][0])
    result['HF_TimeTicks'] = buffer['TimeTicksHF'][0][0][:, 0]
    result['HF_Datetimes'] = to_datetimes(result['HF_TimeTicks'])

    # Copy Labels/TaggingInfo if they exist
    if 'TaggingInfo' in buffer.dtype.names:
        tag_info = [[x[0][0] for x in y] for y in buffer['TaggingInfo'][0][0]]
        tag_info = [[x[0], x[1][0], x[2], x[3]] for x in tag_info]
        result['TaggingInfo'] = tag_info

    return result


def load_file(path):
    data = loadmat(root_path + path)['Buffer']
    return process_raw_data(data)


def load_sample_file(idx=0, reason='train', h_id='h1'):
    if reason == 'train':
        path = f'{h_id}/{train_files[h_id][idx]}'
        return load_file(path)
