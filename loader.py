import numpy as np
from scipy.io import loadmat

h1_path = '../resources/h1/'

train_files = [
    'Tagged_Training_04_13_1334300401.mat',
    'Tagged_Training_10_22_1350889201.mat',
    'Tagged_Training_10_23_1350975601.mat',
    'Tagged_Training_10_24_1351062001.mat',
    'Tagged_Training_10_25_1351148401.mat',
    'Tagged_Training_12_27_1356595201.mat'
]

test_files = [
    'Testing_07_09_1341817201.mat',
    'Testing_07_11_1341990001.mat',
    'Testing_07_12_1342076401.mat',
    'Testing_07_16_1342422001.mat'
]

def belkin_process_raw_data(buffer):
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

    # Move over HF Noise and Device label (tagging) data to our final structure as well
    result['HF'] = np.transpose(buffer['HF'][0][0])
    result['HF_TimeTicks'] = buffer['TimeTicksHF'][0][0][:, 0]

    # Copy Labels/TaggingInfo if they exist
    if 'TaggingInfo' in buffer.dtype.names:
        result['TaggingInfo'] = [[x[0][0] for x in y] for y in buffer['TaggingInfo'][0][0]]

    return result

def load_h1(file):
    data = loadmat(h1_path + file)['Buffer']
    return belkin_process_raw_data(data)