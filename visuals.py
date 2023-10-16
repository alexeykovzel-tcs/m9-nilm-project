import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime
import numpy as np
import random
import copy

def to_datetimes(timestamps):
    return np.array([datetime.fromtimestamp(ts) for ts in timestamps])

def find_closest(arr, el):
    return min(range(len(arr)), key=lambda i: abs(arr[i]-el))

def trunc_range(data, ts_range, time_key, keys):
    start = find_closest(data[time_key], ts_range[0])
    stop = find_closest(data[time_key], ts_range[1])
    for key in keys:
        data[key] = data[key][start:stop]

def belkin_plot_data(data, tags=[], ts_range=None):
    fig, axs = plt.subplots(4, 1, figsize=(10, 8))

    # Truncate data by the time series range
    if ts_range != None:
        data = copy.deepcopy(data)
        l1_keys = ['L1_TimeTicks', 'L1_Real', 'L1_App', 'L1_Imag', 'L1_Pf']
        l2_keys = ['L2_TimeTicks', 'L2_Real', 'L2_App', 'L2_Imag', 'L2_Pf']
        trunc_range(data, ts_range, 'L1_TimeTicks', l1_keys)
        trunc_range(data, ts_range, 'L2_TimeTicks', l2_keys)
        trunc_range(data, ts_range, 'HF_TimeTicks', ['HF_TimeTicks', 'HF'])

    # Convert time ticks to datetime format for better visualization
    times_l1 = to_datetimes(data['L1_TimeTicks'])
    times_l2 = to_datetimes(data['L2_TimeTicks'])
    times_hf = to_datetimes(data['HF_TimeTicks'])

    # Plot Real Power (W) and ON/OFF Device Category IDs
    axs[0].plot(times_l1, data['L1_Real'])
    axs[0].plot(times_l2, data['L2_Real'], 'c')
    axs[0].set_title('Real Power (W) and ON/OFF Device Category IDs')
    axs[0].grid(True)

    if tags and 'TaggingInfo' in data:
        for tag in data['TaggingInfo']:
            if tag[1] not in tags: continue
            height = random.randrange(500, 5000)

            # device ON and OFF times
            on_time = datetime.fromtimestamp(tag[2])
            off_time = datetime.fromtimestamp(tag[3])
            
            axs[0].axvline(x=on_time, color='g')
            axs[0].text(on_time, height, f"ON\n{tag[1]}", size='small')
            axs[0].axvline(x=off_time, color='r')
            axs[0].text(off_time, height + 750, "OFF", size='small')

    # Plot Imaginary/Reactive power (VAR)
    axs[1].plot(times_l1, data['L1_Imag'])
    axs[1].plot(times_l2, data['L2_Imag'], 'c')
    axs[1].set_title('Imaginary/Reactive power (VAR)')
    axs[1].grid(True)

    # Plot Power Factor
    axs[2].plot(times_l1, data['L1_Pf'])
    axs[2].plot(times_l2, data['L2_Pf'], 'c')
    axs[2].set_title('Power Factor')
    axs[2].grid(True)

    # Plot HF Noise
    freqs = np.linspace(0, 1e6, 6)
    times_hf = np.transpose(times_hf)
    axs[3].imshow(np.transpose(data['HF']), aspect='auto', origin='lower',
                  extent=[times_hf[0], times_hf[-1], 0, 1e6])

    axs[3].set_title('High Frequency Noise')
    axs[3].set_ylabel('Frequency KHz')
    axs[3].set_yticks(freqs)
    axs[3].set_yticklabels([f"{int(f/1e3)}K" for f in freqs])

    for ax in axs:
        ax.xaxis.set_major_formatter(DateFormatter("%H-%M"))

    plt.tight_layout(h_pad=2)
    plt.show()