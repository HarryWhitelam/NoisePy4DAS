import sys
sys.path.append("./src")
sys.path.append("./DASstore")

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from operator import itemgetter

from TDMS_Read import TdmsReader
from TDMS_Utilities import scale


def max_min_strain_rate(data, channel_bounds=None):
    """Takes in a 2D numpy array of TDMS data and returns the min and max values

    Keyword arguments:
        data -- A numpy array containing TDMS data
    """
    max_val = 0
    max_idx = 0
    min_val = 0
    min_idx = 0

    for time in data:
        if channel_bounds != None:
            time = time[channel_bounds[0]:channel_bounds[1]]
        max = time.max()
        min = time.min()
        if max > max_val:
            max_val = max
            max_idx = np.where(time == max)[0]

        if min < min_val:
            min_val = min
            min_idx = np.where(time == min)[0]

    if channel_bounds != None: 
        max_idx += channel_bounds[0]
        min_idx += channel_bounds[0]
    
    return max_val, max_idx, min_val, min_idx


file_path =  "../../temp_data_store/Linewalk/LineWalkData_UTC_20240110_120340.208.tdms"
# print('File: {0}'.format(file_path))

tdms = TdmsReader(file_path)
props = tdms.get_properties()

#where does data recording start
zero_offset = props.get('Zero Offset (m)')
#where does each channel sit along the cable
channel_spacing = props.get('SpatialResolution[m]') * props.get('Fibre Length Multiplier')
#how many channels are there
n_channels = tdms.fileinfo['n_channels']
#distance along the cable called depth here but hey
depth = zero_offset + np.arange(n_channels) * channel_spacing
#sampling frequency
fs = props.get('SamplingFrequency[Hz]')
time = props.get('GPSTimeStamp')

# print('Number of channels in file: {0}'.format(n_channels))
# print('Time samples in file: {0}'.format(tdms.channel_length))
# print('Sampling frequency (Hz): {0}'.format(fs))
# print(f'Time of Recording: {time}')

first_channel = 0
#If you want to read to the end get the channel length minus one
last_channel = n_channels

start_time = datetime(year=2024, month=1, day=10, hour=12, minute=3, second=40)
target_time = datetime(year=2024, month=1, day=10, hour=12, minute=35, second=50)
ms_diff = (target_time - start_time).total_seconds() * 1000

first_time_sample = ms_diff - 10000
second_time_sample = ms_diff + 10000

some_data = tdms.get_data(first_channel, last_channel, first_time_sample, second_time_sample)
# print('Size of data loaded: {0}'.format(some_data.shape))
some_data = scale(some_data, props)

max_strain, max_channel, min_strain, min_channel = max_min_strain_rate(some_data, [8000, 8400])
print(f"Min Strain: {min_strain} at channel {min_channel}, Max Strain: {max_strain} at channel {max_channel}")


fig1 = plt.figure()

bounds = 10000000

# img1 = plt.imshow(some_data, aspect='auto', interpolation='none', extent=(depth[first_channel],depth[last_channel-1], ((second_time_sample - 1)/fs), (first_time_sample/fs)), vmin=-bounds, vmax=bounds)
img1 = plt.imshow(some_data, aspect='auto', interpolation='none', extent=(first_channel, last_channel, ((second_time_sample - 1)/fs), (first_time_sample/fs)), vmin=-bounds, vmax=bounds)
plt.ylabel('Time (seconds)')
#plt.xlim(-100, 2000)
plt.xlabel('Channel No.')
plt.title((props.get('GPSTimeStamp')))
plt.set_cmap('bwr')
fig1.colorbar(img1, label= "Nano Strain per Second [nm/m/s]")

plt.show()
