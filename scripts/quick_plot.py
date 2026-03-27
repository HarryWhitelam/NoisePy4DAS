import sys
sys.path.append("./src")
sys.path.append("./DASstore")

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from operator import itemgetter

from TDMS_Read import TdmsReader
from TDMS_Utilities import scale

from tdms_io import get_filepath_array, get_data_from_array, get_subset_paths


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


dir_path =  "/data/QNAP1_Data/Data/"
# file_path =  "/data/QNAP1_Data/Data/20241015_1m_spatial/100hz_UTC_20241017_113924.138.tdms"

n_minutes = 10
t_start = datetime(year=2025, month=2, day=16, hour=0, minute=0, second=0)

t_start = datetime(year = 2025, month = 3, day = 27,
                   hour = 11, minute = 22, second = 39)
t_end = t_start + timedelta(minutes=n_minutes)

prepro_para = {
                'samp_freq': 100,
                'target_spatial_res': 1,
                'n_minute': n_minutes,
                'freqmin': 0.01,
                'freqmax': 49.9,
                'cha1':100,
                'cha2':1650,
            }

filepath_array, timestamps = get_filepath_array(dir_path, t_start, t_end)
t_start = timestamps[0].replace(microsecond=0)
t_end = timestamps[-1].replace(microsecond=0)
print(f'Data running from {t_start} to {t_end}')
data = get_data_from_array(filepath_array, prepro_para, t_start, duration=timedelta(minutes=n_minutes))

fig1 = plt.figure()
img1 = plt.imshow(data, aspect='auto', interpolation='none', extent=(prepro_para.get('cha1'), prepro_para.get('cha2'), data.shape[0]/prepro_para.get('samp_freq'), 0), vmin=np.nanpercentile(data, 1), vmax=np.nanpercentile(data, 99))
plt.ylabel('Time (seconds)')
plt.xlabel('Channel No.')
plt.title(t_start)
plt.set_cmap('bwr')
fig1.colorbar(img1, label= "Nano Strain per Second [nm/m/s]")
plt.tight_layout()
# plt.savefig('./results/figures/kamchatka_waterfall.png')
plt.show()





##### OLD CODE #####
# tdms = TdmsReader(file_path)
# props = tdms.get_properties()

# #where does data recording start
# zero_offset = props.get('Zero Offset (m)')
# #where does each channel sit along the cable
# channel_spacing = props.get('SpatialResolution[m]') * props.get('Fibre Length Multiplier')
# #how many channels are there
# n_channels = tdms.fileinfo['n_channels']
# #distance along the cable called depth here but hey
# depth = zero_offset + np.arange(n_channels) * channel_spacing
# #sampling frequency
# fs = props.get('SamplingFrequency[Hz]')
# time = props.get('GPSTimeStamp')

# print('Number of channels in file: {0}'.format(n_channels))
# print('Time samples in file: {0}'.format(tdms.channel_length))
# print('Sampling frequency (Hz): {0}'.format(fs))
# print(f'Time of Recording: {time}')

# first_channel = 0
# last_channel = n_channels

### needed to select        # not really working cbb to fix it
# start_time = datetime.strptime(file_path[-24:-5], '%Y%m%d_%H%M%S.%f')
# end_time = start_time + timedelta(seconds=tdms.channel_length/fs)
# ms_diff = (end_time - start_time).total_seconds() * 1000

# first_time_sample = ms_diff - 10000
# second_time_sample = ms_diff + 10000

# data = tdms.get_data(first_channel, last_channel, first_time_sample, second_time_sample)
# print('Size of data loaded: {0}'.format(data.shape))

### or get all
# data = tdms.get_data()
# data = scale(data, props)