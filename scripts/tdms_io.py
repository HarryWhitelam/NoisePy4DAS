import sys
sys.path.append("./src")
sys.path.append("./DASstore")

import os
import numpy as np
from datetime import datetime, timedelta
from math import floor
from TDMS_Read import TdmsReader

from random import random


def get_tdms_array(dir_path):
    tdms_array = np.empty(int(len([filename for filename in os.listdir(dir_path) if filename.endswith(".tdms")])), dtype=TdmsReader)
        # tdms_array = np.empty(len(os.listdir(dir_path)), TdmsReader)
    timestamps = np.empty(len(tdms_array), dtype=datetime)

    for count, file in enumerate([filename for filename in os.listdir(dir_path) if filename.endswith(".tdms")]):
        if file.endswith('.tdms'):
            tdms = TdmsReader(dir_path + file)
            tdms_array[count] = tdms
            timestamps[count] = tdms.get_properties().get('GPSTimeStamp')
    timestamps.sort()
    print(f'{len(timestamps)} files available from {timestamps[0]} to {timestamps[-1]}')

    return [x for y, x in sorted(zip(np.array(timestamps), tdms_array))], timestamps


def get_closest_index(timestamps, time):
    """retrieves the index of the closest timestamp within timestamps to time

    Args:
        timestamps (ndarray): _description_
        time (timestamp): _description_

    Returns:
        _type_: _description_
    """    
    # array must be sorted
    idx = timestamps.searchsorted(time)
    idx = np.clip(idx, 1, len(timestamps)-1)
    idx -= time - timestamps[idx-1] < timestamps[idx] - time
    return idx


def get_dir_properties(dir_path):
    with os.scandir(dir_path) as files:
        for file in files:
            if file.is_file():
                file_path = file.path
                break
    tdms_file = TdmsReader(file_path)
    tdms_file._read_properties()
    return tdms_file.get_properties()


# returns a delta-long array of tdms files starting at the timestamp given
def get_time_subset(tdms_array, start_time, timestamps, tpf, delta, tolerance=300):
    # tolerence is the time in s that the closest timestamp can be away from the desired start_time
    # timestamps MUST be orted, and align with TDMS array (i.e. timestamps[n] represents tdms_array[n]
    start_idx = get_closest_index(timestamps, start_time)
    if abs((start_time - timestamps[start_idx]).total_seconds()) > tolerance:
        print(f"Error: first tdms is over {tolerance} seconds away from the given start time.")
        return
    
    end_time = timestamps[start_idx] + delta - timedelta(seconds=tpf)
    end_idx = get_closest_index(timestamps, end_time)
    if (end_time - timestamps[end_idx]).total_seconds() > tolerance:
        print(f"WARNING: end tdms is over {tolerance} seconds away from the calculated end time.")
    # print(f"Given t={start_time}, snippet selected from {timestamps[start_idx]} to {timestamps[end_idx]}!")
    
    if (end_idx - start_idx + 1) != (delta.seconds/tpf):
        print(f"WARNING: time subset not continuous; only {(end_idx - start_idx + 1)*tpf} seconds represented.")
    # for i in range(start_idx, end_idx+1):
    #     print(timestamps[i])
    
    return tdms_array[start_idx:end_idx+1]


# returns a minute of data
# currently CAN NOT HANDLE TDMS > 30 SECONDS - I THINK (it'll just clip the rest of the file, it can probably handle exactly 60s of data)
def get_data_from_array(tdms_array, prepro_para, start_time, timestamps):
    cha1, cha2, sps, spatial_ratio, duration = prepro_para.get('cha1'), prepro_para.get('cha2'), prepro_para.get('sps'), prepro_para.get('spatial_ratio'), prepro_para.get('duration')

    # make it so that if start_time is not a timestamp, the first minute in the array is returned
    current_time = 0
    tdms_t_size = tdms_array[0].get_data(cha1, cha2).shape[0]
    tdata = np.empty((int(duration * sps), floor((cha2-cha1+1)/spatial_ratio)))
    
    if type(start_time) is datetime:
        tdms_array = get_time_subset(tdms_array, start_time, timestamps, tpf=tdms_t_size/sps, delta=timedelta(seconds=duration), tolerance=30)   # tpf = time per file
    
    while current_time != duration and len(tdms_array) != 0:
        tdms = tdms_array.pop(0)
        props = tdms.get_properties()
        data = tdms.get_data(cha1, cha2)
        data = scale(data, props)
        data = data[:, ::spatial_ratio]
        current_row = current_time * sps
        tdata[int(current_row):int(current_row+(tdms_t_size)), :] = data
        current_time += tdms_t_size/sps
    
    return tdata


def scale(data, props):
    """Takes in TDMS data and its properties using them to scale the data as it is compressed within the file format. Returns scaled data

    strainrate nm/m/s = 116 * iDAS values * sampling freq (Hz) / gauge lenth (m)

    Keyword arguments:
        data -- numpy array containing TDMS data
        props -- properties struct from TDMS reader
    """
    data = data * 1.8192
    data = (116 * data * props.get('SamplingFrequency[Hz]')) / props.get('GaugeLength')
    return data


def slice_downsample(data, temporal_ratio, spatial_ratio):
    return data[::temporal_ratio, ::spatial_ratio]


def mean_downsample(data, temporal_ratio, spatial_ratio):
    shape = data.shape    
    ds_data = np.empty(shape=(int(shape[0]/temporal_ratio), int(shape[1]/spatial_ratio)))
    
    # for i in range(shape[1]):
    #     ds_data[i] = np.convolve(data[i], np.ones(spatial_ratio), 'valid') / spatial_ratio
    
    for i in range(0, shape[0], temporal_ratio):
        for j in range(0, shape[1], spatial_ratio):
            # print(i, j)
            ds_data[i // temporal_ratio, j // spatial_ratio] = \
                np.mean(data[i:i+temporal_ratio, j:j+spatial_ratio])
    
    return ds_data


sps = 1
spatial_res = 1
target_sps = 1
target_spatial_res = .67

if (temporal_ratio := int(sps/target_sps)) != sps/target_sps:             # reversed as time-reciprocal
    print(f'Target sps not a factor of current sps, some data will be lost.')
if (spatial_ratio := int(target_spatial_res/spatial_res)) != target_spatial_res/spatial_res:
    print(f'Target spatial res not a factor of current spatial res, some data will be lost.')

print(temporal_ratio, spatial_ratio)

test_data = np.empty(shape=(10, 6))
for i in range(1, 11):
    test_data[i-1] = [i, i*2, i*5, i*10, random(), random()]
# [[  1.           2.           5.          10.           0.90996733           0.11557851]
#  [  2.           4.          10.          20.           0.25683587           0.89005394]
#  [  3.           6.          15.          30.           0.10812548           0.45505769]
#  [  4.           8.          20.          40.           0.46343448           0.71672940]
#  [  5.          10.          25.          50.           0.86825790           0.69531352]
#  [  6.          12.          30.          60.           0.12835542           0.79795516]
#  [  7.          14.          35.          70.           0.31663804           0.85532936]
#  [  8.          16.          40.          80.           0.55079609           0.67358710]
#  [  9.          18.          45.          90.           0.16375882           0.96487782]
#  [ 10.          20.          50.         100.           0.66585205           0.81153159]]

# sliced_data = slice_downsample(test_data, temporal_ratio, spatial_ratio)
# print(sliced_data)

mean_data = mean_downsample(test_data, temporal_ratio, spatial_ratio)
print(mean_data)
