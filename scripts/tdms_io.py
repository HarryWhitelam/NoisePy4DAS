import sys
sys.path.append("./src")
sys.path.append("./DASstore")

from TDMS_Read import TdmsReader
import os
import numpy as np
from obspy import Stream, Trace
from obspy.core.trace import Stats
from obspy.core.utcdatetime import UTCDateTime
from datetime import datetime, timedelta
from math import floor, ceil
from skimage.transform import resize


def get_tdms_array(dir_path:str):
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


def get_closest_index(timestamps:np.ndarray, time:datetime):
    """retrieves the index of the closest timestamp within timestamps to time

    Args:
        timestamps (ndarray): _description_
        time (datetime): _description_

    Returns:
        _type_: _description_
    """    
    # array must be sorted
    idx = timestamps.searchsorted(time)
    idx = np.clip(idx, 1, len(timestamps)-1)
    idx -= time - timestamps[idx-1] < timestamps[idx] - time
    return idx


def get_dir_properties(dir_path:str):
    with os.scandir(dir_path) as files:
        for file in files:
            if file.is_file():
                file_path = file.path
                break
    tdms_file = TdmsReader(file_path)
    tdms_file._read_properties()
    return tdms_file.get_properties()


# returns a delta-long array of tdms files starting at the timestamp given
def get_time_subset(tdms_array:np.ndarray, start_time:datetime, timestamps:np.ndarray, tpf:int, delta:int, tolerance:int=300):
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


# returns a duration of data (default is 60s)
def get_data_from_array(tdms_array:list, prepro_para:dict, start_time:datetime, timestamps:np.ndarray):
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


def scale(data:np.ndarray, props:dict):
    """Takes in TDMS data and its properties using them to scale the data as it is compressed within the file format. Returns scaled data

    strainrate nm/m/s = 116 * iDAS values * sampling freq (Hz) / gauge lenth (m)

    Keyword arguments:
        data -- numpy array containing TDMS data
        props -- properties struct from TDMS reader
    """
    data = data / 8192
    data = (116 * data * props.get('SamplingFrequency[Hz]')) / props.get('GaugeLength')
    return data


def slice_downsample(data, temporal_ratio, spatial_ratio):
    return data[::temporal_ratio, ::spatial_ratio]


def mean_downsample(data:np.ndarray, temporal_ratio:int, spatial_ratio:int):
    shape = data.shape
    ds_data = np.empty(shape=(ceil(shape[0]/temporal_ratio), ceil(shape[1]/spatial_ratio)))
    
    for i in range(0, shape[0], temporal_ratio):
        if i > temporal_ratio:
            i_left = i - floor(temporal_ratio / 2)
        else: 
            i_left = i
        i_right = i + ceil(temporal_ratio / 2)
        
        for j in range(0, shape[1], spatial_ratio):
            if j > temporal_ratio:
                j_left = j - floor(spatial_ratio / 2)
            else:
                j_left = j
            j_right = j + ceil(spatial_ratio / 2)
            
            # print(f'{i_left}:{i_right}, {j_left}:{j_right}')
            ds_data[i // temporal_ratio, j // spatial_ratio] = \
                np.mean(data[i_left:i_right, j_left:j_right])
    
    return ds_data


def downsample_file(file_path:str, output_path:str, target_sps:float, target_spatial_res:float):
    tdms = TdmsReader(file_path)
    props = tdms.get_properties()
    print(props)
    
    sps = props.get('SamplingFrequency[Hz]')
    spatial_res = props.get('SpatialResolution[m]')
    data = tdms.get_data(0, tdms.fileinfo['n_channels'])
    
    if (temporal_ratio := int(sps/target_sps)) != sps/target_sps:             # reversed as time-reciprocal
        print(f'Target sps not a factor of current sps, some data will be lost.')
    if (spatial_ratio := int(target_spatial_res/spatial_res)) != target_spatial_res/spatial_res:
        print(f'Target spatial res not a factor of current spatial res, some data will be lost.')
    resize_data = resize(data, output_shape=(data.shape[0] / temporal_ratio, data.shape[1] / spatial_ratio))
    
    # TODO: write to TDMS file (how?)


def tdms_to_stream(file_path:str):
    tdms = TdmsReader(file_path)
    props = tdms.get_properties()
    data = tdms.get_data()
    print(data.shape)
    
    # NOTE: This is starting as just one Stats object for ALL traces in the stream, this may have to change
    stats = Stats()
    stats.network = 'CLF'       # (hehe cliff)
    stats.sampling_rate = props.get('SamplingFrequency[Hz]')
    stats.npts = data.shape[0]
    stats.starttime = UTCDateTime(props.get('GPSTimeStamp'))
    # stats.endtime = stats.starttime + (data.shape[0] / stats.sampling_rate)       # redundant? calculated automatically? 
    
    stream = Stream()
    for count, channel in enumerate(data.T):
        stats.station = str(count)
        stream += Trace(channel, stats)
    
    print(stream)


def max_min_strain_rate(data:np.ndarray, channel_bounds:list=None):
    """Takes in a 2D numpy array of TDMS data and returns the min and max values

    Keyword arguments:
        data -- A numpy array containing TDMS data
    """
    max_val = 0
    max_idx = 0
    min_val = 0
    min_idx = 0

    for sample in data:
        if channel_bounds != None:
            sample = sample[channel_bounds[0]:channel_bounds[1]]
        max = sample.max()
        min = sample.min()
        if max > max_val:
            max_val = max
            max_idx = np.where(sample == max)[0]

        if min < min_val:
            min_val = min
            min_idx = np.where(sample == min)[0]

    if channel_bounds != None: 
        max_idx += channel_bounds[0]
        min_idx += channel_bounds[0]
    
    return max_val, max_idx, min_val, min_idx


# print(f'temporal_ratio: {temporal_ratio}, spatial_ratio: {spatial_ratio}')

# data = np.empty(shape=(10, 7))
# for i in range(1, 11):
#     data[i-1] = [i, i*2, i*4, i*5, i*10, random(), random()]
# print(data)
# downsample_file("../../temp_data_store/Snippets/FirstData_UTC_20231109_134947.573.tdms", None, 100, 1.3)
tdms_to_stream("../../temp_data_store/Snippets/FirstData_UTC_20231109_134947.573.tdms")
