import sys
sys.path.append("./src")
sys.path.append("./DASstore")
import gc
from TDMS_Read import TdmsReader
from SegyReader import SegyReader
import os
import warnings
import numpy as np
import pickle
from tqdm import tqdm
from obspy import Stream, Trace
from obspy.core.trace import Stats
from obspy.signal.filter import bandpass
from obspy.core.utcdatetime import UTCDateTime
from datetime import datetime, timedelta, time as dt_time
from math import floor, ceil
from skimage.transform import resize
from scipy.signal import decimate


# def get_reader_array(dir_path:str, allowed_times:dict=None):
#     dir_list = [filename for filename in os.listdir(dir_path) if filename.endswith(('.tdms', '.segy'))]
#     file_ext = '.' + dir_list[0].rsplit('.', 1)[1]
#     reader_array = [None] * int(len(dir_list))
#     timestamps = np.empty(len(reader_array), dtype=datetime)
#     for count, file in enumerate(dir_list):
#         match file_ext:
#             case '.tdms': reader = TdmsReader(dir_path + file)
#             case '.segy': reader = SegyReader(dir_path + file)
#         timestamp = reader.get_properties().get('GPSTimeStamp')
#         if allowed_times and not is_valid_time(timestamp, allowed_times):
#             continue
#         reader_array[count] = reader
#         timestamps[count] = timestamp
#     timestamps = np.delete(timestamps, np.where(timestamps == None))
#     reader_array = [reader for reader in reader_array if reader is not None]
#     reader_array = [x for y, x in sorted(zip(timestamps, reader_array))]
#     timestamps.sort()
#     print(f'{len(timestamps)} files available from {timestamps[0]} to {timestamps[-1]}')
#     return reader_array, timestamps


def get_reader_array(root_dir:str, t_start:datetime=None, t_end:datetime=None):
    files = []
    for dir_path, dir_names, file_names in os.walk(root_dir):
        for file in file_names:
            if file.endswith(('.tdms', '.segy')): 
                try: 
                    timestamp = datetime.strptime(file[-24:-9], '%Y%m%d_%H%M%S')
                except Exception: 
                    continue        # basically there was no datetime within the file
                if t_start and timestamp < t_start:
                    continue
                if t_end and timestamp > t_end:
                    continue
                files.append((TdmsReader(os.path.join(dir_path, file)), timestamp))
    files.sort(key=lambda x: x[1])
    reader_array, timestamps = zip(*files)
    # print(f'Range: {timestamps[0]}: {timestamps[-1]}')
    # timestamps = np.asarray(timestamps, dtype=datetime)
    # timestamps.sort()
    return list(reader_array), np.asarray(timestamps, dtype=datetime)


def get_filepath_array(root_dir:str, t_start:datetime=None, t_end:datetime=None):
    files = []
    exclude = ['20250326_Jack_Experiments']
    for dir_path, dir_names, file_names in os.walk(root_dir, topdown=True):
        dir_names[:] = [d for d in dir_names if d not in exclude]
        for file in file_names:
            if file.endswith(('.tdms', '.segy')): 
                try: 
                    timestamp = datetime.strptime(file[-24:-9], '%Y%m%d_%H%M%S')
                except Exception: 
                    continue        # basically there was no datetime within the file
                if t_start and timestamp < t_start:
                    continue
                if t_end and timestamp > t_end:
                    continue
                files.append((os.path.join(dir_path, file), timestamp))
    files.sort(key=lambda x: x[1])
    reader_array, timestamps = zip(*files)
    return list(reader_array), np.asarray(timestamps, dtype=datetime)


def get_subset_paths(t0, dir_path, dir_list, timestamps, delta=timedelta(minutes=1)):
    tpf = (timestamps[1] - timestamps[0]).total_seconds()
    start_idx = get_closest_index(timestamps, t0)
    
    end_time = timestamps[start_idx] + delta - timedelta(seconds=tpf)
    end_idx = get_closest_index(timestamps, end_time)
    
    ### this is broken, doesn't account for the time in the last file
    # if (end_idx - start_idx + 1) != (delta.total_seconds()/tpf):
    #     warnings.warn(f"WARNING: time subset not continuous; only {(end_idx - start_idx + 1)*tpf} seconds represented.")
    
    dir_list = [dir_path + file for file in dir_list[start_idx:end_idx+1]]
    return dir_list, timestamps[start_idx:end_idx+1]


def is_valid_time(timestamp, allowed_times):
    '''checks if timestamp is within allowed times'''
    t = timestamp.time()
    for t1, t2 in allowed_times.items():
        if t1 < t2:
            if t1 <= t <= t2:
                return True
        else:
            if t1 <= t or t <= t2:
                return True
    return False


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
            if file.is_file() and file.path.endswith(('.tdms', '.segy')):
                file_path = file.path
                file_ext = '.' + file_path.rsplit('.', 1)[1]
                break
    match file_ext:
        case '.tdms': file_reader = TdmsReader(file_path)
        case '.segy': file_reader = SegyReader(file_path)
        case _: print("BAD FILE EXTENSION!!!!!!")
    file_reader._read_properties()
    return file_reader.get_properties()


# returns a delta-long array of reader files starting at the timestamp given
def get_time_subset(reader_array:np.ndarray, start_time:datetime, timestamps:np.ndarray, delta:timedelta=timedelta(seconds=60), tolerance:int=300):
    # tolerence is the time in s that the closest timestamp can be away from the desired start_time
    start_idx = get_closest_index(timestamps, start_time)
    if abs((start_time - timestamps[start_idx]).total_seconds()) > tolerance:
        warnings.warn(f"Error: first file ({timestamps[start_idx]}) is over {tolerance} seconds away from the given start time ({start_time}).")
    
    end_time = timestamps[start_idx] + delta
    print(f'end_time: {end_time}')
    end_idx = get_closest_index(timestamps, end_time)
    if (end_time - timestamps[end_idx]).total_seconds() > tolerance:
        warnings.warn(f"WARNING: end file ({timestamps[end_idx]}) is over {tolerance} seconds away from the calculated end time.")    
    ### this is broken, as above in get_subset_paths, doesn't account for the time in the last file
    # if (timestamps[end_idx] - timestamps[start_idx]).total_seconds() != (delta.total_seconds()):
    #     warnings.warn(f"WARNING: time subset not continuous; only {(timestamps[end_idx] - timestamps[start_idx]).total_seconds()} seconds represented.")
    return reader_array[start_idx:end_idx-1]


# returns a duration of data (default is 60s)
# def get_data_from_array(data_array:list, prepro_para:dict, start_time:datetime, timestamps:np.ndarray, duration:timedelta, channels=False):
#     cha1, cha2, sps, spatial_ratio = prepro_para.get('cha1'), prepro_para.get('cha2'), prepro_para.get('sps'), prepro_para.get('spatial_ratio')

#     # make it so that if start_time is not a timestamp, the first minute in the array is returned
#     current_time = 0
#     if channels: 
#         tdata = np.empty((int(duration.total_seconds() * sps), len(channels)))
#     else: 
#         tdata = np.empty((int(duration.total_seconds() * sps), ceil((cha2-cha1+1)/spatial_ratio)))
#     data_array = get_time_subset(data_array, start_time, timestamps, delta=duration, tolerance=30)
    
#     with tqdm(total=(duration.total_seconds()), desc=f'[Process {os.getpid()}] Loading data (in seconds)', position=1, leave=False) as pbar:
#         while current_time != duration.total_seconds() and len(data_array) != 0:
#             data_file = data_array.pop(0)
#             if type(data_file) == str:
#                 data_file = TdmsReader(data_file)
#             props = data_file.get_properties()
#             if channels:
#                 data = data_file.get_data(channels[0], channels[-1])
#                 data = data[:,np.array(channels)-channels[0]]
#             else:
#                 data = data_file.get_data(cha1, cha2)
#                 data = data[:, ::spatial_ratio]
#             data = scale(data, props)
#             if props.get('SamplingFrequency[Hz]') != sps:
#                 data = decimate(data,
#                         int(sps / props.get('SamplingFrequency[Hz]')),
#                         ftype='iir',
#                         zero_phase=True)
#             current_row = current_time * sps
#             t_size = data.shape[0]
#             tdata[int(current_row):int(current_row+(t_size)), :] = data
#             current_time += t_size/sps
            
#             # attempting to avoid issue of open file limit
#             data_file._tdms_file.close()
#             del data_file
#             gc.collect()
#             pbar.update(t_size/sps)
    
#     return tdata


# adapting this so that all channel requests work off of 1m channels - as though distance along cable
def get_data_from_array(data_array:list, prepro_para:dict, start_time:datetime, timestamps:np.ndarray, duration:timedelta, channels=False):
    cha1, cha2, target_sps, target_spatial_res, freqmin, freqmax = prepro_para.get('cha1'), prepro_para.get('cha2'), prepro_para.get('target_sps'), prepro_para.get('target_spatial_res'), prepro_para.get('freqmin'), prepro_para.get('freqmax')
    # make it so that if start_time is not a timestamp, the first minute in the array is returned
    current_time = 0
    if channels: 
        tdata = np.empty((int(duration.total_seconds() * target_sps), len(channels)))
    else: 
        tdata = np.empty((int(duration.total_seconds() * target_sps), ceil((cha2-cha1+1)/target_spatial_res)))
    tdata.fill(np.NaN)
    # data_array = get_time_subset(data_array, start_time, timestamps, delta=duration, tolerance=30)        # removed on 27/10/25 not needed anymore? 
    
    with tqdm(total=(duration.total_seconds()), desc=f'[Process {os.getpid()}] Loading data (in seconds)', position=1, leave=False) as pbar:
        while current_time != duration.total_seconds() and len(data_array) != 0:
            data_file = data_array.pop(0)
            if type(data_file) == str:
                try:
                    data_file = TdmsReader(data_file)
                except: 
                    print(f'Failed at {data_file}')
            props = data_file.get_properties()
            
            if props.get('GPSTimeStamp').replace(microsecond=0) != (start_time + timedelta(seconds=current_time)):
                # print(f'Padding from {(start_time + timedelta(seconds=current_time))} to {props.get("GPSTimeStamp").replace(microsecond=0)}')
                diff = (props.get('GPSTimeStamp').replace(microsecond=0) - (start_time + timedelta(seconds=current_time))).total_seconds()
                # current_row = current_time * target_sps
                # tdata[int(current_row):int(current_row+(diff*target_sps)), :] = np.NaN
                current_time += diff
                pbar.update(diff)
            spatial_ratio = int(target_spatial_res/props.get('SpatialResolution[m]'))
            if channels:
                file_channels = [int(channel*spatial_ratio) for channel in channels]
                data = data_file.get_data(file_channels[0], file_channels[-1])[:,np.array(file_channels)-file_channels[0]]
            else:
                cha1, cha2 = int(cha1*spatial_ratio), int(cha2*spatial_ratio)
                data = data_file.get_data(cha1, cha2)
                data = data[:, ::spatial_ratio]
            data = scale(data, props)
            if props.get('SamplingFrequency[Hz]') > target_sps:
                data = decimate(data,
                                int(props.get('SamplingFrequency[Hz]') / target_sps),
                                ftype='iir',
                                zero_phase=True, 
                                axis=0)
            elif props.get('SamplingFrequency[Hz]') < target_sps:
                warnings.warn(f"Sampling frequency below target frequency! Timestamp: {props.get('GPSTimeStamp')}; fs: {props.get('SamplingFrequency[Hz]')}")
            data = np.float32(bandpass(data,
                            0.9 * freqmin,
                            freqmax,
                            df=target_sps,
                            corners=4,
                            zerophase=True))
            current_row = int(current_time * target_sps)
            t_size = data.shape[0]
            if t_size > tdata.shape[0] - current_row:        # clips final file
                data = data[:tdata.shape[0]-current_row,:]
                t_size = data.shape[0]
            tdata[current_row:current_row+t_size, :] = data
            current_time += t_size/target_sps
            
            # attempting to avoid issue of open file limit
            data_file._tdms_file.close()
            del data_file
            gc.collect()
            pbar.update(t_size/target_sps)
    
    return tdata


def scale(data:np.ndarray, props:dict):
    """Takes in DAS data and its properties using them to scale the data as it is compressed within the file format. Returns scaled data

    strainrate nm/m/s = 116 * iDAS values * sampling freq (Hz) / gauge lenth (m)

    Keyword arguments:
        data -- numpy array containing DAS data
        props -- properties struct from DAS reader
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


def downsample_data(data:np.ndarray, props:dict, target_sps:float, target_spatial_res:float):
    sps = props.get('SamplingFrequency[Hz]')
    spatial_res = props.get('SpatialResolution[m]')
    
    if not target_sps: target_sps = sps
    if not target_spatial_res: target_spatial_res = spatial_res
    
    if (temporal_ratio := int(sps/target_sps)) != sps/target_sps:             # reversed as time-reciprocal
        warnings.warn(f'Target sps not a factor of current sps, some data will be lost. Resultant ratio cast to {temporal_ratio}.')
    if (spatial_ratio := int(target_spatial_res/spatial_res)) != target_spatial_res/spatial_res:
        warnings.warn(f'Target spatial res not a factor of current spatial res, some data will be lost. Resultant ratio cast to {spatial_ratio}.')
    return resize(data, output_shape=(data.shape[0] / temporal_ratio, data.shape[1] / spatial_ratio))


def downsample_tdms(file_path:str, save_as:str=None, out_dir:str=None, target_sps:int=None, target_spatial_res:int=None):
    if not file_path.endswith('.tdms'):
        warnings.warn(f'Not TDMS file! Use other downsampler please.')
        return
    tdms = TdmsReader(file_path)
    props = tdms.get_properties()
    data = tdms.get_data()
    data = scale(data, props)
    
    if target_sps or target_spatial_res:
        data = downsample_data(data, props, target_sps, target_spatial_res)
        if target_sps:         props.update({'SamplingFrequency[Hz]': target_sps})
        if target_spatial_res: props.update({'SpatialResolution[m]': target_spatial_res})
    
    # NOTE: This is starting as just one Stats object for ALL traces in the stream, this may have to change
    stats = Stats({
        'network': 'DS',
        'sampling_rate': props.get('SamplingFrequency[Hz]'),
        'npts': data.shape[0],
        'starttime': UTCDateTime(props.get('GPSTimeStamp')),
    })
    stats.update(props)
    
    stream = Stream()
    for count, channel in enumerate(data.T):
        stats.station = str(count)
        trace = Trace(channel, stats)
        if save_as in ['SEGY', 'SU']:
            trace.data = np.require(trace.data, dtype=np.float32)
        trace.data = np.ascontiguousarray(trace.data)
        stream += trace
    
    if save_as:
        out_name = os.path.splitext(file_path.split('/')[-1])[0] + '.' + save_as.lower()
        if out_dir: out_name = out_dir + out_name
        stream.write(out_name, format=save_as)


def max_min_strain_rate(data:np.ndarray, channel_bounds:list=None):
    """Takes in a 2D numpy array of DAS data and returns the min and max values

    Keyword arguments:
        data -- A numpy array containing DAS data
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


def nparray_to_obspy(data:np.ndarray, props:dict):
    stats = Stats({
        'network': 'DS',
        'sampling_rate': props.get('SamplingFrequency[Hz]'),
        'npts': data.shape[0],
        'starttime': UTCDateTime(props.get('GPSTimeStamp')),
    })
    stats.update(props)
    
    stream = Stream()
    for count, channel in enumerate(data.T):
        stats.station = str(count)
        trace = Trace(channel, stats)
        trace.data = np.ascontiguousarray(trace.data)
        stream += trace
    
    return stream


def load_tdms(dir_path, n_minute):
    task_t0 = datetime(year = 2023, month = 11, day = 9, 
                       hour = 13, minute = 41, second = 17)
    
    properties = get_dir_properties(dir_path)
    prepro_para = {
        'cha1': 3850,
        'cha2': 5750,
        'sps': properties.get('SamplingFrequency[Hz]'),
        'spatial_res': properties.get('SpatialResolution[m]'),
        'spatial_ratio': 1,          # int(target_spatial_res/spatial_res), set not to downsample rn
        'n_minute': 1,
        'freqmax': 49.9,
        'freqmin': 1,
    }
    
    reader_array, timestamps = get_reader_array(dir_path)
    return get_data_from_array(reader_array, prepro_para, task_t0, timestamps, n_minute), prepro_para


def load_xcorr(file_path, normalise=False, as_stream=False, chas=None):
    xdata = np.loadtxt(file_path, delimiter=',')
    xdata = xdata[:, ~np.all(np.isnan(xdata), axis=0)]      # 06/01/25 added for Ni's data
    if normalise:
        xdata = xdata/np.sqrt(np.sum(xdata**2))
    if as_stream:
        stream = Stream()
        stats = Stats()
        stats.delta = 1/100
        stats.npts = 801
        if chas: 
            for cha in chas:
                stream.append(Trace(xdata[:, cha], stats))
        else:
            for i in range(0, xdata.shape[1]):
                stream.append(Trace(xdata[:, i], stats))
        return stream
    if chas:
        return xdata[:, chas].T
    return xdata.T


if __name__ == '__main__':
    ### downsample test for HPC data: 
    # dir_path = "../../../../gpfs/data/DAS_data/30mins/"
    # out_dir = os.path.join(dir_path, 'segys/')
    # dir_list = os.listdir(os.fsencode(dir_path))
    
    # pbar = tqdm(range(len(dir_list)))

    # props_bool = False      # boolean to only export properties once
    # for file_idx in pbar:
    #     file_path = os.path.join(dir_path, os.fsdecode(dir_list[file_idx]))
    #     if not os.path.isfile(file_path): continue
        
    #     if not props_bool:
    #         tdms = TdmsReader(file_path)
    #         file_info = tdms.fileinfo
    #         tdms._read_properties()
    #         properties = tdms.get_properties()

    #         with open(os.path.join(out_dir, 'properties.p'), 'wb') as prop_path:
    #             pickle.dump(properties, prop_path)
    
    #         with open(os.path.join(out_dir, 'fileinfo.p'), 'wb') as file_info_path:
    #             pickle.dump(file_info, file_info_path)
    #         props_bool = True
    
    #     downsample_tdms(file_path, save_as='SEGY', out_dir=out_dir, target_sps=None, target_spatial_res=1)

    dir_path = '../../temp_data_store/FirstData/'
    
    times = {dt_time(13, 41, 10):dt_time(13, 41, 39), 
             dt_time(13, 42, 20):dt_time(13, 42, 39), }
    readers, timestamps = get_reader_array(dir_path, times)
    print(timestamps)
