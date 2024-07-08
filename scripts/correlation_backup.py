suppress_figures = True

# attach local resources
import sys
sys.path.append("./src")
sys.path.append("./DASstore")

import os
import time
from math import floor
from datetime import datetime, timedelta
from dateutil.parser import parse
from bisect import bisect_left

import h5py
import numpy as np
import DAS_module
import matplotlib.pyplot as plt
import matplotlib as mpl
import obspy
from tqdm import tqdm

from dasstore.zarr import Client
from TDMS_Read import TdmsReader


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

# returns a minute-long array of tdms files starting at the timestamp given
def get_time_subset(tdms_array, start_time, timestamps, tpf, delta=timedelta(seconds=60), tolerance=300):
    # tolerence is the time in s that the closest timestamp can be away from the desired start_time
    # timestamps MUST be orted, and align with TDMS array (i.e. timestamps[n] represents tdms_array[n]
    start_idx = get_closest_index(timestamps, start_time)
    if abs((start_time - timestamps[start_idx]).total_seconds()) > tolerance:
        print(f"WARNING: first tdms is over {tolerance} seconds away from the given start time.")
    
    end_time = timestamps[start_idx] + delta - timedelta(seconds=tpf)
    end_idx = get_closest_index(timestamps, end_time)
    if (end_time - timestamps[end_idx]).total_seconds() > tolerance:
        print(f"WARNING: end tdms is over {tolerance} seconds away from the calculated end time.")
    # print(f"Given t={start_time}, snippet selected from {timestamps[start_idx]} to {timestamps[end_idx]}!")
    
    if (end_idx - start_idx + 1) != (delta.seconds/tpf):
        print(f"WARNING: time subset not continuous; only {(end_idx - start_idx + 1)*tpf} seconds represented.")
    
    return tdms_array[start_idx:end_idx+1]

# returns a minute of data
def get_minute_data(tdms_array, channels, start_time, timestamps, sps):
    # make it so that if start_time is not a timestamp, the first minute in the array is returned
    
    current_time = 0
    tdms_t_size = tdms_array[0].get_data(channels[0], channels[1]).shape[0]
    minute_data = np.empty((tdms_t_size * 6, floor((cha2-cha1+1)/spatial_ratio)))
    
    if type(start_time) is datetime:
        tdms_array = get_time_subset(tdms_array, start_time, timestamps, tpf=tdms_t_size/sps, tolerance=30)   # tpf = time per file
    
    while current_time != 60 and len(tdms_array) != 0:
        data = tdms_array.pop(0).get_data(cha1, cha2)
        data = data[:, ::spatial_ratio]
        current_row = current_time * sps
        minute_data[int(current_row):int(current_row+(tdms_t_size)), :] = data
        current_time += tdms_t_size/sps
    
    return minute_data

def plot_das_data(data):
    plt.figure(figsize = (12, 5), dpi = 150)
    plt.imshow(data, aspect = 'auto', 
               cmap = 'RdBu', vmax = 1.5, vmin = -1.5, origin='lower')
    _=plt.xticks(np.linspace(cha1, cha2, 9) - cha1, 
                  [int(i) for i in np.linspace(cha1, cha2, 9)], fontsize = 12)
    _=plt.yticks(np.linspace(0, 60000, 7), 
                  [int(i) for i in np.linspace(0, 60, 7)], fontsize = 12)
    plt.xlabel("Channel number", fontsize = 15)
    plt.ylabel("Time (s)", fontsize = 15)

    twinx = plt.gca().twiny()
    twinx.set_xticks(np.linspace(0, 2000, 9),
                     [int(i*cha_spacing) for i in np.linspace(cha1, cha2, 9)])
    # twinx.set_xticks(np.linspace(cha1*cha_spacing, cha2*cha_spacing, 9),
    #                  [int(i*cha_spacing) for i in np.linspace(cha1, cha2, 9)])
    twinx.set_xlabel("Distance along cable (m)", fontsize=15)
    plt.colorbar(pad = 0.1)
    
def plot_correlation(corr, cmap_param='bwr'):
    plt.figure(figsize = (12, 5), dpi = 150)

    plt.imshow(corr[:, :(cha2 - cha1)].T, aspect = 'auto', cmap = cmap_param, 
            vmax = 2e-2, vmin = -2e-2, origin = 'lower', interpolation=None)

    _ =plt.yticks(np.linspace(cha1, cha2, 4) - cha1, 
                [int(i) for i in np.linspace(cha1, cha2, 4)], fontsize = 12)
    plt.ylabel("Channel number", fontsize = 16)
    _ = plt.xticks(np.arange(0, 1601, 200), (np.arange(0, 801, 100) - 400)/50, fontsize = 12)
    plt.xlabel("Time lag (sec)", fontsize = 16)
    bar = plt.colorbar(pad = 0.1, format = lambda x, pos: '{:.1f}'.format(x*100))
    bar.set_label('Cross-correlation Coefficient ($\\times10^{-2}$)', fontsize = 15)

    twiny = plt.gca().twinx()
    twiny.set_yticks(np.linspace(0, cha2 - cha1, 4), 
                                [int(i* cha_spacing) for i in np.linspace(cha1, cha2, 4)])
    twiny.set_ylabel("Distance along cable (m)", fontsize = 15)
    
    # follow convention of: {timestamp}_{t length}_{channels}.png
    t_start = task_t0 - timedelta(minutes=n_minute)
    plt.savefig(f'./results/figures/{task_t0}_{n_minute}-mins_{cha1}:{cha2}__{cmap_param}.png')

file_path = "../../scratch/DAS_data/Second_Survey_UTC_20240119_151907.161.tdms"
tdms_file = TdmsReader(file_path)
tdms_file._read_properties()

# subsets based on channels can be done later
n_channels = tdms_file.fileinfo['n_channels']
cha1, cha2 = 0, n_channels
properties = tdms_file.get_properties()

# print(properties)

cha_spacing = properties.get('SpatialResolution[m]') * properties.get('Fibre Length Multiplier')
start_dist, stop_dist = properties.get('Start Distance (m)'), properties.get('Stop Distance (m)')
sps = properties.get('SamplingFrequency[Hz]')
spatial_res = properties.get('SpatialResolution[m]')

t_start = properties.get('GPSTimeStamp')
time_delta = timedelta(seconds=tdms_file.channel_length / sps)
t_end = t_start + time_delta

sps                = properties.get('SamplingFrequency[Hz]')        # current sampling rate (Hz)
samp_freq          = 100                                            # target sampling rate (Hz)
target_spatial_res = 0.25                                             # target spatial resolution (m)
spatial_ratio      = int(target_spatial_res/spatial_res)
freqmin            = 1                                              # pre filtering frequency bandwidth
freqmax            = 49.9                                           # note this cannot exceed Nquist freq

freq_norm          = 'rma'             # 'no' for no whitening, or 'rma' for running-mean average, 'phase_only' for sign-bit normalization in freq domain.
time_norm          = 'one_bit'         # 'no' for no normalization, or 'rma', 'one_bit' for normalization in time domain
cc_method          = 'xcorr'           # 'xcorr' for pure cross correlation, 'deconv' for deconvolution; FOR "COHERENCY" PLEASE set freq_norm to "rma", time_norm to "no" and cc_method to "xcorr"
smooth_N           = 100               # moving window length for time domain normalization if selected (points)
smoothspect_N      = 100               # moving window length to smooth spectrum amplitude (points)
maxlag             = 8                 # lags of cross-correlation to save (sec)

# criteria for data selection
max_over_std       = 10                # threshold to remove window of bad signals: set it to 10*9 if prefer not to remove them

cc_len             = 180               # correlate length in second
step               = 90                # stepping length in second

cha1, cha2         = 4000, 5999        # USE ONLY FOR CHANNEL SUBSET SELECTION (repeated later)

cha_list = np.array(range(cha1, cha2+1, spatial_ratio))
nsta = len(cha_list)
n_pair = int((nsta+1)*nsta/2)
n_lag = maxlag * samp_freq * 2 + 1

prepro_para = {'freqmin':freqmin,
               'freqmax':freqmax,
               'sps':sps,
               'npts_chunk':cc_len*sps,           # <-- I don't know why this is hard coded like this
               'nsta':nsta,
               'cha_list':cha_list,
               'samp_freq':samp_freq,
               'freq_norm':freq_norm,
               'time_norm':time_norm,
               'cc_method':cc_method,
               'smooth_N':smooth_N,
               'smoothspect_N':smoothspect_N,
               'maxlag':maxlag,
               'max_over_std':max_over_std}


corr_full = np.zeros([n_lag, n_pair], dtype = np.float32)
stack_full = np.zeros([1, n_pair], dtype = np.int32)

print(f"Shape of corr_full: {corr_full.shape}; shape of stack_full: {stack_full.shape}")

# refresh data bc of popping and stuff
task_t0 = datetime(year = 2024, month = 1, day = 19, 
                   hour = 15, minute = 19, second = 7, microsecond = 0)

dir_path = "../../scratch/DAS_data/"
tdms_array = np.empty(len(os.listdir(dir_path)), TdmsReader)
timestamps = np.empty(len(tdms_array), dtype=datetime)

for count, file in enumerate(os.listdir(dir_path)):
    if file.endswith('.tdms'):
        tdms = TdmsReader(dir_path + file)
        tdms_array[count] = tdms
        timestamps[count] = tdms.get_properties().get('GPSTimeStamp')
timestamps.sort()
print(f'{len(timestamps)} files available from {timestamps[0]} to {timestamps[-1]}')

tdms_array = [x for y, x in sorted(zip(np.array(timestamps), tdms_array))]

# each task is one minute
n_minute = 120
pbar = tqdm(range(n_minute))
t_query = 0; t_compute = 0
for imin in pbar:
    t0 = time.time()
    pbar.set_description(f"Processing {task_t0}")
    tdata = get_minute_data(tdms_array, [cha1, cha2], task_t0, timestamps, sps)
    task_t0 += timedelta(minutes = 1)
    
    t_query += time.time() - t0
    t1 = time.time()
    # perform pre-processing
    trace_stdS, dataS = DAS_module.preprocess_raw_make_stat(tdata, prepro_para)

    # do normalization if needed
    white_spect = DAS_module.noise_processing(dataS, prepro_para)
    Nfft = white_spect.shape[1]; Nfft2 = Nfft // 2
    data = white_spect[:, :Nfft2]
    del dataS, white_spect

    ind = np.where((trace_stdS < prepro_para['max_over_std']) &
                            (trace_stdS > 0) &
                    (np.isnan(trace_stdS) == 0))[0]
    if not len(ind):
        raise ValueError('the max_over_std criteria is too high which results in no data')
    sta = cha_list[ind]
    white_spect = data[ind]

    # loop over all stations
    for iiS in range(len(sta)):
        # smooth the source spectrum
        sfft1 = DAS_module.smooth_source_spect(white_spect[iiS], prepro_para)
        
        # correlate one source with all receivers
        corr, tindx = DAS_module.correlate(sfft1, white_spect[iiS:], prepro_para, Nfft)

        # update the receiver list
        tsta = sta[iiS:]
        receiver_lst = tsta[tindx]

        iS = int((cha2*2 - cha1 - sta[iiS] + 1) * (sta[iiS] - cha1) / 2)

        # stacking one minute
        corr_full[:, iS + receiver_lst - sta[iiS]] += corr.T
        stack_full[:, iS + receiver_lst - sta[iiS]] += 1
        
    t_compute += time.time() - t1
corr_full /= stack_full
# print("%.3f seconds in data query, %.3f seconds in xcorr computing" % (t_query, t_compute))
print(f"{round(t_query, 2)} seconds in data query, {round(t_compute, 2)} seconds in xcorr computing")

plot_correlation(corr_full)
