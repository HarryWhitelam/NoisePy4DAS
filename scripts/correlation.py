suppress_figures = True

# attach local resources
import sys
sys.path.append("./src")
sys.path.append("./DASstore")

import os
import time
from math import floor, ceil
from datetime import datetime, timedelta
from dateutil.parser import parse
from bisect import bisect_left

import h5py
import numpy as np
import DAS_module
import matplotlib.pyplot as plt
import matplotlib as mpl
import obspy
import pyasdf
from tqdm import tqdm

from TDMS_Read import TdmsReader


def get_tdms_array(dir_path):
    tdms_array = np.empty(int(len([filename for filename in os.listdir(dir_path) if filename.endswith(".tdms")])), dtype=TdmsReader)
        # tdms_array = np.empty(len(os.listdir(dir_path)), TdmsReader)
    timestamps = np.empty(len(tdms_array), dtype=datetime)

    for count, file in enumerate(os.listdir(dir_path)):
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
    cha1, cha2, sps, spatial_ratio, duration = prepro_para.get('cha1'), prepro_para.get('cha2'), prepro_para.get('sps'), prepro_para.get('spatial_ratio'), prepro_para.get('cc_len')

    # make it so that if start_time is not a timestamp, the first minute in the array is returned
    current_time = 0
    tdms_t_size = tdms_array[0].get_data(cha1, cha2).shape[0]
    minute_data = np.empty((int(duration * sps), floor((cha2-cha1+1)/spatial_ratio)))
    
    if type(start_time) is datetime:
        tdms_array = get_time_subset(tdms_array, start_time, timestamps, tpf=tdms_t_size/sps, delta=timedelta(seconds=duration), tolerance=30)   # tpf = time per file
    
    while current_time != duration and len(tdms_array) != 0:
        data = tdms_array.pop(0).get_data(cha1, cha2)
        data = data[:, ::spatial_ratio]
        current_row = current_time * sps
        minute_data[int(current_row):int(current_row+(tdms_t_size)), :] = data
        current_time += tdms_t_size/sps
    
    return minute_data


def get_dir_properties(dir_path):
    with os.scandir(dir_path) as files:
        for file in files:
            if file.is_file():
                file_path = file.path
                break
    tdms_file = TdmsReader(file_path)
    tdms_file._read_properties()
    return tdms_file.get_properties()


def set_prepro_parameters(dir_path, freqmin=1, freqmax=49.9):
    properties = get_dir_properties(dir_path)
    
    cha_spacing = properties.get('SpatialResolution[m]') * properties.get('Fibre Length Multiplier')
    # start_dist, stop_dist = properties.get('Start Distance (m)'), properties.get('Stop Distance (m)')

    sps                = properties.get('SamplingFrequency[Hz]')        # current sampling rate (Hz)
    samp_freq          = 100                                            # target sampling rate (Hz)
    
    spatial_res = properties.get('SpatialResolution[m]')
    target_spatial_res = 1                                              # target spatial resolution (m)
    spatial_ratio      = int(target_spatial_res/spatial_res)

    # freqmin: pre filtering frequency bandwidth
    # freqmax: note this cannot exceed Nquist freq
    freq_norm          = 'rma'             # 'no' for no whitening, or 'rma' for running-mean average, 'phase_only' for sign-bit normalization in freq domain.
    time_norm          = 'one_bit'         # 'no' for no normalization, or 'rma', 'one_bit' for normalization in time domain
    cc_method          = 'xcorr'           # 'xcorr' for pure cross correlation, 'deconv' for deconvolution; FOR "COHERENCY" PLEASE set freq_norm to "rma", time_norm to "no" and cc_method to "xcorr"
    smooth_N           = 100               # moving window length for time domain normalization if selected (points)
    smoothspect_N      = 100               # moving window length to smooth spectrum amplitude (points)
    maxlag             = 4                 # lags of cross-correlation to save (sec)

    max_over_std       = 20                # threshold to remove window of bad signals: set it to 10*9 if prefer not to remove them

    cc_len             = 60                # correlate length in second
    # step               = 60                # stepping length in second [not used]

    cha1, cha2         = 6000, 7999        # USE ONLY FOR CHANNEL SUBSET SELECTION 
    effective_cha2     = floor(cha1 + (cha2 - cha1) / spatial_ratio)

    cha_list = np.array(range(cha1, effective_cha2 + 1))
    nsta = len(cha_list)
    n_pair = int((nsta+1)*nsta/2)
    n_lag = maxlag * samp_freq * 2 + 1
    
    return {
        'freqmin':freqmin,
        'freqmax':freqmax,
        'sps':sps,
        'cc_len':cc_len,
        'npts_chunk':cc_len*sps,
        'nsta':nsta,
        'cha_list':cha_list,
        'samp_freq':samp_freq,
        'freq_norm':freq_norm,
        'time_norm':time_norm,
        'cc_method':cc_method,
        'smooth_N':smooth_N,
        'smoothspect_N':smoothspect_N,
        'maxlag':maxlag,
        'max_over_std':max_over_std,
        'cha1':cha1,
        'cha2':cha2,
        'effective_cha2':effective_cha2,
        'cha_spacing':cha_spacing,
        'spatial_ratio':spatial_ratio, 
        'n_pair':n_pair,
        'n_lag':n_lag
    }


def correlation(tdms_array, prepro_para, timestamps, task_t0, save_ccf=False):
    n_lag, n_pair, cha1,  effective_cha2, cha_list, cc_len, nsta = prepro_para.get('n_lag'), prepro_para.get('n_pair'), prepro_para.get('cha1'), prepro_para.get('effective_cha2'), prepro_para.get('cha_list'), prepro_para.get('cc_len'), prepro_para.get('nsta')
    
    corr_full = np.zeros([n_lag, n_pair], dtype = np.float32)
    stack_full = np.zeros([1, n_pair], dtype = np.int32)
    
    pbar = tqdm(range(n_minute))
    t_query = 0; t_compute = 0

    for imin in pbar:
        t0 = time.time()
        pbar.set_description(f"Processing {task_t0}")
        tdata = get_data_from_array(tdms_array, prepro_para, task_t0, timestamps)
        task_t0 += timedelta(seconds=cc_len)
        
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

            iS = int((effective_cha2*2 - cha1 - sta[iiS] + 1) * (sta[iiS] - cha1) / 2)

            # print(f'iiS: {iiS}; iS: {iS}; sta[iiS]: {sta[iiS]}; corr_full idx: {iS + receiver_lst - sta[iiS]}')
            # print(f'iiS: {iiS}; iS: {iS}; sta[iiS]: {sta[iiS]}')
            # stacking one minute
            corr_full[:, iS + receiver_lst - sta[iiS]] += corr.T
            stack_full[:, iS + receiver_lst - sta[iiS]] += 1
            
        t_compute += time.time() - t1
    corr_full /= stack_full
    # print("%.3f seconds in data query, %.3f seconds in xcorr computing" % (t_query, t_compute))
    print(f"{round(t_query, 2)} seconds in data query, {round(t_compute, 2)} seconds in xcorr computing")
    
    if save_ccf:
        save_ccf(corr_full, sta, nsta)
    return corr_full, stack_full


def plot_das_data(data, prepro_para):
    cha1, cha2, effective_cha2, cha_spacing = prepro_para.get('cha1'), prepro_para.get('cha2'), prepro_para.get('effective_cha2'), prepro_para.get('cha_spacing')
    
    plt.figure(figsize = (12, 5), dpi = 150)
    plt.imshow(data, aspect = 'auto', 
               cmap = 'RdBu', vmax = 1.5, vmin = -1.5, origin='lower')
    _=plt.xticks(np.linspace(cha1, effective_cha2, 9) - cha1, 
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


def plot_correlation(corr, prepro_para, cmap_param='bwr'):
    cha1, cha2, effective_cha2, spatial_ratio, cha_spacing, freqmin, freqmax, target_spatial_res, maxlag = prepro_para.get('cha1'), prepro_para.get('cha2'), prepro_para.get('effective_cha2'), prepro_para.get('spatial_ratio'), prepro_para.get('cha_spacing'), prepro_para.get('freqmin'), prepro_para.get('freqmax'), prepro_para.get('target_spatial_res'), prepro_para.get('maxlag')

    plt.figure(figsize = (12, 5), dpi = 150)
    plt.imshow(corr[:, :(effective_cha2 - cha1)].T, aspect = 'auto', cmap = cmap_param, 
            vmax = 2e-2, vmin = -2e-2, origin = 'lower', interpolation=None)

    _ =plt.yticks((np.linspace(cha1, cha2, 4) - cha1)/spatial_ratio, 
                [int(i) for i in np.linspace(cha1, cha2, 4)], fontsize = 12)
    plt.ylabel("Channel number", fontsize = 16)
    # _ = plt.xticks(np.arange(0, 1601, 200), (np.arange(0, 801, 100) - 400)/50, fontsize = 12)
    _ = plt.xticks(np.arange(0, maxlag*200+1, 200), np.arange(-maxlag, maxlag+1, 2), fontsize=12)
    plt.xlabel("Time lag (sec)", fontsize = 16)
    bar = plt.colorbar(pad = 0.1, format = lambda x, pos: '{:.1f}'.format(x*100))
    bar.set_label('Cross-correlation Coefficient ($\\times10^{-2}$)', fontsize = 15)

    twiny = plt.gca().twinx()
    twiny.set_yticks(np.linspace(0, cha2 - cha1, 4), 
                                [int(i* cha_spacing) for i in np.linspace(cha1, cha2, 4)])
    twiny.set_ylabel("Distance along cable (m)", fontsize = 15)
    
    t_start = task_t0 - timedelta(minutes=n_minute)
    plt.savefig(f'./results/figures/{t_start}_{n_minute}mins_{freqmin}:{freqmax}Hz__{cha1}:{cha2}_{target_spatial_res}m.png')


def save_ccf(corr_full, sta, nsta):
    cha1, cha2, samp_freq, freqmin, freqmax, maxlag, target_spatial_res = prepro_para.get('cha1'), prepro_para.get('cha2'), prepro_para.get('samp_freq'), prepro_para.get('freqmin'), prepro_para.get('freqmax'), prepro_para.get('maxlag'), prepro_para.get('target_spatial_res')
    
    t0 = time.time()
    with pyasdf.ASDFDataSet(f"./results/CCF/{t_start}_{n_minute}-mins_{cha1}:{cha2}_{target_spatial_res}.h5", mpi = False) as ccf_ds:
        for iiS in tqdm(range(len(sta))):
            for iiR in range(nsta - iiS):
                # use the channel number as a way to figure out distance
                Sindx = iiS + cha1
                iS = int((cha2*2 - cha1 - Sindx + 1) * (Sindx - cha1) / 2)
                param = {'sps':samp_freq,
                        'dt': 1/samp_freq,
                        'maxlag':maxlag,
                        'freqmin':freqmin,
                        'freqmax':freqmax,
                        'dist':target_spatial_res}

                # source-receiver pair
                data_type = str(sta[iiS])
                path = f'{Sindx}_{Sindx + iiR}'
                ccf_ds.add_auxiliary_data(data=corr_full[:, iS + iiR], 
                                        data_type=data_type, 
                                        path=path, 
                                        parameters=param)
    t1 = time.time()
    print(f"it takes %.3f seconds to write this ASDF file" % (t1 - t0))


dir_path = "../../../../gpfs/data/DAS_data/Data/"
# dir_path = "../../temp_data_store/"
task_t0 = datetime(year = 2023, month = 11, day = 9, 
                   hour = 13, minute = 42, second = 57)
n_minute = 4
t_start = task_t0 - timedelta(minutes=n_minute)

prepro_para = set_prepro_parameters(dir_path)
tdms_array, timestamps = get_tdms_array()

corr_full, stack_full = correlation(tdms_array, prepro_para, timestamps, task_t0, save_ccf=True)
plot_correlation(corr_full, prepro_para)
