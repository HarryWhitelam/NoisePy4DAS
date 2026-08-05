suppress_figures = True

# attach local resources
import sys
sys.path.append("./src")
sys.path.append("./DASstore")

import os
from warnings import warn
import time
from math import floor
from datetime import datetime, timedelta

import numpy as np
import DAS_module
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import multiprocessing
import gc
from memory_profiler import profile

# from dasstore.zarr import Client
from TDMS_Read import TdmsReader
from tdms_io import get_reader_array, get_filepath_array, get_data_from_array, get_dir_properties, get_subset_paths


def set_prepro_parameters(dir_path, task_t0, freqmin=1.0, freqmax=49.9, target_spatial_res=5, cha1=4000, cha2=7999, n_minute=360, src_ch=None, rcv_ch=None, stack_method='linear'):
    properties = get_dir_properties(dir_path)
    
    cha_spacing = properties.get('SpatialResolution[m]') * properties.get('Fibre Length Multiplier')
    # start_dist, stop_dist = properties.get('Start Distance (m)'), properties.get('Stop Distance (m)')

    sps             = properties.get('SamplingFrequency[Hz]')       # current sampling rate (Hz)
    samp_freq       = 100                                           # target sampling rate (Hz)
    
    spatial_res     = properties.get('SpatialResolution[m]')
    spatial_ratio   = int(target_spatial_res/spatial_res)           # both values in m

    time_norm       = 'rma'                 # 'no' for no normalization, or 'rma', 'one_bit' for normalization in time domain
    freq_norm       = 'rma'                 # 'no' for no whitening, or 'rma' for running-mean average, 'phase_only' for sign-bit normalization in freq domain.
    cc_method       = 'xcorr'               # 'xcorr' for pure cross correlation, 'deconv' for deconvolution; FOR "COHERENCY" PLEASE set freq_norm to "rma", time_norm to "no" and cc_method to "xcorr"
    stack_method    = stack_method          # 'pws' for phase-weighted stack, 'robust' for robust stack, 'linear' for linear stack
    chunk_size      = 360                   # chunk size in minutes
    
    smooth_N        = int(0.5*samp_freq)    # moving window length for time domain normalization if selected (points)
    smoothspect_N   = 100                   # moving window length to smooth spectrum amplitude (points)
    maxlag          = 4                     # lags of cross-correlation to save (sec)
    n_lag           = maxlag * samp_freq * 2 + 1

    max_over_std    = 10                    # threshold to remove window of bad signals: set it to 10*9 if prefer not to remove them

    cc_len          = 60                    # correlate length in second
    # step            = 60                  # stepping length in second [not used]

    if not rcv_ch:
        effective_cha2  = floor(cha1 + (cha2 - cha1) / spatial_ratio)
        cha_list        = np.array(range(cha1, effective_cha2 + 1))
        nsta            = len(cha_list)
        n_pair          = int((nsta+1)*nsta/2) if src_ch == None else nsta
    else:
        effective_cha2  = 1
        cha_list        = np.array([cha1, cha2])
        nsta            = 2
        n_pair          = 1
    
    return {
        'freqmin':freqmin,
        'freqmax':freqmax,
        'sps':sps,
        'npts_chunk':cc_len*samp_freq,           # <-- I don't know why this is hard coded like this i should probably find out
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
        'target_spatial_res':target_spatial_res,
        'spatial_ratio':spatial_ratio, 
        'n_pair':n_pair,
        'n_lag':n_lag,
        'n_minute':n_minute,
        'task_t0':task_t0,
        'src_ch':src_ch,
        'rcv_ch':rcv_ch,
        'stack_method': stack_method,   # 'linear' 'robust', or 'pws'
        'pws_exponent': 0.3,      # ν in |<e^{iφ}>|^ν
        'chunk_size': chunk_size,
    }


try:
    from scipy.signal import hilbert
    def analytic_signal(x, axis=0):
        return hilbert(x, axis=axis)
except Exception:
    def analytic_signal(x, axis=0):
        # Numpy-only analytic signal along `axis`
        x = np.asarray(x)
        n = x.shape[axis]
        X = np.fft.rfft(x, n=n, axis=axis)
        # Build the RFFT "Hilbert" multiplier
        H = np.zeros_like(X, dtype=np.float32)
        if n % 2 == 0:
            H[0] = 1.0
            H[1:-1] = 2.0
            H[-1] = 1.0  # Nyquist
        else:
            H[0] = 1.0
            H[1:] = 2.0
        Xh = X * H
        z = np.fft.irfft(Xh, n=n, axis=axis)
        return z + 1j * np.imag(z)


def phase_weighted_stack(data, nu=2):
    z = analytic_signal(data, axis=1)
    phase = z / (np.abs(z) + 1e-10)

    phase_stack = np.abs(np.mean(phase, axis=0)) ** nu
    linear_stack = np.mean(data, axis=0)

    return linear_stack * phase_stack


def robust_stack(data, n_iter=10, eps=1e-6, tol=1e-5):
    stack = np.mean(data, axis=0)
    for _ in range(n_iter):
        prev_stack = stack.copy()
        
        residuals = np.abs(data - stack)
        r = np.sum(residuals, axis=1)

        weights = 1.0 / (r + eps)
        weights /= np.sum(weights)

        stack = np.sum(data * weights[:, None], axis=0)
        if np.linalg.norm(stack - prev_stack) < tol:
            break

    return stack


def process_minute(args):
    (minute_t0, file_paths, prepro_para) = args
    n_lag           = prepro_para.get('n_lag')
    n_pair          = prepro_para.get('n_pair')
    cha1            = prepro_para.get('cha1')
    effective_cha2  = prepro_para.get('effective_cha2')
    spatial_ratio   = prepro_para.get('spatial_ratio')
    cha_list        = prepro_para.get('cha_list')
    src_ch          = prepro_para.get('src_ch')
    
    # file_array = [TdmsReader(path) for path in file_paths]
    tdata = get_data_from_array(file_paths, prepro_para, minute_t0, duration=timedelta(seconds=60), filter=False)
    trace_stdS, dataS = DAS_module.preprocess_raw_make_stat(tdata, prepro_para)
    white_spect = DAS_module.noise_processing(dataS, prepro_para)
    Nfft = white_spect.shape[1]; Nfft2 = Nfft // 2
    data = white_spect[:, :Nfft2]
    del file_paths, tdata, dataS, white_spect
    gc.collect()

    ind = np.where((trace_stdS < prepro_para['max_over_std']) &
                   (trace_stdS > 0) &
                   (np.isnan(trace_stdS) == 0))[0]
    if not len(ind):
        warn(f'{minute_t0} had no valid indices :(')
        corr_zero = np.zeros((n_lag, n_pair), dtype=np.float32)
        return corr_zero

    sta = cha_list[ind]
    white_spect = data[ind]

    corr_full = np.zeros([n_lag, n_pair], dtype=np.float32)
    stack_full = np.zeros([1, n_pair], dtype=np.int32)

    if src_ch:
        sfft1 = DAS_module.smooth_source_spect(data[int((src_ch - cha1)/spatial_ratio)], prepro_para)
        # rcv_idx = int((prepro_para["rcv_ch"] - cha1) / spatial_ratio)
        corr, tindx = DAS_module.correlate(sfft1,data[1:],prepro_para,Nfft)
        corr = corr.T
        corr_full[:, :] += corr
        stack_full[:, :] += 1
    else:
        for iiS in range(len(sta)):
            sfft1 = DAS_module.smooth_source_spect(white_spect[iiS], prepro_para)
            corr, tindx = DAS_module.correlate(sfft1, white_spect[iiS:], prepro_para, Nfft)
            corr = corr.T
            tsta = sta[iiS:]
            receiver_lst = tsta[tindx]
            iS = int((effective_cha2*2 - cha1 - sta[iiS] + 1) * (sta[iiS] - cha1) / 2)
            sl = iS + receiver_lst - sta[iiS]
            corr_full[:, sl] += corr
            stack_full[:, sl] += 1
    
    # subtract median
    corr_full = corr_full - np.median(corr_full, axis=0)

    valid = stack_full > 0
    ccf = np.zeros_like(corr_full)
    ccf[:, valid[0]] = corr_full[:, valid[0]] / stack_full[:, valid[0]]
        
    return ccf


def parallel_xcorr(dir_path, prepro_para, corr_path=None):
    n_lag        = prepro_para['n_lag']
    n_pair       = prepro_para['n_pair']
    n_minute     = prepro_para['n_minute']
    task_t0      = prepro_para['task_t0']
    stack_method = prepro_para['stack_method']
    pws_exponent = prepro_para['pws_exponent']
    chunk_size   = prepro_para['chunk_size']

    try:
        file_paths, timestamps = get_filepath_array(dir_path, task_t0, task_t0+timedelta(minutes=n_minute))
    except: 
        return 0
    task_t0 = timestamps[0].replace(microsecond=0)
    
    # Prepare argument list for each minute
    args_list = []
    pbar = tqdm(range(n_minute))
    for imin in pbar:
        pbar.set_description(f"Preparing arguments {imin}")
        minute_t0 = task_t0 + timedelta(minutes=imin)
        paths_subset, _ = get_subset_paths(minute_t0, file_paths, timestamps)
        args_list.append((minute_t0, paths_subset, prepro_para))

    # Use process_map for parallel processing with progress bar
    buffer = []
    corr_full = np.zeros([n_lag, n_pair], dtype=np.float32)
    stack_count = 0
    
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        with tqdm(total=len(args_list), desc=f"Parallel xcorr [{task_t0.date()}]", position=0) as pbar:
            for corr in pool.imap_unordered(process_minute, args_list, chunksize=1):
                if np.count_nonzero(corr) == 0:
                    continue
                corr /= np.max(np.abs(corr))
                buffer.append(corr)
                
                if len(buffer) == chunk_size:
                    chunk = np.array(buffer)
                    stacked_chunk = np.zeros((n_lag, n_pair))

                    for ipair in range(n_pair):
                        data = chunk[:, :, ipair]
                        valid = np.any(data != 0, axis=1)
                        data = data[valid]

                        if len(data) == 0:
                            continue
                        if stack_method == 'robust':
                            stacked_chunk[:, ipair] = robust_stack(data)
                        elif stack_method == 'pws':
                            stacked_chunk[:, ipair] = phase_weighted_stack(data, nu=pws_exponent)
                        else:  # linear
                            stacked_chunk[:, ipair] = np.mean(data, axis=0)

                    corr_full += stacked_chunk
                    stack_count += 1

                    buffer = []
                
                del corr
                gc.collect()
                pbar.update(1)
    
    if len(buffer) > 0:
        chunk = np.array(buffer)
        stacked_chunk = np.zeros((n_lag, n_pair))
        for ipair in range(n_pair):
            data = chunk[:, :, ipair]
            valid = np.any(data != 0, axis=1)
            data = data[valid]
            if len(data) == 0:
                continue
            if stack_method == 'robust':
                stacked_chunk[:, ipair] = robust_stack(data)
            elif stack_method == 'pws':
                stacked_chunk[:, ipair] = phase_weighted_stack(data, nu=pws_exponent)
            else:  # linear
                stacked_chunk[:, ipair] = np.mean(data, axis=0)
        corr_full += stacked_chunk
        stack_count += 1
    
    corr_full /= max(stack_count, 1)
    print(f'corr_full max: {np.nanmax(corr_full)}; min: {np.nanmin(corr_full)}')

    if corr_path:
        saved_corr = np.loadtxt(corr_path, delimiter=',')
        if saved_corr.shape == corr_full.shape:
            corr_full += saved_corr
        else:
            warn(f'Shape mismatch between current correlation matrix and saved correlation matrix: {corr_full.shape} (current) against {saved_corr.shape} (saved)')
    return corr_full


def correlation(dir_path, prepro_para, corr_path=None, allowed_times=None):
    # Unpack required parameters from prepro_para
    n_lag = prepro_para['n_lag']
    n_pair = prepro_para['n_pair']
    cha1 = prepro_para['cha1']
    cha2 = prepro_para['cha2']
    effective_cha2 = prepro_para['effective_cha2']
    cha_list = prepro_para['cha_list']
    n_minute = prepro_para['n_minute']
    task_t0 = prepro_para['task_t0']
    src_ch = prepro_para['src_ch']
    spatial_ratio = prepro_para['spatial_ratio']
    
    ### FIXME: these funcs require a LOT of listdir calls, could be made more efficient in the future
    file_array, timestamps = get_reader_array(dir_path, allowed_times)
    if allowed_times: n_minute = floor((file_array[0].get_data(cha1, cha2).shape[0] / prepro_para.get('sps')) * len(file_array) / 60)
    
    corr_full = np.zeros([n_lag, n_pair], dtype = np.float32)
    stack_full = np.zeros([1, n_pair], dtype = np.int32)
    
    pbar = tqdm(range(n_minute))
    t_query = 0; t_compute = 0

    for imin in pbar:
        t0 = time.time()
        pbar.set_description(f"Processing {task_t0}")
        tdata = get_data_from_array(file_array, prepro_para, task_t0, timestamps, duration=timedelta(seconds=60))
        task_t0 += timedelta(minutes = 1)
        
        t_query += time.time() - t0
        t1 = time.time()
        # perform pre-processing
        trace_stdS, dataS = DAS_module.preprocess_raw_make_stat(tdata, prepro_para)
        # print(f'Processed data shape: {trace_stdS.shape}')

        # do normalization if needed
        white_spect = DAS_module.noise_processing(dataS, prepro_para)
        Nfft = white_spect.shape[1]; Nfft2 = Nfft // 2
        data = white_spect[:, :Nfft2]
        del dataS, white_spect

        ind = np.where((trace_stdS < prepro_para['max_over_std']) &
                                (trace_stdS > 0) &
                        (np.isnan(trace_stdS) == 0))[0]
        if not len(ind):
            ### debugging max over std errors
            print(f"max_over_std check: {np.where(trace_stdS < prepro_para['max_over_std'])[0]}")
            print(f"      over 0 check: {np.where(trace_stdS > 0)[0]}")
            print(f"       isnan check: {np.where(np.isnan(trace_stdS) == 0)[0]}")
            raise ValueError('the max_over_std criteria is too high which results in no data')
        sta = cha_list[ind]
        white_spect = data[ind]

        # loop over all stations
        if src_ch:
            sfft1 = DAS_module.smooth_source_spect(white_spect[int((src_ch - cha1)/spatial_ratio)], prepro_para)
            corr, tindx = DAS_module.correlate(sfft1, white_spect, prepro_para, Nfft)

            # stacking one minute
            corr_full[:, :] += corr.T
            stack_full[:, :] += 1
        else:
            for iiS in range(len(sta)):
                # smooth the source spectrum
                sfft1 = DAS_module.smooth_source_spect(white_spect[iiS], prepro_para)
                
                # correlate one source with all receivers
                corr, tindx = DAS_module.correlate(sfft1, white_spect[iiS:], prepro_para, Nfft)

                # update the receiver list
                tsta = sta[iiS:]
                receiver_lst = tsta[tindx]

                iS = int((effective_cha2*2 - cha1 - sta[iiS] + 1) * (sta[iiS] - cha1) / 2)

                # stacking one minute
                corr_full[:, iS + receiver_lst - sta[iiS]] += corr.T
                stack_full[:, iS + receiver_lst - sta[iiS]] += 1
            
        t_compute += time.time() - t1
    corr_full /= stack_full
    print(f'corr_full max: {np.nanmax(corr_full)}; min: {np.nanmin(corr_full)}')
    print(f"{round(t_query, 2)} seconds in data query, {round(t_compute, 2)} seconds in xcorr computing")    
    return corr_full


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


def plot_correlation(corr, prepro_para, cmap_param='bwr', save_corr=False, normalise=True, allowed_times=None, velocities=[600]):
    cha1, cha2, effective_cha2, spatial_ratio, cha_spacing, target_spatial_res, samp_freq, freqmin, freqmax, maxlag, n_minute, task_t0, src_ch, stack_method = prepro_para.get('cha1'), prepro_para.get('cha2'), prepro_para.get('effective_cha2'), prepro_para.get('spatial_ratio'), prepro_para.get('cha_spacing'), prepro_para.get('target_spatial_res'), prepro_para.get('samp_freq'), prepro_para.get('freqmin'), prepro_para.get('freqmax'), prepro_para.get('maxlag'), prepro_para.get('n_minute'), prepro_para.get('task_t0'), prepro_para.get('src_ch'), prepro_para.get('stack_method')

    out_dir = f'./results/figures/corrs/{task_t0}_{n_minute}mins_{cha1}:{cha2}/'
    out_name = f'{task_t0}_{n_minute}mins_{samp_freq}f{freqmin}:{freqmax}_{cha1}:{cha2}_{target_spatial_res}m'
    if allowed_times:
        for t1, t2 in allowed_times.items():
            out_name += f'_{t1}:{t2}'
    if src_ch:
        out_name += f'_{src_ch}src'
    if stack_method:
        out_name += f'_{stack_method}'
    if save_corr:
        np.savetxt(f'/data/localraid/saved_corrs/{out_name}.txt', corr[:, :(effective_cha2 - cha1)], delimiter=",")

    plt.figure(figsize = (12, 5), dpi = 150)
    if normalise:
        corr /= np.nanpercentile(np.abs(corr), 99.9)
        plt.imshow(corr[:, :(effective_cha2 - cha1)].T, aspect = 'auto', cmap = cmap_param,
                   vmax=1.0, vmin=-1.0, origin = 'lower', interpolation=None)
    else:
        plt.imshow(corr.T, aspect = 'auto', cmap = cmap_param, 
                   vmax = 2e-2, vmin = -2e-2, origin = 'lower', interpolation=None)
    
    if velocities:
        x = np.linspace(int(corr.shape[0]/2),corr.shape[0]-1,maxlag+1)
        for velocity in velocities:
            y = [(velocity*(xi/samp_freq-maxlag)) for xi in x]
            if src_ch:
                y = [yi+(src_ch - cha1) for yi in y]
            y = [yi/spatial_ratio for yi in y]
            plt.plot(x, y, linestyle='--', color='k', alpha=0.6)
        plt.xlim(0, corr.shape[0]-1)
        plt.ylim(0, corr.shape[1]-spatial_ratio)
    
    if src_ch:
        plt.scatter(int(corr.shape[0]/2), int((src_ch - cha1)/spatial_ratio), marker='*', s=50, color='k')
    _ =plt.yticks((np.linspace(cha1, cha2, 6) - cha1)/spatial_ratio, [int(i) for i in np.linspace(cha1, cha2, 6)], fontsize = 12)
    plt.ylabel("Channel number", fontsize = 12)
    _ = plt.xticks(np.arange(0, maxlag*samp_freq*2+1, samp_freq), np.arange(-maxlag, maxlag+1, 1), fontsize=12)
    plt.xlabel("Time lag (sec)", fontsize = 12)
    # bar = plt.colorbar(pad = 0.1, format = lambda x, pos: '{:.1f}'.format(x*100))
    # bar.set_label('Cross-correlation Coefficient', fontsize = 15)
    
    twiny = plt.gca().twinx()
    twiny.set_yticks((np.linspace(cha1, cha2, 6) - cha1)/spatial_ratio, [int(i* cha_spacing) for i in np.linspace(cha1, cha2, 6)], fontsize = 12)
    twiny.set_ylabel("Distance along cable (m)", fontsize = 12)
    
    plt.tight_layout()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(f'{out_dir}{out_name}.eps')


def save_correlation(corr, prepro_para):
    cha1, cha2, effective_cha2, target_spatial_res, samp_freq, freqmin, freqmax, n_minute, task_t0, src_ch, stack_method = prepro_para.get('cha1'), prepro_para.get('cha2'), prepro_para.get('effective_cha2'), prepro_para.get('target_spatial_res'), prepro_para.get('samp_freq'), prepro_para.get('freqmin'), prepro_para.get('freqmax'), prepro_para.get('n_minute'), prepro_para.get('task_t0'), prepro_para.get('src_ch'), prepro_para.get('stack_method')

    out_name = f'{task_t0}_{n_minute}mins_{samp_freq}f{freqmin}:{freqmax}_{cha1}:{cha2}_{target_spatial_res}m'
    if src_ch:
        out_name += f'_{src_ch}src'
    if stack_method:
        out_name += f'_{stack_method}'
    np.savetxt(f'/data/localraid/saved_corrs/rma/{out_name}.txt', corr, delimiter=",")


def plot_multiple_correlations(corrs:list, prepro_para:dict, vars, experiment_var:str, cmap_param:str='bwr', save_corr:bool=False):
    cha1, cha2, effective_cha2, spatial_ratio, cha_spacing, target_spatial_res, samp_freq, freqmin, freqmax, maxlag, n_minute, task_t0 = prepro_para.get('cha1'), prepro_para.get('cha2'), prepro_para.get('effective_cha2'), prepro_para.get('spatial_ratio'), prepro_para.get('cha_spacing'), prepro_para.get('target_spatial_res'), prepro_para.get('samp_freq'), prepro_para.get('freqmin'), prepro_para.get('freqmax'), prepro_para.get('maxlag'), prepro_para.get('n_minute'), prepro_para.get('task_t0')

    nrows = len(vars) // 2 + (len(vars) % 2 > 0)
    fig, axs = plt.subplots(2, nrows, figsize=(15, 10))
    for ax, corr, var in zip(axs.ravel(), corrs, vars):
        if experiment_var == 'channels':
            cha1, cha2 = var[0], var[1]       # for channel experiments (correct labelling)
            effective_cha2 = floor(cha1 + (cha2 - cha1) / spatial_ratio)

        plt.sca(ax)
        plt.imshow(corr[:, :(effective_cha2 - cha1)].T, aspect = 'auto', cmap = cmap_param, 
                vmax = 2e-2, vmin = -2e-2, origin = 'lower', interpolation=None)      # vmax, vmin original values of 2e-2, -2e-2 respectively

        _ =plt.yticks((np.linspace(cha1, cha2, 4) - cha1)/spatial_ratio, 
                    [int(i) for i in np.linspace(cha1, cha2, 4)], fontsize = 12)
        plt.ylabel("Channel number", fontsize = 12)
        _ = plt.xticks(np.arange(0, maxlag*samp_freq*2+1, 200), np.arange(-maxlag, maxlag+1, 2), fontsize=12)
        plt.xlabel("Time lag (sec)", fontsize = 12)
        # bar = plt.colorbar(pad = 0.1, format = lambda x, pos: '{:.1f}'.format(x*100))
        # bar.set_label('Cross-correlation Coefficient ($\\times10^{-2}$)', fontsize = 8)
        ax.label_outer()

        twiny = plt.gca().twinx()
        twiny.set_yticks(np.linspace(0, cha2 - cha1, 4), 
                                    [int(i* cha_spacing) for i in np.linspace(cha1, cha2, 4)])
        twiny.set_ylabel("Distance along cable (m)", fontsize = 12)
        twiny.label_outer()
        
        plt.tight_layout()
        plt.title(f"{experiment_var}: {var}")
    
    # follow convention of: {t_start}_{n_minute}mins_f{freqmin}:{freqmax}__{cha1}:{cha2}_{target_spatial_res}m
    # t_start = task_t0 - timedelta(minutes=n_minute)
    match experiment_var:
        case 'channels':
            plt.savefig(f'./results/figures/{task_t0}_{n_minute}mins_{samp_freq}f{freqmin}:{freqmax}__{target_spatial_res}m__{experiment_var}_experiment.eps')
            out_name = f'{task_t0}_{n_minute}mins_f{freqmin}:{freqmax}__{vars[0][0]}:{vars[0][1]}_{target_spatial_res}m'
        case 'frequencies':
            plt.savefig(f'./results/figures/{task_t0}_{n_minute}mins_{samp_freq}f__{cha1}:{cha2}_{target_spatial_res}m__{experiment_var}_experiment.eps')
            out_name = f'{task_t0}_{n_minute}mins_f{vars[0][0]}:{vars[0][1]}__{cha1}:{cha2}_{target_spatial_res}m'
        case 'stack_length':
            plt.savefig(f'./results/figures/{task_t0}_{samp_freq}f{freqmin}:{freqmax}__{cha1}:{cha2}_{target_spatial_res}m__{experiment_var}_experiment.eps')
            out_name = f'{task_t0}_{vars[0]}mins_f{freqmin}:{freqmax}__{cha1}:{cha2}_{target_spatial_res}m'
        case 'spatial_res':
            plt.savefig(f'./results/figures/{task_t0}_{n_minute}mins_{samp_freq}f{freqmin}:{freqmax}__{cha1}:{cha2}__{experiment_var}_experiment.eps')
            out_name = f'{task_t0}_{n_minute}mins_f{freqmin}:{freqmax}__{cha1}:{cha2}_{vars[0]}m'
        case _:
            plt.savefig(f'./results/figures/{task_t0}_{n_minute}mins_{samp_freq}f{freqmin}:{freqmax}__{cha1}:{cha2}_{target_spatial_res}m.eps')
            out_name = f'{task_t0}_{n_minute}mins_f{freqmin}:{freqmax}__{cha1}:{cha2}_{target_spatial_res}m'
    if save_corr:
        np.savetxt(f'./results/saved_corrs/{out_name}.txt', corrs[0][:, :(effective_cha2 - cha1)], delimiter=",")


def daily_correlations(dir_path, prepro_para):
    ### do daily xcorrs, save as one file
    # do reference at the same time? 
    n_lag               = prepro_para['n_lag']
    n_minute            = prepro_para['n_minute']
    target_spatial_res  = prepro_para['target_spatial_res']
    samp_freq           = prepro_para['samp_freq']
    freqmin             = prepro_para['freqmin']
    freqmax             = prepro_para['freqmax']
    src_ch              = prepro_para['src_ch']
    rcv_ch              = prepro_para['rcv_ch']
    stack_method        = prepro_para['stack_method']
    task_t0             = prepro_para['task_t0']
    task_t1             = task_t0 + timedelta(minutes=n_minute)
    
    ### reference trace
    ref_corr = parallel_xcorr(dir_path, prepro_para)
    out_name = f'{task_t0.date()}_{task_t1.date()}mins_{samp_freq}f{freqmin}:{freqmax}_{src_ch}:{rcv_ch}_{target_spatial_res}m_{stack_method}'
    np.savetxt(f'/data/localraid/dv_v_corrs/ref_stacks/{out_name}.txt', ref_corr, delimiter=",")
    
    n_days = n_minute // 1440
    prepro_para['n_minute'] = 1440
    
    corr_full = np.zeros([n_lag, n_days], dtype=np.float32)
    for day in range(0, n_days):
        # Prepare argument list for each minute
        day_corr = parallel_xcorr(dir_path, prepro_para)
        if type(day_corr) != int:
            corr_full[:, day] = day_corr[:, 0]
        else:
            corr_full[:, day].fill(np.nan)
        prepro_para['task_t0'] += timedelta(days=1)

    out_name = f'{task_t0.date()}_{task_t1.date()}mins_{samp_freq}f{freqmin}:{freqmax}_{src_ch}:{rcv_ch}_{target_spatial_res}m_{stack_method}'
    np.savetxt(f'/data/localraid/dv_v_corrs/{out_name}.txt', corr_full, delimiter=",")
    
    return corr_full
