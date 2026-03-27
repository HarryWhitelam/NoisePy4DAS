##### Following Aquifer Monitoring..., Rodriguez Tribaldos, 2021
### Overview: 
# 1. Cross correlation (seperate)
    # This needs to be adapted, ccf between two channels
    # Stack ccfs for each day, then save as a 'row' in the output ccf
# 2. fk-filtering (OR bandpassing?)
# 3. Take coda window
# 4. Apply stretching technique
    # Stretch between -10% and 10%, applied with moving window of 0.25 s, shift of 0.02 s
    # Cross correlate previous and current (stretched) trace for that window
    # Remove outliers, and take median of values

import sys
sys.path.append("./src")
sys.path.append("./DASstore")
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import gc
import multiprocessing
from warnings import warn
import scipy
from scipy.signal import hilbert
import matplotlib.pyplot as plt

import DAS_module
from correlation_funcs import set_prepro_parameters
from tdms_io import get_reader_array, get_filepath_array, get_data_from_array, get_dir_properties, get_subset_paths


### Cross-correlation!
def correlate(fft1_smoothed_abs, fft2, D, Nfft):
    """
    Correlate two 1D spectra (source and receiver).
    Inputs:
      - fft1_smoothed_abs: 1D array, length Nfft2 (positive-frequency bins)
      - fft2: 1D array, length Nfft2 (positive-frequency bins)
      - D: dict with keys 'samp_freq', 'maxlag', optional 'cc_method'
      - Nfft: full FFT length used for ifft (Nfft >= 2*Nfft2)
    Returns:
      - s_corr: 1D real array (time-domain correlation limited to maxlag)
      - tindx: array([0]) if trace accepted, else empty array
    """
    sps = D['samp_freq']
    dt = 1.0 / sps
    maxlag = D['maxlag']
    method = D['cc_method']

    # ensure 1D numpy arrays
    fft1 = np.asarray(fft1_smoothed_abs).ravel()
    fft2 = np.asarray(fft2).ravel()

    if fft1.ndim != 1 or fft2.ndim != 1:
        raise ValueError("fft1 and fft2 must be 1D arrays")
    Nfft2 = fft1.size
    if fft2.size != Nfft2:
        raise ValueError(f"fft1 length ({Nfft2}) must equal fft2 length ({fft2.size})")

    if Nfft2 == 0:
        raise ValueError("No frequency bins provided (Nfft2 == 0)")

    # cross-correlation (conjugation of fft1 done in advance)
    cross_spec = fft2 * fft1  # 1D complex length Nfft2

    # build full spectrum (Nfft) with Hermitian symmetry
    crap = np.zeros(Nfft, dtype=np.complex64)
    crap[:Nfft2] = cross_spec.astype(np.complex64)
    if Nfft2 > 1:
        crap[-(Nfft2)+1:] = np.flip(np.conj(crap[1:Nfft2]), axis=0)
    crap[0] = 0+0j
    # subtract mean over the whole spectrum (optional step to reduce zero-lag spike)
    crap -= np.mean(crap)


    # inverse FFT -> time domain, shift so zero lag is centered
    td = scipy.fft.ifft(crap, Nfft, axis=0)
    td = np.fft.ifftshift(td)
    s_corr_full = np.real(td).astype(np.float32)

    # select lags within maxlag
    t = np.arange(-Nfft2, Nfft2) * dt
    ind = np.where(np.abs(t) <= maxlag)[0]
    if ind.size == 0:
        return np.array([], dtype=np.float32)
    s_corr = s_corr_full[ind]

    return s_corr


def process_minute(args):
    (minute_t0, file_paths, prepro_para) = args
    n_lag           = prepro_para.get('n_lag')
    src_ch          = prepro_para.get('src_ch')
    rcv_ch          = prepro_para.get('rcv_ch')
    stack_method    = prepro_para.get('stack_method')
    pws_eps         = prepro_para.get('pws_eps')
    
    # file_array = [TdmsReader(path) for path in file_paths]
    tdata = get_data_from_array(file_paths, prepro_para, minute_t0, duration=timedelta(seconds=60), filter=False, channels=[src_ch, rcv_ch])
    trace_stdS, dataS = DAS_module.preprocess_raw_make_stat(tdata, prepro_para)
    white_spect = DAS_module.noise_processing(dataS, prepro_para)
    Nfft = white_spect.shape[1]; Nfft2 = Nfft // 2
    white_spect = white_spect[:, :Nfft2]
    del file_paths, tdata, dataS
    gc.collect()
    
    ind = np.where((trace_stdS < prepro_para['max_over_std']) &
                   (trace_stdS > 0) &
                   (np.isnan(trace_stdS) == 0))[0]
    if len(ind) < 2:
        print(f'{minute_t0} had no valid data.')
        corr_zero   = np.zeros((n_lag), dtype=np.float32)
        stack_zero  = np.zeros((1), dtype=np.int32)

        if stack_method == 'pws':
            phasor_zero = np.zeros((n_lag), dtype=np.complex64)
            phcount_zero = np.zeros((1), dtype=np.int32)
            return corr_zero, stack_zero, phasor_zero, phcount_zero
        else:
            return corr_zero, stack_zero

    corr_full = np.zeros([n_lag], dtype=np.float32)
    stack_full = np.zeros([1], dtype=np.int32)
    if stack_method == 'pws':
        phasor_full = np.zeros((n_lag), dtype=np.complex64)
        phasor_count_full = np.zeros((1), dtype=np.int32)

    sfft1 = DAS_module.smooth_source_spect(white_spect[0], prepro_para)
    # rcv_idx = int((rcv_ch - cha1) / spatial_ratio) if rcv_ch is not None else int(effective_cha2)
    corr = correlate(sfft1, white_spect[1], prepro_para, Nfft)
    corr = corr.T
    corr_full += corr
    stack_full += 1
    if stack_method == 'pws':
        z = hilbert(corr, axis=0)
        amp = np.abs(z)
        ph = np.where(amp > pws_eps, z/amp, 0.0j)
        phasor_full += ph
        phasor_count_full += 1

    if stack_method == 'pws':
        return corr, stack_full, phasor_full, phasor_count_full
    else:
        return corr_full, stack_full


def parallel_xcorr(dir_path, prepro_para):
    n_lag        = prepro_para['n_lag']
    n_minute     = prepro_para['n_minute']
    task_t0      = prepro_para['task_t0']
    stack_method = prepro_para['stack_method']
    pws_exponent = prepro_para['pws_exponent']

    file_paths, timestamps = get_filepath_array(dir_path, task_t0, task_t0+timedelta(minutes=n_minute))
    task_t0 = timestamps[0].replace(microsecond=0)

    # Prepare argument list for each minute
    args_list = []
    pbar = tqdm(range(n_minute))
    for imin in pbar:
        pbar.set_description(f"Processing {imin}")
        minute_t0 = task_t0 + timedelta(minutes=imin)
        paths_subset, _ = get_subset_paths(minute_t0, file_paths, timestamps)
        args_list.append((minute_t0, paths_subset, prepro_para))

    # Use process_map for parallel processing with progress bar
    corr_full = np.zeros([n_lag], dtype=np.float32)
    stack_full = np.zeros([1], dtype=np.int32)
    if stack_method == 'pws':
        phasor_full = np.zeros([n_lag], dtype=np.complex64)
        phasor_count_full = np.zeros([1], dtype=np.int32)
    
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        with tqdm(total=len(args_list), desc="Parallel xcorr", position=0) as pbar:
            for result in pool.imap_unordered(process_minute, args_list, chunksize=1):
                if stack_method == 'pws': 
                    corr, stack, phasor, phcount = result
                    corr_full += corr
                    stack_full += stack
                    phasor_full += phasor
                    phasor_count_full += phcount
                    del phasor, phcount
                else:
                    corr, stack = result
                    corr_full  += corr
                    stack_full += stack
                del corr, stack
                gc.collect()
                pbar.update(1)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        corr_linear = corr_full / np.maximum(stack_full, 1)    
    if stack_method == 'pws':
        coherency = np.abs(phasor_full / np.maximum(phasor_count_full, 1))
        corr_full = corr_linear * coherency**pws_exponent
    else:
        corr_full = corr_linear
    
    # center = int(prepro_para['maxlag'] * prepro_para['samp_freq'])
    # mute = 3
    # corr_full[:, center-mute:center+mute+1] = 0.0
    print(f'corr_full max: {np.nanmax(corr_full)}; min: {np.nanmin(corr_full)}')

    return corr_full


def daily_pair_xcorr(dir_path: str, cha1: int, cha2: int, task_t0: datetime, n_days: int, target_spatial_res: int, save_path: str = None, stack_method: str = 'pws', freqmin: float = 0.1, freqmax: float = 25.0):
    rows = []
    for d in range(n_days):
        day_t0 = (task_t0 + timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0)
        # create a per-day prepro dict (keep other params from prepro_base)
        pre = set_prepro_parameters(dir_path,
                                    day_t0,
                                    target_spatial_res=target_spatial_res,
                                    cha1=cha1, cha2=cha2,
                                    n_minute=1440,
                                    freqmin=freqmin, freqmax=freqmax,
                                    stack_method=stack_method,
                                    src_ch=cha1, rcv_ch=cha2)
        # compute correlation for the day (should return shape [n_lag, 1])
        corr_full = parallel_xcorr(dir_path, pre)
        # ensure shape (n_lag,)
        if corr_full is None:
            raise RuntimeError(f"parallel_xcorr returned None for day {day_t0}")
        if corr_full.ndim == 2 and corr_full.shape[1] == 1:
            ccf = corr_full[:, 0].astype(np.float32)
        elif corr_full.ndim == 1:
            ccf = corr_full.astype(np.float32)
        else:
            # if multiple pairs returned, attempt to find the pair corresponding to (a,b)
            # fallback: take the first column but warn
            import warnings
            warnings.warn(f"Expected single pair for channels {cha1}:{cha2}, got shape {corr_full.shape}; taking first column")
            ccf = corr_full[:, 0].astype(np.float32)

        rows.append(ccf)

    if not rows:
        return np.empty((0, 0), dtype=np.float32)

    ccf_matrix = np.vstack(rows)   # shape (n_days, n_lag)

    if save_path:
        corr_name = f'{task_t0.date()}_{(task_t0+timedelta(days=n_days)).date()}_f{freqmin}:{freqmax}_{cha1}:{cha2}_{target_spatial_res}m_{stack_method}'
        np.save(save_path + corr_name, ccf_matrix)
    return ccf_matrix


def stretch_trace(trace, dt, eps, kind='cubic', fill_value=0.0):
    """
    Return trace stretched by factor (1+eps): test_stretched(t) = trace(t / (1+eps))
    - trace: 1D array
    - dt: sample interval (s)
    - eps: fractional stretch (e.g. 0.01 => +1% stretch)
    """
    n = len(trace)
    t = np.arange(n) * dt
    f = scipy.interpolate.interp1d(t, trace, kind=kind, bounds_error=False, fill_value=fill_value)
    # sample original trace at scaled times t/(1+eps)
    t_sample = t / (1.0 + eps)
    return f(t_sample)


dir_path = "/data/QNAP1_Data/Data/"
save_path='/data/localraid/dv_v_corrs/'
task_t0 = datetime(year = 2025, month = 2, day = 5, 
                   hour = 0, minute = 0, second = 0, microsecond = 0)

stack_method = 'linear'

ccf = daily_pair_xcorr(dir_path, cha1=1000, cha2=1200, task_t0=task_t0, n_days=14, target_spatial_res=10, save_path=save_path, stack_method=stack_method, freqmin=0.1, freqmax=25.0)

# ccf = np.load('/data/localraid/dv_v_corrs/2025-02-16_2025-02-19_f0.1:25.0_1000:1200_10m_pws.npy')


from obspy import Stream, Trace
from obspy.core.trace import Stats
ccf_stream = Stream()
stats = Stats()
stats.delta = 1/100; stats.npts = ccf.shape[1]
for i in range(0, ccf.shape[0]):
    ccf_stream.append(Trace(ccf[i, :], stats))

# from obspy import read, UTCDateTime, Stream
# ccf_stream.filter("bandpass", freqmin=5, freqmax=50)
# print(ccf_stream[0])
for i in range(0, len(ccf_stream)):
    ccf_stream[i].stats.distance = i*100
ccf_stream.plot(type='section', recordstart=0, recordlength=8, fillcolors=('k', None), orientation='horizontal')
