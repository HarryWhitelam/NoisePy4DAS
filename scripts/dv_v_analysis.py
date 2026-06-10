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
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import Stream, Trace
from obspy.core.trace import Stats

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


def stretching_dvv(ccf_ref, ccf, dt, tmin, tmax, eps_range, side='causal'):
    '''Stretching Technique following Sens-Schonfelder & Wegler
    For this, ccf_ref is the day before our 'current' ccf
    Stretching is applied to reference trace, therefore, dv/v = -max(eps)
    
    Side can be 'causal', 'acausal' or 'both' for average
    '''
    mid = int(len(ccf_ref)/2)
    if side == 'causal':
        ccf_ref = ccf_ref[mid:]
        ccf = ccf[mid:]
    elif side == 'acausal':
        ccf_ref = ccf_ref[:mid+1][::-1]
        ccf = ccf[:mid+1][::-1]
    elif side == 'both':
        ccf_ref = (ccf_ref[mid:] + ccf_ref[:mid+1]) / 2
        ccf = (ccf[mid:] + ccf[:mid+1]) / 2
    
    n = len(ccf_ref)
    t = (np.arange(n) * dt)
    mask = (t >= tmin) & (t <= tmax)
    
    t_win = t[mask]
    cur_win = ccf[mask]

    interp_ref = interp1d(t, ccf_ref, kind='cubic', bounds_error=False, fill_value=0.0)

    corr_coeff = []
    for eps in eps_range:
        t_stretch = (1 + eps) * t_win
        ref_stretched = interp_ref(t_stretch)

        num = np.sum(cur_win * ref_stretched)
        den = np.sqrt(np.sum(cur_win**2) * np.sum(ref_stretched**2))

        if den == 0:
            corr_coeff.append(0)
        else:
            corr_coeff.append(num / den)

    corr_coeff = np.array(corr_coeff)

    idx = np.argmax(corr_coeff)
    eps_best_corr = np.max(corr_coeff)

    dvv = -eps_range[idx]

    return dvv, eps_best_corr, corr_coeff


def plot_dv_v(stream, task_t0, n_days, nlag_s=8, bandpass:list=None):
    ccf_stream = stream.copy()
    fig, ax = plt.subplots(1,1)
    if bandpass:
        ccf_stream.filter("bandpass", freqmin=bandpass[0], freqmax=bandpass[1])
    fig = ccf_stream.plot(type='section', recordstart=0, recordlength=nlag_s, fillcolors=('k', None), orientation='horizontal', fig=fig)
    ax.set_xticklabels([i-(nlag_s/2) for i in range(0,nlag_s+1)])
    ax.set_xlabel('Time lag (s)')
    ax.set_yticks([i*0.1 for i in range(0,n_days)])
    ax.set_yticklabels(np.arange(str(task_t0.date()), str((task_t0 + timedelta(days=n_days)).date()), dtype='datetime64[D]'))
    ax.yaxis.minorticks_off()
    plt.show()


dir_path = "/data/QNAP1_Data/Data/"
save_path='/data/localraid/dv_v_corrs/'
task_t0 = datetime(year = 2025, month = 2, day = 1, 
                   hour = 0, minute = 0, second = 0, microsecond = 0)

stack_method = 'pws'
# ccf = daily_pair_xcorr(dir_path, cha1=1000, cha2=1200, task_t0=task_t0, n_days=14, target_spatial_res=10, save_path=save_path, stack_method=stack_method, freqmin=0.1, freqmax=25.0)
# ccf = np.load('/data/localraid/dv_v_corrs/2025-02-05_2025-02-19_f0.1:25.0_1000:1200_10m_pws.npy')

n_days = 14
ch = 19
npts = 40

ccf_stream = Stream()
t = task_t0
for i in range(1,n_days+1):
    cc = np.loadtxt(f'/data/localraid/saved_corrs/{t}_1440mins_100f0.01:25.0_1000:1400_10m_1000src_pws.txt', delimiter=',', dtype=np.float64)
    stats = Stats()
    stats.delta = 1/100; stats.npts = cc.shape[0]
    ccf_stream.append(Trace(cc[:, ch], stats))
    t += timedelta(days=1)

for i in range(0, len(ccf_stream)):
    ccf_stream[i].stats.distance = i*100

# plot_dv_v(ccf_stream, task_t0, n_days)
# plot_dv_v(ccf_stream, task_t0, n_days, bandpass=[0.1, 1.0])
# plot_dv_v(ccf_stream, task_t0, n_days, bandpass=[1.0, 20.0])

# for f0, f1 in [[0.1, 0.5], [0.5, 1.0], [1.0, 5.0], [5.0, 10.0], [10.0, 20.0]]:
# for f0, f1 in [[4, 15]]:
#     print(f'\n--- dv/v between {f0}-{f1} Hz')
#     ccf_copy = ccf_stream.copy()
#     ccf_copy.filter("bandpass", freqmin=f0, freqmax=f1)
#     ccf_0, ccf_1 = ccf_copy[0].data, ccf_copy[1].data
#     dvv, eps_best_corr, corr_coeff = stretching_dvv(ccf_0, ccf_1, 0.01, 0.8, 2.0, np.linspace(-0.05, 0.05, 201), side='causal')
#     print(f'dv/v: {dvv}')
#     print(f'corr value: {eps_best_corr}')


### BANDPASSING STREAM
ccf_stream_cp = ccf_stream.copy()
f_bands = [[0.01, 0.5], [0.5, 1.0], [1.0, 5.0], [5.0, 10.0], [10.0, 15.0]]
for f_band in f_bands:
    ccf_stream = ccf_stream_cp.copy()
    ccf_stream.filter("bandpass", freqmin=f_band[0], freqmax=f_band[1])
    dvvs = [0]
    eps_corrs = [0]
    for i in range(1,n_days):
        ccf_0, ccf_1 = ccf_stream[i-1].data, ccf_stream[i].data
        dvv, eps_best_corr, corr_coeff = stretching_dvv(ccf_0, ccf_1, 0.01, 0.8, 1.3, np.linspace(-0.05, 0.05, 201), side='causal')
        dvvs.append(dvv * 100)
        eps_corrs.append(eps_best_corr)

    dates = np.arange(task_t0.date(), (task_t0 + timedelta(days=n_days)).date())
    fig, axs = plt.subplots(2,1, sharex=True)
    axs[0].plot(dates, np.cumsum(dvvs))
    axs[0].set_ylabel('dv/v (%)')
    axs[1].plot(dates, eps_corrs)
    axs[1].set_ylabel('Stretch correlation value')
    axs[0].set_title(f'dv/v between {f_band[0]} - {f_band[1]} Hz [causal]')
    axs[0].grid(); axs[1].grid()
    axs[0].set_ylim(-8.5, 5.0)
    axs[1].set_ylim(0, 1.0)
    plt.show()
