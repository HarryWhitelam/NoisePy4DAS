import os
import gc
import numpy as np
import psutil
import pandas as pd
import geopandas as gpd
from scipy.signal import welch, ShortTimeFFT, convolve2d, savgol_filter
from scipy.signal.windows import hamming
from scipy.fft import rfft, rfftfreq
from obspy.signal.spectral_estimation import get_nlnm, get_nhnm
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from skimage.util import compare_images
import contextily as cx
from math import ceil, sin, cos, atan2, degrees, radians
import multiprocessing
from tqdm import tqdm
import bisect

from tdms_io import get_reader_array, get_filepath_array, get_data_from_array, get_dir_properties, load_xcorr
from weather import plot_waverider_csv, plot_met_csv, plot_era5_csv


def dms_to_dd(degrees, minutes=0, seconds=0):
    return degrees + (minutes/60) + (seconds/3600)


def plot_gps_coords(file_path):
    gps_df = pd.read_csv(file_path, sep=',', index_col=0)
    
    gps_df[['lat_degs', 'lat_mins']] = gps_df['lat'].str.split(' ', expand=True).astype(float)
    gps_df[['lon_degs', 'lon_mins']] = gps_df['lon'].str.split(' ', expand=True).astype(float)
    gps_df['lat'], gps_df['lon'] = dms_to_dd(gps_df['lat_degs'], gps_df['lat_mins']), dms_to_dd(gps_df['lon_degs'], gps_df['lon_mins'])
    
    gdf = gpd.GeoDataFrame(
        gps_df[['lat', 'lon']], geometry=gpd.points_from_xy(gps_df['lon'], gps_df['lat'], crs='EPSG:4326')
    )
    
    # gps_df[['lat', 'lon']].to_csv('track_gps.csv', sep=',')
    
    ax = gdf.plot(figsize=(10, 10), color='red')
    cx.add_basemap(ax, crs=gdf.crs)
    plt.show()


def image_comparison(data_dict, comp_ids, method='all', ncols=2, cmap='gray'):
    data_list = list(data_dict.values())
    if method in ('diff', 'all'):
        data_dict['diff'] = compare_images(data_dict.get(comp_ids[0]), data_dict.get(comp_ids[1]), method='diff')
    if method in ('blend', 'all'):
        data_dict['blend'] = compare_images(data_dict.get(comp_ids[0]), data_dict.get(comp_ids[1]), method='blend')
    
    nrows = len(data_dict) // ncols + (len(data_dict) % ncols > 0)
    
    fig = plt.figure(figsize=(15, 12))
    # plt.suptitle("TITLE HERE")
    for n, (key, val) in enumerate(data_dict.items()): 
        ax = plt.subplot(nrows, ncols, n + 1)
        ax.imshow(val, cmap=cmap, aspect='auto', interpolation='none')
        ax.title.set_text(key)
    
    fig.tight_layout()
    plt.show()


# def octave_smooth(psd, freqs, width_octaves=0.5, axis=0, eps=1e-12):
#     """
#     Smooth PSD similar to ObsPy's period smoothing around each octave.
#     - psd: array of shape (N_freq, ...) or any shape where `axis` indexes frequency.
#            If psd is in dB (10*log10), this function converts to linear, smooths, then
#            converts back to dB and returns the smoothed result in the same units/shape.
#     - freqs: 1D array of frequencies (Hz), length N_freq
#     - width_octaves: smoothing window width in octaves (factor-of-two units). Default 0.5.
#     - axis: axis in `psd` that corresponds to frequency (default 0).
#     - eps: small floor added to linear PSD to avoid log of zero.
#     Returns: smoothed PSD array with the same shape as `psd`.
#     """
#     freqs = np.asarray(freqs)
#     if freqs[0] == 0.0:
#         freqs[0] = freqs[1] * 1e-6  # avoid 0 Hz clipping
#     periods = 1.0 / freqs
#     logp = np.log10(periods)

#     width_decades = width_octaves * np.log10(2.0)
#     half_width = width_decades / 2.0

#     psd_arr = np.asarray(psd)
#     moved = np.moveaxis(psd_arr, axis, 0)
#     n_freq = moved.shape[0]
#     rest_shape = moved.shape[1:]
#     n_cols = int(np.prod(rest_shape)) if rest_shape != () else 1
#     flat = moved.reshape(n_freq, n_cols)

#     linear = 10.0 ** (flat / 10.0)
#     order = np.argsort(logp)
#     logp_sorted = logp[order]

#     smoothed_linear = np.empty_like(linear)
#     for i in range(n_freq):
#         center = logp[i]
#         left = center - half_width
#         right = center + half_width
#         i0 = bisect.bisect_left(logp_sorted, left)
#         i1 = bisect.bisect_right(logp_sorted, right)
#         sel = order[i0:i1]
#         if sel.size == 0:
#             smoothed_linear[i, :] = linear[i, :]
#         else:
#             smoothed_linear[i, :] = np.mean(linear[sel, :], axis=0)

#     smoothed_db_flat = 10.0 * np.log10(smoothed_linear + eps)
#     smoothed = smoothed_db_flat.reshape((n_freq,) + rest_shape)
#     smoothed = np.moveaxis(smoothed, 0, axis)
#     return smoothed


def octave_smooth(psd, freqs, width_octaves=0.5, axis=0, eps=1e-12):
    """
    ObsPy-like period smoothing: weighted average in linear power with weights
    proportional to period-bin widths (so smoothing conserves integrated power
    and avoids the large positive bias from naive arithmetic linear averaging).
    - psd: PSD in dB (shape with frequency on `axis`)
    - freqs: 1D frequency vector (Hz)
    - width_octaves: smoothing window width in octaves (default 0.5)
    - axis: frequency axis of psd
    - eps: small floor for numerical stability
    Returns: smoothed PSD in dB (same shape as input)
    """
    freqs = np.asarray(freqs, dtype=float)
    # guard DC / zero-frequency
    if freqs.size > 1 and freqs[0] == 0.0:
        freqs[0] = freqs[1] * 1e-6

    # periods and their log10 (period domain smoothing)
    periods = 1.0 / freqs
    logp = np.log10(periods)

    width_decades = width_octaves * np.log10(2.0)
    half_width = width_decades / 2.0

    # move frequency axis to axis 0
    psd_arr = np.asarray(psd)
    moved = np.moveaxis(psd_arr, axis, 0)
    n_freq = moved.shape[0]
    rest_shape = moved.shape[1:]
    n_cols = int(np.prod(rest_shape)) if rest_shape != () else 1
    flat = moved.reshape(n_freq, n_cols)  # shape (n_freq, n_cols)

    # linear power
    linear = 10.0 ** (flat / 10.0)

    # prepare sorted period axis for fast selection
    order = np.argsort(logp)
    logp_sorted = logp[order]
    periods_sorted = periods[order]

    # compute approximate period-bin widths (linear domain) for weighting
    # central difference for interior, forward/backward for ends
    widths = np.empty_like(periods_sorted)
    if periods_sorted.size > 1:
        widths[0] = periods_sorted[1] - periods_sorted[0]
        widths[-1] = periods_sorted[-1] - periods_sorted[-2]
        if periods_sorted.size > 2:
            widths[1:-1] = 0.5 * (periods_sorted[2:] - periods_sorted[:-2])
    else:
        widths[:] = 1.0

    # weighted average in linear domain using period widths as weights
    smoothed_linear = np.empty_like(linear)
    import bisect
    for i in range(n_freq):
        center = logp[i]
        left = center - half_width
        right = center + half_width
        i0 = bisect.bisect_left(logp_sorted, left)
        i1 = bisect.bisect_right(logp_sorted, right)
        sel = order[i0:i1]
        if sel.size == 0:
            smoothed_linear[i, :] = linear[i, :]
        else:
            sel_idx = slice(i0, i1)
            w = widths[sel_idx][:, None]             # shape (n_sel, 1)
            vals = linear[sel, :]                    # shape (n_sel, n_cols)
            weighted = (vals * w).sum(axis=0)
            wsum = w.sum()
            if wsum == 0:
                smoothed_linear[i, :] = linear[i, :]
            else:
                smoothed_linear[i, :] = weighted / wsum

    # convert back to dB and restore original shape/axis
    smoothed_db_flat = 10.0 * np.log10(smoothed_linear + eps)
    smoothed = smoothed_db_flat.reshape((n_freq,) + rest_shape)
    smoothed = np.moveaxis(smoothed, 0, axis)
    return smoothed


def parallel_spectral_analysis():
    dir_list = ['/data/QNAP1_Data/Data/']
    # for y, ms in [[2024, [4,5,6,7,8,9,10,11,12]], [2025, [1,2,3,4,5,6,7]]]:
    for m in [2]:
        t_start = datetime(year=2025, month=m, day=1)
        # t_end = t_start + relativedelta(months=1)
        t_end = t_start + relativedelta(days=2)
        n_minutes = (t_end - t_start).total_seconds() // 60
        
        channels = [750, 788, 875, 1475]
        for dir_path in dir_list:
            args_list = []
            prepro_para = {
                'target_sps': 100,
                'target_spatial_res': 1,
                'n_minute': n_minutes,
                'freqmin': 0.01,
                'freqmax': 49.9,
            }
            
            filepath_array, timestamps = get_filepath_array(dir_path, t_start, t_end)
            t_start = timestamps[0].replace(microsecond=0)
            print(f'Data running from {t_start} to {timestamps[-1].replace(microsecond=0)}')
            data = get_data_from_array(filepath_array, prepro_para, t_start, timestamps, duration=timedelta(minutes=n_minutes), channels=channels)
        
            for i, channel in enumerate(channels):
                run_prepro_para = prepro_para.copy()
                run_prepro_para.update({'cha1':channel,
                                        'cha2':channel+1})
                channel_data = data[:,i]
                #                 dir_path, prepro_para, t_start, save_spec, data, window_length, plot_tides
                #                 dir_path, prepro_para, t_start, save_ppsd, data, window_length
                args_list.append((dir_path, run_prepro_para, t_start, True, channel_data))
        
            # p = multiprocessing.Pool(multiprocessing.cpu_count())
            # with tqdm(total=len(args_list), desc=f"{dir_path} spectrograms", position=0) as pbar:
            #     for _ in p.starmap(ts_spectrogram, args_list, chunksize=1):
            #         pbar.update(1)
            # p.close()
            
            p = multiprocessing.Pool(multiprocessing.cpu_count())
            with tqdm(total=len(args_list), desc=f"{dir_path} PPSDs", position=0) as pbar:
                for _ in p.starmap(ppsd, args_list, chunksize=1):
                    pbar.update(1)
            p.close()

def ts_spectrogram(dir_path:str, prepro_para:dict, t_start:datetime, save_spec=False, data=None, window_length=600, plot_tides=None):
    cha1, sps, freqmin, freqmax, n_minute = prepro_para.get('cha1'), prepro_para.get('target_sps'), prepro_para.get('freqmin'), prepro_para.get('freqmax'), prepro_para.get('n_minute')
    
    out_dir = f"./results/figures/PSD_Experiments/{window_length}s_window/"
    
    if type(data)==type(None):
        reader_array, timestamps = get_reader_array(dir_path)
        if t_start == None: t_start = timestamps[0].replace(microsecond=0)
        data = get_data_from_array(reader_array, prepro_para, t_start, timestamps, duration=timedelta(minutes=n_minute))[:, 0]
    
    N = data.shape[0]
    win = hamming(int(sps*window_length), sym=True)               # 28/10/2025 longer window, longer time averaging! Output name changed!
    # g_std = 12
    # gaussian_win = gaussian(sps*60, g_std, sym=True)
    stft = ShortTimeFFT(win, hop=int(sps*(window_length*0.5)), fs=sps, scale_to='psd')
    spec = stft.spectrogram(data)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    t_min, t_max = stft.extent(N)[:2]
    ax.set_title(rf"{t_start} at channel {cha1}")
    spec = 10 * np.log10(spec + 1e-12)
    try:
        f_bins = np.linspace(freqmin, freqmax, stft.f_pts)
        spec = octave_smooth(spec, f_bins)
    except:
        print(f'octave smoothing failed!!!')
    
    ext = stft.extent(N)
    im1 = ax.imshow(spec, origin='lower', aspect='auto', 
                     extent=ext, cmap='jet', vmin=np.nanpercentile(spec,1), vmax=np.nanpercentile(spec,99))
    ax.set_yscale('log')
    plt.grid(which='both')
    
    if plot_tides:
        ax1 = ax.twinx()
        tidal_df = pd.read_csv('./results/checkpoints/CRO_final.csv', parse_dates=True, index_col=0)
        tidal_df = tidal_df[t_start:t_start+timedelta(minutes=n_minute)]
        ax1.plot([(ts - t_start).total_seconds() for ts in tidal_df.index], tidal_df['ASLVBG02'])
        ax1.set_ylabel('Tidal Height (m)')
        plt.sca(ax=ax)
    
    if n_minute > 1440:
        n_days = int(n_minute / 1440) + 1
        midnight_start = t_start.replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            tick_dates = pd.date_range(midnight_start, periods=n_days, freq=timedelta(days=(n_days//6)))
        except:
            tick_dates = pd.date_range(midnight_start, periods=n_days, freq=timedelta(days=1))
        tick_positions = []
        for tick_date in tick_dates:
            minutes_from_start = (tick_date - t_start).total_seconds() / 60
            tick_positions.append((minutes_from_start / n_minute) * ext[1])
        
        valid_ticks = [(pos, date) for pos, date in zip(tick_positions, tick_dates) 
                    if 0 <= pos <= ext[1]]
        if valid_ticks:
            positions, dates = zip(*valid_ticks)
            _ = plt.xticks(positions, [d.strftime('%Y-%m-%d') for d in dates], rotation=30)
        
        minor_tick_interval = 1440 / n_minute * ext[1]
        minor_positions = np.arange(0, ext[1], minor_tick_interval)
        _ = plt.xticks(minor_positions, minor=True)
    else: 
        _ = plt.xticks(np.linspace(0, ext[1], 4), pd.date_range(t_start, t_start+timedelta(minutes=n_minute), periods=4), rotation=30)
        _ = plt.xticks(np.linspace(0, ext[1], 16), minor=True)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(freqmin, freqmax)
    
    fig.colorbar(im1, label='Nano strainrate PSD (dB)')
    # plt.tight_layout()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(f'{out_dir}/{t_start}__{t_start+timedelta(minutes=n_minute)}_f{freqmin}:{freqmax}_{cha1}_spectrogram{"_tides" if plot_tides else ""}.png', bbox_inches='tight')
    if save_spec:
        np.savetxt(f'/data/localraid/saved_specs/{window_length}s_window/{t_start}__{t_start+timedelta(minutes=n_minute)}_f{freqmin}:{freqmax}_{cha1}_spec.txt', spec, delimiter=",")


def ppsd(dir_path:str, prepro_para:dict, t_start:datetime, save_ppsd=False, data=None, window_length=3600):
    cha1, sps, f1, f2, n_minute = prepro_para.get('cha1'), prepro_para.get('target_sps'), prepro_para.get('freqmin'), prepro_para.get('freqmax'), prepro_para.get('n_minute')
    out_dir = f"./results/figures/PSD_Experiments/{window_length}s_window/ppsds/"
    
    if type(data)==type(None):
        reader_array, timestamps = get_reader_array(dir_path)
        if t_start == None: t_start = timestamps[0].replace(microsecond=0)
        data = get_data_from_array(reader_array, prepro_para, t_start, timestamps, duration=timedelta(minutes=n_minute))[:, 0]

    nfft = 2 ** 17
    nr = 201        # number of amplitude bins, changed from 501
    hn = nfft // 2
    
    # # First pass: collect sample of PSD values to determine range     COMMENTED WHEN SPECIFYING LIMITS BELOW
    sample_fd_values = []
    seg_length = int(window_length * sps)
    n_segs = len(data) // seg_length
    sample_segs =  n_segs // 10 # Sample first 10% of segments
    
    for i in range(sample_segs):
        d = data[i*seg_length:(i+1)*seg_length]
        if len(d) == seg_length:
            fft_d = np.fft.fft(d, nfft)
            psd_lin = (np.abs(fft_d) ** 2) / (nfft * sps)
            fd = 10 * np.log10(psd_lin + 1e-12)
            sample_fd_values.extend(fd[:hn].flatten())
    fd_min = np.nanpercentile(sample_fd_values, 0.01)
    fd_max = np.nanpercentile(sample_fd_values, 99.99)

    # fd_min, fd_max = [-30, 50]
    # print(f'fd_min: {fd_min}; fd_max: {fd_max}')
    
    # Create adaptive binning
    db_range = fd_max - fd_min
    scale = (nr - 1) / db_range
    offset = -fd_min
    
    # Second pass: bin with adaptive scaling
    psd = np.zeros((nr, hn))
    p = np.zeros(hn)
    
    f = np.arange(nfft) * sps / (nfft)
    fn1 = int(f1*nfft/sps); fn2 = int(f2*nfft/sps)
    
    for i in range(n_segs):
        d = data[i*seg_length:(i+1)*seg_length]
        ### current fix for NaNs is to skip any section containing NaNs, not exactly robust >:|
        if len(d) == seg_length and ~np.isnan(d).any():
            fft_d = np.fft.fft(d, nfft)
            psd_lin = (np.abs(fft_d) ** 2) / (nfft * sps)
            fd = 10 * np.log10(psd_lin + 1e-12)
            fd_sm = octave_smooth(fd[:hn], f[:hn])     # octave smoothing            
            p += fd_sm
            for j in range(hn):
                index = int((fd_sm[j] + offset) * scale)
                if index < 0:
                    index = 0
                elif index >= nr:
                    index = nr - 1
                psd[index, j] += 1
    
    # Create proper dB axis
    db = np.linspace(fd_min, fd_max, nr)
    # print(f'down: {down_clip}; up: {up_clip}')
    
    nrf = 10
    f = f[fn1: fn2+1]; f = f[::nrf]
    P = psd[:, fn1: fn2+1]
    pp = savgol_filter(p, 11, 2)
    pp = pp[fn1: fn2+1] / n_segs; pp = pp[::nrf]
    P = P[::1, ::nrf]
    sl = 2
    P = convolve2d(P, np.ones((sl, sl))/sl**2, 'same')
    for i in range(len(P[0])):
        P[:, i] /= np.sum(P[:, i])
    print(f'PPSD peak percentage after smoothing+renorm: {P.max()*100.0:.2f}%')

    plt.figure()
    plt.pcolormesh(f, db, P*100, cmap='viridis')
    plt.xscale('log')
    plt.grid(which='both')
    plt.colorbar(shrink=0.75, aspect=30, pad=0.05, extend='both', label=r'Probability (%)')
    # cbar.ax.tick_params(labelsize=16)
    plt.semilogx(f, pp, lw=1.2, color='#888888')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Nano strainrate PSD (dB)')
    plt.xlim([0.01, 50])
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    t_end = t_start+timedelta(minutes=n_minute) - timedelta(hours=1)
    plt.title(rf"{t_start.date()} to {t_end.date()} at channel {cha1}")
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{t_start}__{t_start+timedelta(minutes=n_minute)}_f{f1}:{f2}_{cha1}_PPSD.png', bbox_inches='tight')


def get_spec_files(dir:str, t0:datetime, t1:datetime, target_cha:int):
    files = []
    for dir_path, dir_names, file_names in os.walk(dir):
        for file in file_names:
            try:
                f_split = file.split('_')
                t_start = datetime.strptime(f_split[0], '%Y-%m-%d %H:%M:%S')
                t_end = datetime.strptime(f_split[2], '%Y-%m-%d %H:%M:%S')
                if target_cha and int(f_split[-2]) != target_cha: continue
                if t0 and t_start.date() < t0.date(): continue
                if t1 and t_end.date() > t1.date(): continue
            except Exception: 
                continue        # basically there was no datetime within the file
            files.append((os.path.join(dir_path, file), t_start))
    files.sort(key=lambda x: x[1])
    return files


def spectral_power_ts(f_ranges:list, window_length:timedelta, target_cha=None, t0=None, t1=None):
    files = get_spec_files('/data/localraid/saved_specs/dense_f/', t0, t1, target_cha)
    
    mean_df = pd.DataFrame(columns=['timestamp', *[f'{f_range[0]}_{f_range[1]}' for f_range in f_ranges]]).set_index('timestamp')
    t_size = np.loadtxt(files[0][0], delimiter=',').shape[1]
    t_bin_length = (files[1][1] - files[0][1]).total_seconds() / t_size
    cosine_window = np.cos(np.linspace(-np.pi/2, np.pi/2, int(window_length.total_seconds() / t_bin_length)))
    cosine_window = cosine_window / cosine_window.sum()
    for file_path, t_start in files:
        spec = np.loadtxt(file_path, delimiter=',')
        f_size, t_size = spec.shape
        f_bins = np.linspace(0.0, 50.016666666666666, f_size)
        
        for f_min, f_max in f_ranges:
            f_idxs = [i for i, f in enumerate(f_bins) if f_min <= f <= f_max]
            freq_mean = spec[f_idxs, :].mean(axis=0)
            filtered = np.convolve(freq_mean, cosine_window, mode='same')
            for j in range(t_size):
                t = t_start + timedelta(seconds=t_bin_length * j)
                mean_df.loc[t, f'{f_min}_{f_max}'] = filtered[j]
    
    # mean_df = pd.DataFrame(mean_fs)
    mean_df.sort_index(inplace=True)
    mean_df.plot(figsize=(10,3))
    plt.tight_layout()
    plt.savefig(f'./results/figures/ts_spec_lines/low_f_{t0}_{t1}_{window_length.total_seconds()/60}min_avg.png', bbox_inches='tight')
    # plt.show()


def plot_spectrogram(target_cha, t0, t1, norm=False, tides=False, markers=None, weather=[], c_range=None, octave_smoothing=True):
    delta = (t1 - t0).total_seconds()
    files = get_spec_files('/data/localraid/saved_specs/', t0, t1, target_cha)
    
    if norm:
        means_df = pd.read_csv(f'./results/checkpoints/dense_monthly_means_{target_cha}.csv', index_col=0, header=0)
        if delta <= 2419200: means_df = means_df[pd.date_range(t0, t1, freq='MS').strftime("%Y-%m-%d").to_list()]
        else:                means_df = means_df[pd.date_range(t0, t1, freq='MS')[:-1].strftime("%Y-%m-%d").to_list()]
        means = np.asarray(means_df.mean(axis=1))
    ncols = int(np.ceil(len(files)))
    nrows = 1
    if len(weather) > 0:
        nrows += len(weather)
        plot_daily = False if delta <= 2419200 else True
        df_wave = plot_waverider_csv('./results/checkpoints/hpg_wave.csv', plot_daily, get_df=True)
        df_met = plot_met_csv('./results/checkpoints/hpg_met.csv', plot_daily, get_df=True)
        df_era5 = plot_era5_csv('./results/checkpoints/combined_weather.csv', plot_daily, get_df=True)
        weather_df = pd.concat([df_wave, df_met, df_era5], axis=1)[t0.date():t1.date()]
    height_ratios = [2] + [1] * max(0, len(weather))
    if delta <= 2419200:    fig, axs = plt.subplots(nrows, 1, figsize=(12, 4*nrows), sharex=True, gridspec_kw={'height_ratios': height_ratios})
    elif len(weather) == 0: fig, axs = plt.subplots(nrows, 1, figsize=(ncols, 8), sharex=True)
    else:                   fig, axs = plt.subplots(nrows, 1, figsize=(ncols, 4*nrows), sharex=True, gridspec_kw={'height_ratios': height_ratios})
    
    time_bin_counts = []; spec_min = 0; spec_max = 0
    for f in files:     # there has to be a more efficient way to do this
        spec = np.loadtxt(f[0], delimiter=',')
        time_bin_counts.append(spec.shape[1])
        if norm: spec -= means[:,np.newaxis]
        spec_min = min(np.nanmin(np.nanpercentile(spec, 1)), spec_min)
        spec_max = max(np.nanmax(np.nanpercentile(spec,99)), spec_max)
        del spec
        gc.collect()
    if c_range:
        if spec_min < c_range[0] or spec_max > c_range[1]:
            print(f'spec exceeds given c_range! c_range: {c_range} || spec_min: {spec_min}; spec_max: {spec_max}')
        spec_min, spec_max = c_range
    cumulative_offsets = np.cumsum([0] + time_bin_counts[:-1])
    with tqdm(total=len(files), desc=f'Loading and plotting spectrogram files') as pbar:
        for i, (file_path, t_start) in enumerate(files):
            ax = axs if len(weather) == 0 else axs[0]
            spec = np.loadtxt(file_path, delimiter=',')
            if norm: spec -= means[:,np.newaxis]
            
            f_size, t_size = spec.shape
            f_bins = np.linspace(0.01, 50.016666666666666, f_size)
            if octave_smoothing: spec = octave_smooth(spec, f_bins)
            t_bin_length = (datetime.strptime(file_path.split('/')[-1].split('_')[2], '%Y-%m-%d %H:%M:%S') - t_start).total_seconds() / t_size
            time_axis = [t_start + timedelta(seconds=t_bin_length * j) for j in range(t_size)]
            col_offset = cumulative_offsets[i]
            y_edges = np.linspace(f_bins[0], f_bins[-1], f_size+1)
            x_edges = np.arange(col_offset, col_offset + t_size + 1)
            im = ax.pcolormesh(x_edges, y_edges, spec, cmap='bwr' if norm else 'jet',
                        vmin=spec_min, vmax=spec_max, shading='auto')
            ax.set_yscale('log')
            if not hasattr(ax, 'all_time_axes'):
                ax.all_time_axes = []
            ax.all_time_axes.append((col_offset, time_axis))
            pbar.update(1)
        
    if tides or len(weather) > 0:
        spec_times = []
        spec_positions = []
        for col_offset, time_axis in getattr(ax, 'all_time_axes', []):
            for idx, t in enumerate(time_axis):
                spec_times.append(pd.Timestamp(t))
                spec_positions.append(col_offset + idx)
        spec_times = pd.Series(spec_positions, index=spec_times)
        
        if tides:
            ax1 = ax.twinx()
            tidal_df = pd.read_csv('./results/checkpoints/CRO_final.csv', parse_dates=True, index_col=0)
            tidal_df = tidal_df[t0:t1]

            tidal_x = []
            for ts in tidal_df.index:
                ts = pd.Timestamp(ts)
                diffs = np.abs(spec_times.index - ts)
                if len(diffs) == 0:
                    tidal_x.append(np.nan)
                else:
                    nearest_idx = diffs.argmin()
                    tidal_x.append(spec_times.iloc[nearest_idx])
            tidal_y = tidal_df['ASLVBG02']
            valid = ~np.isnan(tidal_x)
            ax1.plot(np.array(tidal_x)[valid], tidal_y[valid], color='k', alpha=0.5)
            ax1.set_ylabel('Tidal Height (m)')
            plt.sca(ax)
        if len(weather) > 0:
            for weather_ax, (col, label) in zip(axs[1:], weather):
                weather_x = []
                for ts in weather_df.index:
                    ts = pd.Timestamp(ts)
                    diffs = np.abs(spec_times.index - ts)
                    if len(diffs) == 0:
                        weather_x.append(np.nan)
                    else:
                        nearest_idx = diffs.argmin()
                        weather_x.append(spec_times.iloc[nearest_idx])
                valid = ~np.isnan(weather_x)
                weather_ax.plot(np.array(weather_x)[valid], weather_df[col][valid])
                weather_ax.set_ylabel(label)
            plt.sca(ax)
    
    day_starts = []
    for col_offset, time_axis in getattr(ax, 'all_time_axes', []):
        day_starts.extend([(col_offset + idx, t) for idx, t in enumerate(time_axis)])
    
    positions, times = zip(*day_starts)
    times = pd.Series(times, index=positions)

    day_starts = times[times.dt.hour == 0]
    minor_tick_positions = day_starts.index
    
    if markers: 
        marker_positions = []
        for m in markers:
            diffs = np.abs(day_starts.values - np.datetime64(m))
            if len(diffs) > 0:
                idx = diffs.argmin()
                marker_positions.append(day_starts.index[idx])
        ax.plot(marker_positions, [45]*len(marker_positions), 'kv')

    months = pd.Series(times.dt.to_period('M').unique())
    major_tick_positions = []
    major_tick_labels = []
    for month in months:
        month_days = day_starts[day_starts.dt.to_period('M') == month]
        if len(month_days) > 0:
            idxs = np.linspace(0, len(month_days)-1, 3, dtype=int)[:-1]
            for i in idxs:
                major_tick_positions.append(month_days.index[i])
                major_tick_labels.append(month_days.iloc[i].strftime('%Y-%m-%d'))

    ax.set_xticks(major_tick_positions, major_tick_labels)
    ax.set_xticks(minor_tick_positions, minor=True)
    ax.tick_params(axis='x', which='minor', length=3, labelsize=0)
    ax.set_ylabel('Frequency (Hz)')
    
    if len(weather) < 0:
        # axs[0].xaxis.set_ticks_position('both')
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='x', which='major', top=True, labeltop=True, bottom=True, labelbottom=True)
        ax.set_xticks(major_tick_positions)
        ax.set_xticklabels(major_tick_labels, rotation=30)
    
    fig.colorbar(im, ax=axs[-1] if len(weather) > 0 else ax, label='Nano strainrate PSD (dB)', pad=0.40 if len(weather) > 0 else 0.2, aspect=40, orientation="horizontal")
    ax.set_title(rf"{t0.date()} to {t1.date()} at channel {target_cha}")
    plt.grid(ax, which='both' if delta <= 2419200 else 'major', linewidth=0.1, alpha=0.5)
    plt.subplots_adjust(hspace=0)
    plt.xticks(rotation=90)
    if len(weather) > 0:
        axs[1].grid(which='major')
        axs[2].grid(which='major')
        plt.sca(axs[-1])
        plt.xticks(rotation=90)
    else: 
        plt.tight_layout()
    f_name = f'./results/figures/adapted_specs/spec_{t0}_{t1}_{target_cha}{"_tides" if tides else ""}{"_norm" if norm else ""}{"_smooth" if octave_smoothing else ""}.png'
    plt.savefig(f_name, bbox_inches='tight')
    plt.close()
    del day_starts
    gc.collect()


def get_spectral_mean(target_cha=None, t0=None, t1=None):
    files = get_spec_files('/data/localraid/saved_specs/sparse_f/', t0, t1, target_cha)
    
    mean_df = pd.DataFrame()
    for i, (file_path, t_start) in enumerate(files):
        spec = np.loadtxt(file_path, delimiter=',')
        means = np.nanmean(spec, axis=1)
        mean_df = pd.concat([mean_df, pd.Series([t_start.date(), *means])], axis=1)
    mean_df.to_csv(f'./results/checkpoints/sparse_monthly_means_{target_cha}.csv', header=False)


def calc_angle_between_points(lat1, lon1, lat2, lon2):
    '''Input lat-lons as degrees!!!'''
    lat1, lon1, lat2, lon2 = radians(lat1), radians(lon1), radians(lat2), radians(lon2)
    
    dlon = lon2 - lon1
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    
    angle = degrees(atan2(y, x))
    angle = (angle + 360) % 360
    return angle


def sensitivity_analysis(gps_track:pd.DataFrame, target_ch, plot=True):
    angles = []
    for current_ch in gps_track.index:
        if current_ch == gps_track.index[0]:
            current_ch += 1
        elif current_ch == gps_track.index[-1]:
            current_ch -= 1
        lat1, lon1 = gps_track.loc[current_ch - 1, ['lat', 'lon']]
        lat2, lon2 = gps_track.loc[current_ch + 1, ['lat', 'lon']]
        angles.append(calc_angle_between_points(lat1, lon1, lat2, lon2))
    
    gps_track['relative_angle'] = angles
    target_angle = gps_track.loc[current_ch, 'relative_angle']
    relative_angles = []
    long_sens = []
    trans_sens = []
    for current_ch in gps_track.index:
        angle = gps_track.loc[current_ch, 'relative_angle'] - target_angle
        relative_angles.append(abs(angle))
        long_sens.append(abs(cos(radians(angle))))
        trans_sens.append(sin(2*radians(angle)) ** 2)
    
    if plot:
        fig, axs = plt.subplots(3, 1, figsize=(15, 10))
        for i, arr in enumerate([relative_angles, long_sens, trans_sens]):
            im = axs[i].scatter(gps_track['lon'], gps_track['lat'], c=arr, cmap='seismic')
            fig.colorbar(im, ax=axs[i])
            axs[i].scatter(gps_track.loc[target_ch, 'lon'], gps_track.loc[target_ch, 'lat'], s=100, c='k')
        plt.tight_layout()
        plt.show()
    
    return np.sum(long_sens), np.sum(trans_sens)


if __name__ == '__main__':
    parallel_spectral_analysis()
    
    markers_arr = [
        # datetime(2024, 5, 30),
        datetime(2024, 4, 6),       # Kathleen
        datetime(2024, 8, 22),      # Lilian
        # datetime(2024, 9, 27),
        # datetime(2024, 10, 10),
        datetime(2024, 10, 20),     # Ashley
        datetime(2024, 11, 22),     # Bert
        datetime(2024, 11, 27),     # Conall
        datetime(2024, 12, 6),      # Darragh
        datetime(2025, 1, 24),      # Eowyn
        # datetime(2025, 2, 7),
        # datetime(2025, 3, 11),
        # datetime(2025, 4, 2),
    ]
    
    # t0 = datetime(year=2024, month=4, day=1); t1 = datetime(year=2025, month=7, day=1)
    # for cha in [750, 788, 875, 1475]:
    #     get_spectral_mean(cha, t0, t1)
    
    weather_arr = [['Wind(m/s)','Wind speed (m/s)'],['Hs(Hm0)(m)','Wave height (m)']]
    # f_ranges = [[0.01, 0.05], [0.1, 0.6], [0.7, 1.1]]       # [0.01, 0.05], [0.1, 0.6], [0.7, 1.1], [1.2, 5], [8, 11], [12, 25]
    # avg_time = timedelta(days=5)
    c_range_arr = [-30.128117506053417, 38.67278348993597]              # for all non-norm specs 2024-04 to 2025-07
    c_range_norm_arr = [-27.121361009693757, 26.618877627443347]        # for norm specs
    
    c_range_nodes_arr = [-32.7434695431312, 33.040064418459345]
    c_range_norm_nodes_arr = [-23.176340627918208, 24.762160092331]
    
    t0 = datetime(year=2024, month=12, day=8); t1 = datetime(year=2025, month=12, day=11)
    # spectral_power_ts(f_ranges, avg_time, target_cha=788, t0=t0, t1=t1)
    # for cha in [750, 788, 875, 1475]:
        # plot_spectrogram(target_cha=cha, t0=t0, t1=t1)
    #     plot_spectrogram(target_cha=cha, t0=t0, t1=t1, norm=True, markers=markers_arr, weather=weather_arr, c_range=c_range_norm_arr)
    
    # t0 = datetime(year=2025, month=4, day=1); t1 = datetime(year=2025, month=4, day=10)
    # # spectral_power_ts(f_ranges, avg_time, target_cha=788, t0=t0, t1=t1)
    # for cha in [750, 788, 875, 1475]:
    #     plot_spectrogram(target_cha=cha, t0=t0, t1=t1, c_range=c_range_nodes_arr)
        # plot_spectrogram(target_cha=cha, t0=t0, t1=t1, norm=True, markers=markers_arr, weather=weather_arr, c_range=c_range_norm_nodes_arr)
    
    ### tidal plots
    # t0 = datetime(year=2025, month=1, day=1); t1 = datetime(year=2025, month=2, day=1)
    # for i in range(7):
    #     plot_spectrogram(target_cha=788, t0=t0, t1=t1, tides=True)
    #     t0 += relativedelta(months=1)
    #     t1 += relativedelta(months=1)
    
    
    ############################ ARCHIVE ############################
    # dir_path = "/data/localraid/20250208/"
    # task_t0 = datetime(year = 2025, month = 2, day = 8, 
    #                    hour = 12, minute = 7, second = 53, microsecond = 0)
    # dir_path = "/data/localraid/20250108/"
    # task_t0 = datetime(year = 2025, month = 1, day = 8, 
    #                    hour = 12, minute = 0, second = 23, microsecond = 0)
    # dir_path = "/data/localraid/20241208/"
    # task_t0 = datetime(year = 2024, month = 12, day = 8, 
    #                    hour = 12, minute = 7, second = 36, microsecond = 0)
    
    
    # corr_path = './results/saved_corrs/2024-02-05 12:01:00_4320mins_f0.01:49.9__3850:5750_1m.txt'
    # stream = load_xcorr(corr_path, as_stream=True)
    # from obspy import read, UTCDateTime, Stream
    
    # corr_path = './results/saved_corrs/2024-02-05 12:01:00_4320mins_f0.01:49.9__3850:5750_1m.txt'
    # stream = load_xcorr(corr_path, as_stream=True)
    
    # from obspy import read, UTCDateTime, Stream
    # dx = 1.0
    # for i in range(0, len(stream)):
    #     stream[i].stats.distance = i*dx
    # stream.filter("bandpass", freqmin=5, freqmax=50)
    # stream.plot(type='section', recordstart=6, recordlength=4, fillcolors=('k', None))
    
    
    ### CAUSAL | ACAUSAL SPLIT
    # causal = stream.copy()
    # causal.trim(starttime=UTCDateTime("19700101T00:00:08"))
    # causal.plot(type='section', recordlength=2, fillcolors=('k', None))

    # acausal = stream.copy()plt.xticks(rotation=90)
    # acausal.trim(endtime=UTCDateTime("19700101T00:00:08"))
    # for tr in acausal: tr.data = np.flip(tr.data)
    # acausal.plot(type='section', recordlength=2, fillcolors=('k', None))
    
    # plot_weather()
    # plot_rain_storms()
    # plot_era5_data('era5_rainfall.grib')
    # plot_era5_data('./results/checkpoints/daily_rainfall.csv')
    # plot_era5_data('era5_windspeed.grib')
    # plot_era5_data('./results/checkpoints/daily_windspeed.csv')
    # plot_tidal_data('./results/checkpoints/2024CRO.txt')
    
    # gps_coords = pd.read_csv('results/checkpoints/interp_ch_pts.csv', sep=',', index_col=2)
    # long_max, trans_max = 0, 0
    # long_max_ch, trans_max_ch = 0, 0
    # for ch in gps_coords.index:
    #     long_total, trans_total = sensitivity_analysis(gps_coords, ch, plot=False)
    #     if long_total > long_max:
    #         long_total = long_max
    #         long_max_ch = ch
    #     if trans_total > trans_max:
    #         trans_total = trans_max
    #         trans_max_ch = ch
    # print(f'{long_max_ch}: {long_max}')
    # print(f'{trans_max_ch}: {trans_max}')
