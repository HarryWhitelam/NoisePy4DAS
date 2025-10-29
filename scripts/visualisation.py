import os
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
from skimage.util import compare_images
import contextily as cx
from math import ceil, sin, cos, atan2, degrees, radians
import multiprocessing
from tqdm import tqdm

from tdms_io import get_reader_array, get_filepath_array, get_data_from_array, get_dir_properties, load_xcorr


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


def parallel_spectral_analysis():
    dir_list = ['/data/QNAP1_Data/Data/']
    for m in [1,2,3,4,5,6,7]:
        t_start = datetime(year=2025, month=m, day=1)
        t_end = t_start + relativedelta(months=1)
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
                #                 dir_path, prepro_para, t_start, save_spec, data, plot_tides
                args_list.append((dir_path, run_prepro_para, t_start, True, channel_data, False))
        
            p = multiprocessing.Pool(multiprocessing.cpu_count())
            with tqdm(total=len(args_list), desc=f"{dir_path} spectrograms", position=0) as pbar:
                for _ in p.starmap(ts_spectrogram, args_list, chunksize=1):
                    pbar.update(1)
            p.close()
            
            # p = multiprocessing.Pool(multiprocessing.cpu_count())
            # with tqdm(total=len(args_list), desc=f"{dir_path} PPSDs", position=0) as pbar:
            #     for _ in p.starmap(ppsd, args_list, chunksize=1):
            #         pbar.update(1)
            # p.close()

def ts_spectrogram(dir_path:str, prepro_para:dict, t_start:datetime, save_spec=False, data=None, plot_tides=None):
    cha1, sps, freqmin, freqmax, n_minute = prepro_para.get('cha1'), prepro_para.get('target_sps'), prepro_para.get('freqmin'), prepro_para.get('freqmax'), prepro_para.get('n_minute')
    
    out_dir = f"./results/figures/PSD_Experiments/dense_f/"
    
    if type(data)==type(None):
        reader_array, timestamps = get_reader_array(dir_path)
        if t_start == None: t_start = timestamps[0].replace(microsecond=0)
        data = get_data_from_array(reader_array, prepro_para, t_start, timestamps, duration=timedelta(minutes=n_minute))[:, 0]
    
    N = data.shape[0]
    win = hamming(int(sps*600), sym=True)               # 28/10/2025 longer window, longer time averaging! Output name changed!
    # g_std = 12
    # gaussian_win = gaussian(sps*60, g_std, sym=True)
    stft = ShortTimeFFT(win, hop=int(sps*540), fs=sps, scale_to='psd')
    spec = stft.spectrogram(data)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    t_min, t_max = stft.extent(N)[:2]
    ax.set_title(rf"{t_start} at channel {cha1}")
    spec = 10 * np.log10(spec + 1e-12)
    ext = stft.extent(N)
    im1 = ax.imshow(spec, origin='lower', aspect='auto', 
                     extent=ext, cmap='jet', vmin=np.nanpercentile(spec,1), vmax=np.nanpercentile(spec,99))
    ax.set_yscale('log')
    plt.grid(which='both')
    
    if plot_tides:
        ax1 = ax.twinx()
        # print(ax.get_xticks())
        # print(ax1.get_xticks())
        tidal_df = pd.read_csv('./results/checkpoints/CRO_final.csv', parse_dates=True, index_col=0)
        tidal_df = tidal_df[t_start:t_start+timedelta(minutes=n_minute)]
        # [(ts - t_start).total_seconds() / 60 for ts in tidal_df.index]
        ax1.plot([(ts - t_start).total_seconds() for ts in tidal_df.index], tidal_df['ASLVBG02'])
        ax1.set_ylabel('Tidal Height (m)')
        plt.sca(ax=ax)
    
    if n_minute > 1440:
        n_days = int(n_minute / 1440) + 1
        midnight_start = t_start.replace(hour=0, minute=0, second=0, microsecond=0)
        tick_dates = pd.date_range(midnight_start, periods=n_days, freq=timedelta(days=(n_days//6)))
        
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
    plt.savefig(f'{out_dir}/{t_start}__{t_start+timedelta(minutes=n_minute)}_f{freqmin}:{freqmax}_{cha1}_long_spectrogram{"_tides" if plot_tides else ""}.png')
    if save_spec:
        np.savetxt(f'./results/saved_specs/dense_f/{t_start}__{t_start+timedelta(minutes=n_minute)}_f{freqmin}:{freqmax}_{cha1}_spec.txt', spec, delimiter=",")
    # plt.show()


def ppsd(dir_path:str, prepro_para:dict, t_start:datetime, save_ppsd=False, data=None):
    cha1, sps, freqmin, freqmax, n_minute = prepro_para.get('cha1'), prepro_para.get('target_sps'), prepro_para.get('freqmin'), prepro_para.get('freqmax'), prepro_para.get('n_minute')
    out_dir = f"./results/figures/PSD_Experiments/"
    
    if type(data)==type(None):
        reader_array, timestamps = get_reader_array(dir_path)
        if t_start == None: t_start = timestamps[0].replace(microsecond=0)
        data = get_data_from_array(reader_array, prepro_para, t_start, timestamps, duration=timedelta(minutes=n_minute))[:, 0]

    nfft = 2 ** 17
    nr = 501
    hn = nfft // 2
    
    # First pass: collect sample of PSD values to determine range
    sample_fd_values = []
    seg_length = int(60 * sps)
    n_segs = len(data) // seg_length
    sample_segs =  n_segs // 10 # Sample first 10% of segments
    
    for i in range(sample_segs):
        d = data[i*seg_length:(i+1)*seg_length]
        if len(d) == seg_length:
            fft_d = np.fft.fft(d, nfft)
            psd_lin = (np.abs(fft_d) ** 2) / (nfft * sps)
            fd = 10 * np.log10(psd_lin + 1e-12)
            sample_fd_values.extend(fd[:hn].flatten())
    
    # Calculate adaptive range using percentiles
    fd_min = np.nanpercentile(sample_fd_values, 0.01)
    fd_max = np.nanpercentile(sample_fd_values, 99.99)
    # print(f"Adaptive dB range: {fd_min:.1f} to {fd_max:.1f}")
    
    # Create adaptive binning
    db_range = fd_max - fd_min
    scale = (nr - 1) / db_range
    offset = -fd_min
    
    # Second pass: bin with adaptive scaling
    psd = np.zeros((nr, hn))
    p = np.zeros(hn)
    
    for i in range(n_segs):
        d = data[i*seg_length:(i+1)*seg_length]
        ### current fix for NaNs is to skip any section containing NaNs, not exactly robust >:|
        if len(d) == seg_length and ~np.isnan(d).any():
            fft_d = np.fft.fft(d, nfft)
            psd_lin = (np.abs(fft_d) ** 2) / (nfft * sps)
            fd = 10 * np.log10(psd_lin + 1e-12)
            
            p += fd[:hn]
            for j in range(hn):
                index = int((fd[j] + offset) * scale)
                if index < 0:
                    index = 0
                elif index >= nr:
                    index = nr - 1
                psd[index, j] += 1
    
    # Create proper dB axis
    db = np.linspace(fd_min, fd_max, nr)
    # print(f'down: {down_clip}; up: {up_clip}')
    
    f1 = freqmin; f2 = freqmax
    f = np.arange(nfft) * sps / (nfft)
    fn1 = int(f1*nfft/sps); fn2 = int(f2*nfft/sps)
    
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

    plt.figure()
    plt.pcolormesh(f, db, P*100, cmap='viridis')
    plt.xscale('log')
    plt.grid(which='both')
    plt.colorbar(shrink=0.75, aspect=30, pad=0.05, extend='both', label=r'Probability (%)')
    # cbar.ax.tick_params(labelsize=16)
    plt.semilogx(f, pp, lw=1.2, color='#888888')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Nano strainrate PSD (dB)')
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    plt.title(rf"{t_start} at channel {cha1}")
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{t_start}__{t_start+timedelta(minutes=n_minute)}_f{freqmin}:{freqmax}_{cha1}_PPSD.png')


def spectral_power_ts(f_ranges:list, window_length:timedelta, target_cha=None, t0=None, t1=None):
    ### plan
    # load saved_specs
    files = []
    for dir_path, dir_names, file_names in os.walk('./results/saved_specs/months/'):
        for file in file_names:
            try:
                f_split = file.split('_')
                t_start = datetime.strptime(f_split[0], '%Y-%m-%d %H:%M:%S')
                t_end = datetime.strptime(f_split[2], '%Y-%m-%d %H:%M:%S')
                if int(f_split[-2]) != target_cha: continue
                if t0 and t_start <= t0: continue
                if t1 and t_end >= t1: continue
            except Exception: 
                continue        # basically there was no datetime within the file
            files.append((os.path.join(dir_path, file), t_start))
    
    mean_df = pd.DataFrame(columns=['timestamp', *[f'{f_range[0]}_{f_range[1]}' for f_range in f_ranges]]).set_index('timestamp')
    cosine_window = np.cos(np.linspace(-np.pi/2, np.pi/2, int(window_length.total_seconds() / 54)))
    cosine_window = cosine_window / cosine_window.sum()
    for file_path, t_start in files:
        spec = np.loadtxt(file_path, delimiter=',')
        f_size, t_size = spec.shape
        
        # (0.0, 259254.0, 0.0, 50.016666666666666)
        # delta_t = timedelta(seconds=54)
        # slices_per_chunk = int(avg_time.total_seconds() / delta_t.total_seconds())
        # n_chunks = spec.shape[1] // slices_per_chunk
        f_bins = np.linspace(0.0, 50.016666666666666, f_size)
        
        for f_min, f_max in f_ranges:
            f_idxs = [i for i, f in enumerate(f_bins) if f_min <= f <= f_max]
            freq_mean = spec[f_idxs, :].mean(axis=0)
            filtered = np.convolve(freq_mean, cosine_window, mode='same')
            for j in range(t_size):
                t = t_start + timedelta(seconds=54 * j)
                mean_df.loc[t, f'{f_min}_{f_max}'] = filtered[j]
    
    # mean_df = pd.DataFrame(mean_fs)
    mean_df.sort_index(inplace=True)
    mean_df.plot(figsize=(10,3))
    plt.tight_layout()
    plt.savefig(f'./results/figures/ts_spec_lines/high_f_{t0}_{t1}_{window_length.total_seconds()/60}min_avg.png')
    # plt.show()


def plot_year_spectrogram(target_cha=None, t0=None, t1=None):
    files = []
    for dir_path, dir_names, file_names in os.walk('./results/saved_specs/months/'):
        for file in file_names:
            try:
                f_split = file.split('_')
                t_start = datetime.strptime(f_split[0], '%Y-%m-%d %H:%M:%S')
                t_end = datetime.strptime(f_split[2], '%Y-%m-%d %H:%M:%S')
                if target_cha and int(f_split[-2]) != target_cha: continue
                if t0 and t_start <= t0: continue
                if t1 and t_end >= t1: continue
            except Exception: 
                continue        # basically there was no datetime within the file
            files.append((os.path.join(dir_path, file), t_start))
    files.sort(key=lambda x: x[1])
    
    nrows = 4
    nfiles = len(files)
    ncols = int(np.ceil(nfiles / nrows))
    fig, axs = plt.subplots(nrows, 1, figsize=(ncols*8, nrows*4), sharey=True)
    
    time_bin_counts = [np.loadtxt(f[0], delimiter=',').shape[1] for f in files]
    cumulative_offsets = np.cumsum([0] + time_bin_counts[:-1])
    for i, (file_path, t_start) in enumerate(files):
        row = i // ncols
        spec = np.loadtxt(file_path, delimiter=',')
        f_size, t_size = spec.shape
        f_bins = np.linspace(0.01, 50.016666666666666, f_size)
        time_axis = [t_start + timedelta(seconds=54 * j) for j in range(t_size)]
        col_offset = cumulative_offsets[i]
        y_edges = np.linspace(f_bins[0], f_bins[-1], f_size+1)
        x_edges = np.arange(col_offset, col_offset + t_size + 1)
        axs[row].pcolormesh(x_edges, y_edges, spec, cmap='jet',
                            vmin=np.nanpercentile(spec, 1), vmax=np.nanpercentile(spec, 99), shading='auto')
        axs[row].set_yscale('log')
        if not hasattr(axs[row], 'all_time_axes'):
            axs[row].all_time_axes = []
        axs[row].all_time_axes.append((col_offset, time_axis))

    for ax in axs:
        all_times = []
        for col_offset, time_axis in getattr(ax, 'all_time_axes', []):
            all_times.extend([(col_offset + idx, t) for idx, t in enumerate(time_axis)])
        if not all_times:
            continue
        positions, times = zip(*all_times)
        times = pd.Series(times, index=positions)

        # Minor ticks: every day (unlabeled)
        day_starts = times[times.dt.hour == 0]
        minor_tick_positions = day_starts.index

        # Major ticks: ~4 evenly spaced dates per month (at midnight)
        months = pd.Series(times.dt.to_period('M').unique())
        major_tick_positions = []
        major_tick_labels = []
        for month in months:
            month_days = day_starts[day_starts.dt.to_period('M') == month]
            if len(month_days) > 0:
                # Pick 4 evenly spaced days in this month
                idxs = np.linspace(0, len(month_days)-1, 5, dtype=int)[:-1]
                for i in idxs:
                    major_tick_positions.append(month_days.index[i])
                    major_tick_labels.append(month_days.iloc[i].strftime('%Y-%m-%d'))

        ax.set_xticks(major_tick_positions)
        ax.set_xticklabels(major_tick_labels, rotation=30)
        ax.set_xticks(minor_tick_positions, minor=True)
        ax.tick_params(axis='x', which='minor', length=4, labelsize=0)
        ax.set_ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.savefig(f'./results/figures/yearspec_{t0}_{t1}.png')


def get_spectral_mean(target_cha=None, t0=None, t1=None):
    files = []
    for dir_path, dir_names, file_names in os.walk('./results/saved_specs/months/'):
        for file in file_names:
            try:
                f_split = file.split('_')
                t_start = datetime.strptime(f_split[0], '%Y-%m-%d %H:%M:%S')
                t_end = datetime.strptime(f_split[2], '%Y-%m-%d %H:%M:%S')
                if target_cha and int(f_split[-2]) != target_cha: continue
                if t0 and t_start <= t0: continue
                if t1 and t_end >= t1: continue
            except Exception: 
                continue        # basically there was no datetime within the file
            files.append((os.path.join(dir_path, file), t_start))
    files.sort(key=lambda x: x[1])
    
    mean_df = pd.DataFrame()
    for i, (file_path, t_start) in enumerate(files):
        spec = np.loadtxt(file_path, delimiter=',')
        means = np.nanmean(spec, axis=1)
        mean_df = pd.concat([mean_df, pd.Series([t_start, *means])], axis=1)
    mean_df.to_csv('./results/checkpoints/monthly_means.csv')


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
    # parallel_spectral_analysis()
    
    # f_ranges = [[1.2, 5], [8, 11], [12, 25]]       # [0.01, 0.05], [0.1, 0.6], [0.7, 1.1], [1.2, 5], [8, 11], [12, 25]
    # avg_time = timedelta(days=2)
    t0 = datetime(year=2024, month=7, day=1); t1 = datetime(year=2025, month=7, day=2)
    # spectral_power_ts(f_ranges, avg_time, target_cha=788, t0=t0, t1=t1)
    plot_year_spectrogram(target_cha=788, t0=t0, t1=t1)
    # get_spectral_mean(788, t0, t1)
    
    
    
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

    # acausal = stream.copy()
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
