import os
import numpy as np
import psutil
import pandas as pd
import geopandas as gpd
from scipy.signal import welch, ShortTimeFFT, decimate, convolve2d, savgol_filter
from scipy.signal.windows import gaussian, hamming
from scipy.fft import rfft, rfftfreq
from obspy.signal.filter import bandpass
from obspy.signal.spectral_estimation import get_nlnm, get_nhnm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from skimage.util import compare_images
import contextily as cx
from math import ceil, sin, cos, atan2, degrees, radians, log, pi
import multiprocessing
from tqdm import tqdm
from time import time

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


def psd_with_channel_slicing(reader_array, prepro_para, task_t0, timestamps, channels):
    fig = plt.figure(figsize=(15, 10))
    
    prepro_para['cha1'], prepro_para['cha2'] = channels[0], channels[1]
    tdata = get_data_from_array(reader_array, prepro_para, task_t0, timestamps, timedelta(minutes=1))
    
    N = 60000
    yf = rfft(tdata.T)
    yf /= 1/prepro_para.get('sps')
    yf = (2*prepro_para.get('sps')/N) * abs(yf**2)
    yf = 10 * np.log10(yf)
    xf = rfftfreq(N, 1/prepro_para.get('sps'))

    #make figure logarithmic
    ax = fig.add_subplot()
    ax.set_xscale('log')
    # ax.set_ylim(None, 100)

    plt.ylabel('Amplitude (db)')
    plt.xlabel('Frequency [Hz]')

    plt.plot(xf, yf.T)
    
    nlnm_freq, nlnm_psd = get_nlnm()
    nhnm_freq, nhnm_psd = get_nhnm()
    plt.plot(nlnm_freq, nlnm_psd, label="NLNM", linestyle="dashed")
    plt.plot(nhnm_freq, nhnm_psd, label="NHNM", linestyle="dashed")
    plt.show()
    
    
    freqs, psd = welch(tdata.T, fs=prepro_para.get('sps'))
    plt.semilogy(freqs, psd.T, color='b')
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density [V**2/Hz]')
    
    # ax.set_xlim(freqs[0], freqs[-1])
    plt.xlim(freqs[0], 50)
    # ax.set_ylim(1e8, max(psd)*2)
    plt.show()


def animated_spectrogram(reader_array, prepro_para, task_t0, timestamps):
    def update(channel_idx):
        channel_data = tdata[:, channel_idx]
        
        freqs, psd = welch(channel_data.T, fs=prepro_para.get('sps'))
        # plt.semilogy(freqs, psd, color='b')
        line.set_data(freqs, psd)
        title.set_text(f'Power Spectral Density (Channel {prepro_para.get("cha1") + (channel_idx * prepro_para.get("spatial_ratio"))})')
        return line, title
    
    n_channels = ceil((prepro_para.get('cha2') - prepro_para.get('cha1') + 1) / prepro_para.get('spatial_ratio'))
    tdata = get_data_from_array(reader_array, prepro_para, task_t0, timestamps)
    freqs, psd = welch(tdata[:, 0].T, fs=prepro_para.get('sps'))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.semilogy([], [], color='b')
    
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power Spectral Density [V**2/Hz]')
    ax.grid(True)
    
    # ax.set_xlim(freqs[0], freqs[-1])
    ax.set_xlim(freqs[0], 50)
    ax.set_ylim(1e8, max(psd)*2)
    
    title = ax.text(0.5, 1.05, "Test start", transform=ax.transAxes, ha="center")
    
    ani = FuncAnimation(
        fig, 
        update,
        frames=n_channels,
        interval=75,
        repeat=True,
    )
    # plt.show()
    # file name format: psd_cha1:cha2_spatial_res.gif or something like that
    ani.save(f'psd_{prepro_para.get("cha1")}:{prepro_para.get("cha2")}_{prepro_para.get("spatial_ratio")*0.25}m.gif', writer='pillow')


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


def spectral_comparison(data_dict, fs, ncols=2, subplots=False, find_nearest=False):
    if find_nearest:
        fft_arr = []    
    nrows = len(data_dict) // ncols + (len(data_dict) % ncols > 0)
    fig = plt.figure(figsize=(15, 12))
    
    for n, (key, val) in enumerate(data_dict.items()):
        val = val.mean(axis=1)
        freqs, psd = welch(val.T, fs=fs)
        if find_nearest:
            fft_arr.append([key, freqs, psd])
        if subplots:
            ax = plt.subplot(nrows, ncols, n + 1)
            ax.semilogy(freqs, psd, label=f'test {key}')
            ax.title.set_text(key)
        else:
            plt.semilogy(freqs, psd, label=key)

    if find_nearest:
        dists = [np.linalg.norm(fft[2] - fft_arr[0][2]) for fft in fft_arr[1:]]
        print(f'Closest spectrogram is {fft_arr[dists.index(min(dists))+1][0]}')
    
    plt.legend()
    fig.tight_layout()
    plt.show()

    
def numerical_comparison(data_dict):
    df = pd.DataFrame(columns=['id', 'mean', 'std'])
    df['id'] = list(data_dict.keys())
    df['mean'] = [data.mean() for data in data_dict.values()]
    df['std'] = [data.std() for data in data_dict.values()]
    print(df)
    
    for col in df.columns[1:]:
        closest = df.loc[(df[col][1:] - df[col][0]).abs().idxmin()]['id']
        print(f'Closest {col}: {closest}')


def parallel_spectral_analysis():
    dir_list = ['/data/QNAP1_Data/Data/']
    
    t_start = datetime(year=2025, month=4, day=1)
    t_end = datetime(year=2025, month=5, day=1)
    
    for m in [1,2,3,5,6,7]:
        t_start = datetime(year=2025, month=m, day=1)
        t_end = datetime(year=2025, month=m+1, day=1)
        n_minutes = (t_end - t_start).total_seconds() // 60
        
        channels = [750, 788, 875, 1475]  # removed [1550, 1550]
        
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
                #                 dir_path, prepro_para, t_start, save_spec, mem_check, data
                args_list.append((dir_path, run_prepro_para, t_start, True, False, channel_data))
        
            p = multiprocessing.Pool(multiprocessing.cpu_count())
            with tqdm(total=len(args_list), desc=f"{dir_path} spectrograms", position=0) as pbar:
                for spec in p.starmap(ts_spectrogram, args_list, chunksize=1):
                    pbar.update(1)
            p.close()
            
            p = multiprocessing.Pool(multiprocessing.cpu_count())
            with tqdm(total=len(args_list), desc=f"{dir_path} PPSDs", position=0) as pbar:
                for psd in p.starmap(ppsd, args_list, chunksize=1):
                    pbar.update(1)
            p.close()

def ts_spectrogram(dir_path:str, prepro_para:dict, t_start:datetime, save_spec=False, mem_check=False, data=None):
    if mem_check:
        process = psutil.Process(os.getpid())
        mem_dict = {'Start': process.memory_info().rss / (1024 ** 2)}
    cha1, sps, freqmin, freqmax, n_minute = prepro_para.get('cha1'), prepro_para.get('target_sps'), prepro_para.get('freqmin'), prepro_para.get('freqmax'), prepro_para.get('n_minute')
    
    out_dir = f"./results/figures/PSD_Experiments/"
    
    if type(data)==type(None):
        reader_array, timestamps = get_reader_array(dir_path)
        if t_start == None: t_start = timestamps[0].replace(microsecond=0)
        data = get_data_from_array(reader_array, prepro_para, t_start, timestamps, duration=timedelta(minutes=n_minute))[:, 0]
    
    if mem_check: mem_dict.update({'Data loaded': process.memory_info().rss / (1024 ** 2)})
    
    data = np.float32(bandpass(data,
                            0.9 * freqmin,
                            freqmax,
                            df=sps,
                            corners=4,
                            zerophase=True))
    
    N = data.shape[0]
    win = hamming(int(sps*60), sym=True)
    # g_std = 12
    # gaussian_win = gaussian(sps*60, g_std, sym=True)
    stft = ShortTimeFFT(win, hop=int(sps*54), fs=sps, scale_to='psd')
    spec = stft.spectrogram(data)
    if mem_check: mem_dict.update({'Spec gen': process.memory_info().rss / (1024 ** 2)})

    fig = plt.figure()
    ax = fig.add_subplot(111)
    t_min, t_max = stft.extent(N)[:2]
    ax.set_title(rf"{t_start} at channel {cha1}")
    # print(f'spec max: {spec.max()}; spec min: {spec.min()}')
    spec = 10 * np.log10(spec + 1e-12)
    ext = stft.extent(N)
    # print(ext)
    # print(f't slices: {stft.p_num(N)}, delta t: {stft.delta_t}')
    # print(f'f bins: {stft.f_pts}, delta f: {stft.delta_f}')
    im1 = ax.imshow(spec, origin='lower', aspect='auto', 
                     extent=ext, cmap='jet', vmin=np.percentile(spec,1), vmax=np.percentile(spec,99))
    ax.set_yscale('log')
    plt.grid(which='both')
    
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
    # fig.colorbar(im1, label='PSD ' + r"$20\,\log_{10}|S_x(t, f)|$ in dB")
    fig.colorbar(im1, label='Nano strainrate PSD (dB)')
    plt.tight_layout()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(f'{out_dir}/{t_start}__{t_start+timedelta(minutes=n_minute)}_f{freqmin}:{freqmax}_{cha1}_spectrogram.png')
    if save_spec:
        np.savetxt(f'./results/saved_specs/{t_start}__{t_start+timedelta(minutes=n_minute)}_f{freqmin}:{freqmax}_{cha1}_spec.txt', spec, delimiter=",")
    if mem_check: 
        mem_dict.update({'End': process.memory_info().rss / (1024 ** 2)})
        print(f'[PID {os.getpid()}] ts_spectrogram memory usage (MB): {mem_dict}')


def ppsd(dir_path:str, prepro_para:dict, t_start:datetime, save_ppsd=False, mem_check=False, data=None):
    if mem_check:
        process = psutil.Process(os.getpid())
        mem_dict = {'Start': process.memory_info().rss / (1024 ** 2)}
    cha1, sps, freqmin, freqmax, n_minute = prepro_para.get('cha1'), prepro_para.get('target_sps'), prepro_para.get('freqmin'), prepro_para.get('freqmax'), prepro_para.get('n_minute')
    out_dir = f"./results/figures/PSD_Experiments/"
    
    if type(data)==type(None):
        reader_array, timestamps = get_reader_array(dir_path)
        if t_start == None: t_start = timestamps[0].replace(microsecond=0)
        data = get_data_from_array(reader_array, prepro_para, t_start, timestamps, duration=timedelta(minutes=n_minute))[:, 0]
    if mem_check: mem_dict.update({'Data loaded': process.memory_info().rss / (1024 ** 2)})
    
    data = np.float32(bandpass(data,
                            0.9 * freqmin,
                            1.1 * freqmax if 1.1 * freqmax < sps / 2 else freqmax,  # nyquists check
                            df=sps,
                            corners=4,
                            zerophase=True))
    

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
    fd_min = np.percentile(sample_fd_values, 0.01)
    fd_max = np.percentile(sample_fd_values, 99.99)
    # print(f"Adaptive dB range: {fd_min:.1f} to {fd_max:.1f}")
    
    # Create adaptive binning
    db_range = fd_max - fd_min
    scale = (nr - 1) / db_range
    offset = -fd_min
    if mem_check: mem_dict.update({'Data loaded': process.memory_info().rss / (1024 ** 2)})
    
    # Second pass: bin with adaptive scaling
    psd = np.zeros((nr, hn))
    p = np.zeros(hn)
    
    for i in range(n_segs):
        d = data[i*seg_length:(i+1)*seg_length]
        if len(d) == seg_length:
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
    if mem_check: mem_dict.update({'Second pass': process.memory_info().rss / (1024 ** 2)})
    
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
    if mem_check: mem_dict.update({'P gen': process.memory_info().rss / (1024 ** 2)})

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
    if mem_check: 
        mem_dict.update({'End': process.memory_info().rss / (1024 ** 2)})
        print(f'[PID {os.getpid()}] ts_spectrogram memory usage (MB): {mem_dict}')


def spectral_power_ts(f_ranges:list, avg_time:timedelta):
    ### plan
    # load saved_specs
    spec = np.loadtxt('./results/saved_specs/2024-02-05 12:01:00__2024-02-08 12:01:00_f0.01:49.9_3000_spec.txt', delimiter=',')
    # print(spec.shape)       # 3001 frequency bins, 4801 time bins
    
    # (0.0, 259254.0, 0.0, 50.016666666666666)
    # t slices: 4801, delta t: 54.0
    delta_t = timedelta(seconds=54)
    slices_per_chunk = int(avg_time.total_seconds() / delta_t.total_seconds())
    n_chunks = spec.shape[1] // slices_per_chunk
    # f bins: 3001, delta f: 0.016666666666666666
    f_bins = np.array(0.0 + np.arange(0, 3001) * 0.016666666666666666)
    mean_fs = {}
    for f_min, f_max in f_ranges:
        idxs = [i for i,f in enumerate(f_bins) if f_min <= f <= f_max]
        f_spec = spec[idxs,:]
        chunk_means = []
        for i in range(n_chunks):
            chunk = f_spec[:, i*slices_per_chunk:(i+1)*slices_per_chunk]
            chunk_means.append(chunk.mean())
        mean_fs[(f_min, f_max)] = chunk_means
    
    df = pd.DataFrame(mean_fs)
    print(df)
    df.plot()
    plt.show()
    

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
    
    # f_ranges = [[0.1, 0.6], [8, 11]]
    # avg_time = timedelta(minutes=60)
    # spectral_power_ts(f_ranges, avg_time)
    
    
    
    
    
    
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
