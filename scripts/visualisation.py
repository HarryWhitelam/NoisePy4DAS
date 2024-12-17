import tdms_io
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.signal import welch, ShortTimeFFT
from scipy.signal.windows import gaussian
from scipy.fft import rfft, rfftfreq
from obspy.signal.filter import bandpass
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from skimage.util import compare_images
import contextily as cx
from math import ceil

from tdms_io import get_reader_array, get_data_from_array


def dms_to_dd(degrees, minutes=0, seconds=0):
    return degrees + (minutes/60) + (seconds/3600)


def plot_gps_coords(file_path):
    # csv_data = np.genfromtxt(file_path, delimiter=',')
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


def psd_with_channel_slicing(reader_array, prepro_para, task_t0, timestamps, channel_slices):
    fig, axs = plt.subplots(2, ceil(len(channel_slices)/2), figsize=(15, 10))
    
    for ax, channels in zip(axs.ravel(), channel_slices):
        plt.sca(ax)
        prepro_para['cha1'], prepro_para['cha2'] = channels[0], channels[1]
        
        tdata = tdms_io.get_data_from_array(reader_array, prepro_para, task_t0, timestamps)
        print(tdata.shape)
        # f, t, Sxx = spectrogram(tdata, prepro_para.get('sps'), mode="psd")
        
        # cmap = plt.colormaps['Spectral']
        # img1 = plt.pcolormesh(t, f, Sxx, cmap=cmap.reversed(), shading='gouraud')

        # plt.title(f'{channels}')
        # plt.ylabel('Amplitude [m^2/s^4/Hz]')
        # plt.xlabel('Frequency [Hz]')
        # fig.colorbar(img1, label= "Nano Strain per Second [nm/m/s]")
        
        N = 60000
        yf = rfft(tdata.T)
        xf = rfftfreq(N, 1/prepro_para.get('sps'))

        #make figure logarithmic
        ax = fig.add_subplot()
        ax.set_xscale('log')
        ax.set_ylim(None, 10000000000)

        plt.ylabel('Nano Strain per Second [nm/m/s]')
        plt.xlabel('Frequency [Hz]')

        plt.plot(xf, np.abs(yf).T)
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
    tdata = tdms_io.get_data_from_array(reader_array, prepro_para, task_t0, timestamps)
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


def ts_spectrogram(dir_path:str, prepro_para:dict, t_start:datetime):
    out_dir = f'{t_start}_{prepro_para.get('duration')}mins_{prepro_para.get('cha1')}:{prepro_para.get('cha2')}/'
    reader_array, timestamps = get_reader_array(dir_path)
    
    data = get_data_from_array(reader_array, prepro_para, t_start, timestamps, duration=prepro_para.get('duration'))[:, 0]
    
    data = np.float32(bandpass(data,
                            0.9 * prepro_para.get('freqmin'),
                            1.1 * prepro_para.get('freqmax'),
                            df=prepro_para.get('sps'),
                            corners=4,
                            zerophase=True))
    
    N = data.shape[0]
    g_std = 12
    gaussian_win = gaussian(100, g_std, sym=True)
    stft = ShortTimeFFT(gaussian_win, hop=50, fs=prepro_para.get('sps'), scale_to='psd')
    spec = stft.spectrogram(data)

    fig1, ax1 = plt.subplots(figsize=(6., 4.))
    t_lo, t_hi = stft.extent(N)[:2]
    ax1.set_title(rf"{t_start} at channel {prepro_para.get('cha1')}")
    ax1.set(xlabel=f"Time $t$ in seconds ({stft.p_num(N)} slices, " +
                rf"$\Delta t = {stft.delta_t:g}\,$s)",
            ylabel=f"Freq. $f$ in Hz ({stft.f_pts} bins, " +
                rf"$\Delta f = {stft.delta_f:g}\,$Hz)",
            xlim=(t_lo, t_hi))
    print(f'spec max: {spec.max()}; spec min: {spec.min()}')
    # spec = 10 * np.log10(np.fmax(spec, 1e-4))     # disabled for now, norm below is doing the same essentially
    im1 = ax1.imshow(spec, origin='lower', aspect='auto', norm=LogNorm(vmin=1e-4), 
                     extent=stft.extent(N), cmap='magma')
    plt.ylim(prepro_para.get('freqmin'), prepro_para.get('freqmax'))
    fig1.colorbar(im1, label='Power Spectral Density ' + r"$20\,\log_{10}|S_x(t, f)|$ in dB")
    plt.savefig(f'./results/figures/{out_dir}/psd.png')


if __name__ == '__main__':
    # dir_path = "../../temp_data_store/FirstData/"
    # dir_path = "../../../../gpfs/data/DAS_data/Data/"
    dir_path = "../../../../gpfs/scratch/gfs19eku/20240205/"
    task_t0 = datetime(year = 2024, month = 2, day = 5, 
                       hour = 12, minute = 1, second = 0, microsecond = 0)
    
    properties = tdms_io.get_dir_properties(dir_path)
    prepro_para = {
        'cha1': 4000,
        'cha2': 4001,
        'sps': properties.get('SamplingFrequency[Hz]'),
        'spatial_ratio': int(1 / properties.get('SpatialResolution[m]')),          # int(target_spatial_res/spatial_res)
        'duration': timedelta(hours=1),
        'freqmax': 49.9,
        'freqmin': 1,
    }
    task_t0 = datetime(year = 2023, month = 11, day = 9, 
                       hour = 13, minute = 41, second = 17)
    ts_spectrogram(dir_path, prepro_para, task_t0)

    # reader_array, timestamps = get_reader_array(dir_path)

    # channel_slices = [[1500, 1500], [3000, 3000], [5000, 5000], [7000, 7000]]
    # # psd_with_channel_slicing(reader_array, prepro_para, task_t0, timestamps, channel_slices)

    # animated_spectrogram(reader_array, prepro_para, task_t0, timestamps)

    # plot_gps_coords('results/linewalk_gps.csv')
