import tdms_io
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.signal import spectrogram, welch
from scipy.fft import rfft, rfftfreq
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage.util import compare_images
import contextily as cx
from math import ceil


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


def psd_with_channel_slicing(tdms_array, prepro_para, task_t0, timestamps, channel_slices):
    fig, axs = plt.subplots(2, ceil(len(channel_slices)/2), figsize=(15, 10))
    
    for ax, channels in zip(axs.ravel(), channel_slices):
        plt.sca(ax)
        prepro_para['cha1'], prepro_para['cha2'] = channels[0], channels[1]
        
        tdata = tdms_io.get_data_from_array(tdms_array, prepro_para, task_t0, timestamps)
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


def animated_spectrogram(tdms_array, prepro_para, task_t0, timestamps):
    def update(channel_idx):
        channel_data = tdata[:, channel_idx]
        
        freqs, psd = welch(channel_data.T, fs=prepro_para.get('sps'))
        # plt.semilogy(freqs, psd, color='b')
        line.set_data(freqs, psd)
        title.set_text(f'Power Spectral Density (Channel {prepro_para.get("cha1") + (channel_idx * prepro_para.get("spatial_ratio"))})')
        return line, title
    
    n_channels = ceil((prepro_para.get('cha2') - prepro_para.get('cha1') + 1) / prepro_para.get('spatial_ratio'))
    tdata = tdms_io.get_data_from_array(tdms_array, prepro_para, task_t0, timestamps)
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


def image_comparison(data_dict, method, ncols=2, cmap='gray'):
    data_list = list(data_dict.values())
    if method in ('diff', 'all'):
        data_dict['diff'] = compare_images(data_list[0], data_list[1], method='diff')
    if method in ('blend', 'all'):
        data_dict['blend'] = compare_images(data_list[0], data_list[1], method='blend')
    
    nrows = len(data_dict) // ncols + (len(data_dict) % ncols > 0)
    
    fig = plt.figure(figsize=(15, 12))
    # plt.suptitle("TITLE HERE")
    for n, (key, val) in enumerate(data_dict.items()): 
        ax = plt.subplot(nrows, ncols, n + 1)
        ax.imshow(val, cmap=cmap, aspect='auto', interpolation='none')
        ax.title.set_text(key)
    
    fig.tight_layout()
    plt.show()


def spectral_comparison(data_dict, fs, ncols=2, subplots=False):
    nrows = len(data_dict) // ncols + (len(data_dict) % ncols > 0)
    
    fig = plt.figure(figsize=(15, 12))
    for n, (key, val) in enumerate(data_dict.items()):
        val = val.mean(axis=1)
        freqs, psd = welch(val.T, fs=fs)
        if subplots:
            ax = plt.subplot(nrows, ncols, n + 1)
            ax.semilogy(freqs, psd, label=f'test {key}')
            ax.title.set_text(key)
        else:
            plt.semilogy(freqs, psd, label=key)
    
    plt.legend()
    fig.tight_layout()
    plt.show()


# dir_path = "../../temp_data_store/"
# dir_path = "../../../../gpfs/data/DAS_data/Data/"
# properties = tdms_io.get_dir_properties(dir_path)
# prepro_para = {
#     'cha1': 2000,
#     'cha2': 2399,
#     'sps': properties.get('SamplingFrequency[Hz]'),
#     'spatial_ratio': int(1 / properties.get('SpatialResolution[m]')),          # int(target_spatial_res/spatial_res)
#     'duration': timedelta(seconds=120).total_seconds(),
# }
# task_t0 = datetime(year = 2023, month = 11, day = 9, 
#                    hour = 13, minute = 42, second = 57)
# tdms_array, timestamps = tdms_io.get_tdms_array(dir_path)

# channel_slices = [[1500, 1500], [3000, 3000], [5000, 5000], [7000, 7000]]
# # psd_with_channel_slicing(tdms_array, prepro_para, task_t0, timestamps, channel_slices)

# animated_spectrogram(tdms_array, prepro_para, task_t0, timestamps)

# plot_gps_coords('res/linewalk_gps.csv')
