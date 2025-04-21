import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.signal import welch, ShortTimeFFT
from scipy.signal.windows import gaussian
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
from math import ceil, sin, cos, atan2, degrees, radians, log
import xdas as xd

from tdms_io import get_reader_array, get_data_from_array, get_dir_properties, load_xcorr


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


def ts_spectrogram(dir_path:str, prepro_para:dict, t_start:datetime):
    cha1, cha2, sps, freqmin, freqmax, n_minute = prepro_para.get('cha1'), prepro_para.get('cha2'), prepro_para.get('sps'), prepro_para.get('freqmin'), prepro_para.get('freqmax'), prepro_para.get('n_minute')
    
    # out_dir = f"./results/figures/{t_start}_{n_minute}mins_{cha1}:{cha2}/"        # changed for PSD experiments 17/02
    out_dir = f"./results/figures/PSD_Experiments/"
    
    # reader_array, timestamps = get_reader_array(dir_path)
    if type(dir_path == str): 
        reader_array, timestamps = get_reader_array(dir_path)

    elif type(dir_path == list):
        reader_array, timestamps = get_reader_array(dir_path[0])
        for path in dir_path[1:]:
            arr, stamps = get_reader_array(path)
            reader_array += arr; timestamps = np.concatenate((timestamps, stamps))
    else: 
        print(f'dir_path bad format: expected list/str, got {type(dir_path)}')

    mid_cha = int(0.5 * (cha1 + cha2))
    prepro_para.update({'cha1':mid_cha, 'cha2':mid_cha+1})
    
    data = get_data_from_array(reader_array, prepro_para, t_start, timestamps, duration=timedelta(minutes=n_minute))[:, 0]
    
    data = np.float32(bandpass(data,
                            0.9 * freqmin,
                            1.1 * freqmax,
                            df=sps,
                            corners=4,
                            zerophase=True))
    
    N = data.shape[0]
    g_std = 12
    gaussian_win = gaussian(100, g_std, sym=True)
    stft = ShortTimeFFT(gaussian_win, hop=50, fs=sps, scale_to='psd')
    spec = stft.spectrogram(data)

    fig1, ax1 = plt.subplots(figsize=(6., 4.))
    t_lo, t_hi = stft.extent(N)[:2]
    ax1.set_title(rf"{t_start} at channel {cha1}")
    ax1.set(xlabel=f"Time $t$ in seconds ({stft.p_num(N)} slices, " +
                rf"$\Delta t = {stft.delta_t:g}\,$s)",
            ylabel=f"Freq. $f$ in Hz ({stft.f_pts} bins, " +
                rf"$\Delta f = {stft.delta_f:g}\,$Hz)",
            xlim=(t_lo, t_hi))
    print(f'spec max: {spec.max()}; spec min: {spec.min()}')
    # spec = 10 * np.log10(np.fmax(spec, 1e-4))     # disabled for now, norm below is doing the same essentially
    ext = stft.extent(N)
    im1 = ax1.imshow(spec, origin='lower', aspect='auto', norm=LogNorm(vmin=1e-4), 
                     extent=ext, cmap='jet')
    if n_minute > 1440:
        _ = plt.xticks(np.linspace(0, ext[1], int(n_minute/1440)+1), pd.date_range(t_start, t_start+timedelta(minutes=n_minute), freq='D'), rotation=30)
        _ = plt.xticks(np.linspace(0, ext[1], int(n_minute/360)+1), minor=True)
    else: 
        _ = plt.xticks(np.linspace(0, ext[1], 4), pd.date_range(t_start, t_start+timedelta(minutes=n_minute), periods=4), rotation=30)
        _ = plt.xticks(np.linspace(0, ext[1], 16), minor=True)
    plt.ylim(freqmin, freqmax)
    fig1.colorbar(im1, label='PSD ' + r"$20\,\log_{10}|S_x(t, f)|$ in dB")
    plt.tight_layout()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # plt.savefig(f'{out_dir}/{t_start}__{t_start+timedelta(minutes=n_minute)}_f{freqmin}:{freqmax}_psd.png')     # also changed for experiments 17/02
    plt.savefig(f'{out_dir}/{t_start}__{t_start+timedelta(minutes=n_minute)}_f{freqmin}:{freqmax}_{mid_cha}_psd.png')


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
    

def plot_weather():
    weather_data = pd.read_csv('./results/checkpoints/weather.csv', sep=',', index_col=[0, 1], comment='#', na_values=['   --- ', '   ---'])
    weather_data.index = [np.datetime64(f'{date[0]}-{date[1] if date[1] > 9 else f"0{date[1]}"}', 'D') for date in weather_data.index]
    # print(weather_data)
    deployment_data = weather_data.loc[np.datetime64('2023-09-01'):]
    axs = deployment_data.plot.line(None, subplots=True, legend=False, grid=True, figsize=(12, 12))
    for ax, label in zip(axs, ['Max temp (degC)', 'Min temp (degC)', 'AF (days)', 'Rainfall (mm)', 'Sun (hours)']):
        ax.set_ylabel(label)
    plt.tight_layout()
    plt.savefig('./results/figures/weather_data.png')
    plt.show()


def plot_rain_storms():
    weather_data = pd.read_csv('./results/checkpoints/weather.csv', sep=',', index_col=[0, 1], comment='#', na_values=['   --- ', '   ---'], skipinitialspace=True)
    weather_data.index = [np.datetime64(f'{date[0]}-{date[1] if date[1] > 9 else f"0{date[1]}"}', 'D') for date in weather_data.index]
    rain_data = weather_data.loc[np.datetime64('2023-09-01'):, ['rain']]
    rain_data['rain'] = rain_data['rain'].astype(float)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(rain_data.index, rain_data['rain'])
    ax.set_ylabel('Rainfall (mm)')
    ax.grid(which='both') 
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,4,7,10)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=()))
    
    storms_data = pd.read_csv('./results/checkpoints/storms.csv', sep=',', index_col=0, comment='#')
    prev_storm_end = 0
    for storm in storms_data.index:
        dates = storms_data.loc[storm, ['start_date', 'end_date']]
        ax.axvspan(np.datetime64(dates['start_date']), np.datetime64(dates['end_date'])+1, label=storm, facecolor='r', alpha=0.5)
        if prev_storm_end and np.datetime64(dates['end_date']) - prev_storm_end < 10:
            ax.text(np.datetime64(dates['start_date']), 30, storm, rotation=90)
        else:
            ax.text(np.datetime64(dates['start_date']), 20, storm, rotation=90)
        prev_storm_end = np.datetime64(dates['end_date'])
    plt.tight_layout()
    # plt.savefig('./results/figures/rainfall_storms.png')
    plt.show()


def plot_era5_data(grib_path):
    from mpl_toolkits.basemap import Basemap, shiftgrid
    import pygrib
    plt.figure(figsize=(12,8))
    
    grbs = pygrib.open(grib_path)
    
    grb = grbs.select()[0]
    data = grb.values
    
    # need to shift data grid longitudes from (0..360) to (-180..180)
    lons = np.linspace(float(grb['longitudeOfFirstGridPointInDegrees']), float(grb['longitudeOfLastGridPointInDegrees']), int(grb['Ni'])+1)
    lats = np.linspace(float(grb['latitudeOfFirstGridPointInDegrees']), float(grb['latitudeOfLastGridPointInDegrees']), int(grb['Nj'])+1)
    data, lons = shiftgrid(0., data, lons, start=False)
    grid_lon, grid_lat = np.meshgrid(lons, lats) #regularly spaced 2D grid
    
    m = Basemap(projection='cyl', llcrnrlon=lons.min(), \
        urcrnrlon=lons.max(),llcrnrlat=lats.min(),urcrnrlat=lats.max(), \
        resolution='c')
    
    x, y = m(grid_lon, grid_lat)
    
    cs = m.pcolormesh(x,y,data,shading='flat',cmap=plt.cm.gist_stern_r)
    
    m.drawcoastlines()
    m.drawmapboundary()
    m.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
    m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
    
    plt.colorbar(cs,orientation='vertical', shrink=0.5)
    plt.title('CAMS AOD forecast') # Set the name of the variable to plot
    plt.savefig(grib_path+'.png') # Set the output file name


if __name__ == '__main__':
    # dir_path = "../../temp_data_store/FirstData/"
    # dir_path = "../../../../gpfs/data/DAS_data/Data/"
    dir_path = "../../../../gpfs/scratch/gfs19eku/20241008/"
    # dir_path = [dir_path, "../../../../gpfs/scratch/gfs19eku/2024_01_19/"]
    task_t0 = datetime(year = 2024, month = 10, day = 8, 
                       hour = 12, minute = 7, second = 46, microsecond = 0)
    
    properties = get_dir_properties(dir_path)
    prepro_para = {
        'cha1': 5900,
        'cha2': 5901,
        'sps': properties.get('SamplingFrequency[Hz]'),
        'spatial_ratio': int(1 / properties.get('SpatialResolution[m]')),          # int(target_spatial_res/spatial_res)
        'n_minute': 4320,
        'freqmin': 0.01,
        'freqmax': 49.9,
    }

    # reader_array, timestamps = get_reader_array(dir_path)
    # if type(dir_path) == list:
    #     reader_array, timestamps = get_reader_array(dir_path[0])
    #     for path in dir_path[1:]:
    #         arr, stamps = get_reader_array(path)
    #         reader_array += arr; timestamps = np.concatenate((timestamps, stamps))
    # else: 
    #     reader_array, timestamps = get_reader_array(dir_path)

    # reader_array, timestamps = get_reader_array(dir_path)

    # channel_slices = [[1500, 1500], [3000, 3000], [5000, 5000], [7000, 7000]]
    channel_slices = [[3000, 3000], [3150, 3150], [3500, 3500], [5900, 5900], [6200, 6200]]
    # psd_with_channel_slicing(reader_array, prepro_para, task_t0, timestamps, channel_slices)
    # ppsd_attempt(dir_path)

    for channels in channel_slices:
        print(f'Beginning {channels} run...')
        run_prepro_para = prepro_para.copy()
        run_prepro_para.update({'cha1':channels[0],
                                'cha2':channels[1]+1})
        ts_spectrogram(dir_path, run_prepro_para, task_t0)
        # Second run between 0.01-5 Hz
        run_prepro_para.update({'freqmax':5.0})
        ts_spectrogram(dir_path, run_prepro_para, task_t0)

    # animated_spectrogram(reader_array, prepro_para, task_t0, timestamps)

    # plot_gps_coords('results/linewalk_gps.csv')
    
    
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
    # plot_era5_data('era5_data.grib')
    
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
