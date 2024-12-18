import sys
sys.path.append("./src")
sys.path.append("./DASstore")
from visualisation import image_comparison, spectral_comparison, numerical_comparison
from tdms_io import scale, slice_downsample, mean_downsample, max_min_strain_rate, downsample_tdms, read_das_file

from time import time
from TDMS_Read import TdmsReader
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize

def initial_downsample_comparison():
    t1 = time()

    # file_path =  "../../temp_data_store/Linewalk/LineWalkData_UTC_20240110_120340.208.tdms"
    file_path =  "../../temp_data_store/Snippets/FirstData_UTC_20231109_134947.573.tdms"
    
    tdms = TdmsReader(file_path)
    props = tdms.get_properties()

    sps = props.get('SamplingFrequency[Hz]')
    spatial_res = props.get('SpatialResolution[m]')
    target_sps = sps
    target_spatial_res = 2

    if (temporal_ratio := int(sps/target_sps)) != sps/target_sps:             # reversed as time-reciprocal
        print(f'Target sps not a factor of current sps, some data will be lost.')
    if (spatial_ratio := int(target_spatial_res/spatial_res)) != target_spatial_res/spatial_res:
        print(f'Target spatial res not a factor of current spatial res, some data will be lost.')

    first_channel = 2000
    # last_channel = tdms.fileinfo['n_channels']
    last_channel = 3999

    start_time = datetime(year=2023, month=11, day=9, hour=13, minute=49, second=37)
    target_time = datetime(year=2023, month=11, day=9, hour=13, minute=54, second=37)
    ms_diff = (target_time - start_time).total_seconds() * 1000

    first_time_sample = ms_diff - 5000
    second_time_sample = ms_diff + 5000
    bounds = 10000000

    # data = tdms.get_data(first_channel, last_channel, first_time_sample, second_time_sample)
    data = tdms.get_data(first_channel, last_channel)
    data = scale(data, props)
    print(data.shape)

    current_file_ext = 'microseism.csv'
    try:
        sliced_data = pd.read_csv(f'res/downsample_tests/sliced_data_{current_file_ext}', sep=',').to_numpy()
    except:
        print("Sliced downsampled data missing, generating now.")
        t_sliced = time()
        sliced_data = slice_downsample(data, temporal_ratio, spatial_ratio)
        pd.DataFrame(sliced_data).to_csv(f'res/downsample_tests/sliced_data_{current_file_ext}', index=False, header=False)
        print(f"Sliced data took {time() - t_sliced} seconds to generate and save.")
    try:
        mean_data = pd.read_csv(f'res/downsample_tests/mean_data_{current_file_ext}', sep=',').to_numpy()
    except: 
        print("Mean downsampled data missing, generating now.")
        t_mean = time()
        mean_data = mean_downsample(data, temporal_ratio, spatial_ratio)
        pd.DataFrame(mean_data).to_csv(f'res/downsample_tests/mean_data_{current_file_ext}', index=False, header=False)
        print(f"Mean data took {time() - t_mean} seconds to generate and save.")

    print(f'preplotting time: {time() - t1}')
    
    # fig1, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # img1 = ax1.imshow(data, aspect='auto', interpolation='none', extent=(first_channel, last_channel, ((second_time_sample - 1)/fs), (first_time_sample/fs)), vmin=-bounds, vmax=bounds, cmap='bwr')
    # ax1.set_title('Original')
    # img2 = ax2.imshow(sliced_data, aspect='auto', interpolation='none', extent=(first_channel, last_channel, ((second_time_sample - 1)/fs), (first_time_sample/fs)), vmin=-bounds, vmax=bounds, cmap='bwr')
    # ax2.set_title('Sliced')
    # img3 = ax3.imshow(mean_data, aspect='auto', interpolation='none', extent=(first_channel, last_channel, ((second_time_sample - 1)/fs), (first_time_sample/fs)),  vmin=-bounds, vmax=bounds, cmap='bwr')
    # ax3.set_title('Mean')
    # plt.sca(ax1)
    # plt.ylabel('Time (seconds)')
    # plt.sca(ax2)
    # plt.xlabel('Channel No.')
    # plt.suptitle((props.get('GPSTimeStamp')))
    # # fig1.colorbar(img1, label="Nano Strain per Second [nm/m/s]")
    # plt.tight_layout()
    
    # extent2 = ax2.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    # fig1.savefig('res/downsample_tests/sliced_data.png', bbox_inches=extent2)
    # extent3 = ax3.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    # fig1.savefig('res/downsample_tests/mean_data.png', bbox_inches=extent3)
    
    # plt.show()
    
    # comp_img1 = imread('res/downsample_tests/sliced_data.png')
    # comp_img2 = imread('res/downsample_tests/mean_data.png')
    
    resize_data = resize(data, output_shape=(data.shape[0] / temporal_ratio, data.shape[1] / spatial_ratio))
    
    data_dict = {
        'data': data,
        'sliced': sliced_data,
        'mean': mean_data,
        # 'mean - sliced': (mean_data - sliced_data),
        # 'sliced - mean': (sliced_data - mean_data), 
        'resize': resize_data,
    }
    for data in data_dict.values(): 
        # print(data.shape)
        max_strain, max_channel, min_strain, min_channel = max_min_strain_rate(data)
        print(f"Min Strain: {min_strain} at channel {min_channel}, Max Strain: {max_strain} at channel {max_channel}")

    image_comparison(data_dict.copy(), ['sliced', 'mean'], method='all', cmap='bwr')            # .copy() because dict is mutable, and we add records in image_comparison
    spectral_comparison(data_dict, fs=sps, find_nearest=True)
    numerical_comparison(data_dict)
    
    print(f'\nTotal time: {time() - t1}')


def downsample_comparison(tdms_path):
    tdms = TdmsReader(tdms_path)
    props = tdms.get_properties()
    tdms_data = tdms.get_data()
    tdms_data = scale(tdms_data, props)
    
    # print(f'Beginning load check, no downsampling.')
    # downsample_tdms(tdms_path, save_as='MSEED', out_dir="../../temp_data_store/downsamples/", target_sps=None, target_spatial_res=None)
    # mseed_data, stats = read_das_file("../../temp_data_store/downsamples/FirstData_UTC_20231109_134947.573.mseed")
    # print(np.array_equal(tdms_data, mseed_data))
    
    print(f'Beginning downsample check, to 1m spacings.')
    downsample_tdms(tdms_path, save_as='MSEED', out_dir="../../temp_data_store/downsamples/", target_sps=None, target_spatial_res=1)
    downsample_tdms(tdms_path, save_as='SEGY', out_dir="../../temp_data_store/downsamples/", target_sps=None, target_spatial_res=1)
    downsample_tdms(tdms_path, save_as='SU', out_dir="../../temp_data_store/downsamples/", target_sps=None, target_spatial_res=1)
    downsample_tdms(tdms_path, save_as='PICKLE', out_dir="../../temp_data_store/downsamples/", target_sps=None, target_spatial_res=1)
    
    mseed_data, mseed_stats = read_das_file("../../temp_data_store/downsamples/FirstData_UTC_20231109_134947.573.mseed")
    segy_data, segy_stats = read_das_file("../../temp_data_store/downsamples/FirstData_UTC_20231109_134947.573.segy")
    su_data, su_stats = read_das_file("../../temp_data_store/downsamples/FirstData_UTC_20231109_134947.573.su")
    pickle_data, pickle_stats = read_das_file("../../temp_data_store/downsamples/FirstData_UTC_20231109_134947.573.pickle")
    
    data_dict = {
        'tdms_data': tdms_data,
        'mseed_data': mseed_data,
        'segy_data': segy_data,
        'su_data': su_data,
        'pickle_data': pickle_data,
    }
    # image_comparison(data_dict.copy(), ['tdms_data', 'mseed_data'], method='all', cmap='bwr')
    # spectral_comparison(data_dict, fs=props.get('SamplingFrequency[Hz]'), find_nearest=True)
    # numerical_comparison(data_dict)


def format_size_comparisons(tdms_path):
    tdms = TdmsReader(tdms_path)
    props = tdms.get_properties()
    tdms_data = tdms.get_data()
    tdms_data = scale(tdms_data, props)
    
    downsample_tdms(tdms_path, save_as='MSEED', out_dir="../../temp_data_store/downsamples/", target_sps=None, target_spatial_res=1)
    downsample_tdms(tdms_path, save_as='SEGY', out_dir="../../temp_data_store/downsamples/", target_sps=None, target_spatial_res=1)
    downsample_tdms(tdms_path, save_as='SU', out_dir="../../temp_data_store/downsamples/", target_sps=None, target_spatial_res=1)
    downsample_tdms(tdms_path, save_as='PICKLE', out_dir="../../temp_data_store/downsamples/", target_sps=None, target_spatial_res=1)


# initial_downsample_comparison()
# format_size_comparisons("../../temp_data_store/Snippets/FirstData_UTC_20231109_134947.573.tdms")
downsample_comparison("../../temp_data_store/Snippets/FirstData_UTC_20231109_134947.573.tdms")
