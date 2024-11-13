suppress_figures = True

# attach local resources
import sys
sys.path.append("./src")
sys.path.append("./DASstore")
# sys.path.append("./scripts")

from datetime import datetime
from correlation_funcs import *


def channels_experiment():
    corrs = []
    channels_range = [[2000, 3499], [3500, 4999], [5000, 6499], [6500, 7999]]
    
    for chas in channels_range:
        print(f'Channels experiment subsection: {chas}')
        prepro_para = set_prepro_parameters(dir_path, task_t0, cha1=chas[0], cha2=chas[1])
        tdms_array, timestamps = get_tdms_array(dir_path)
        corr_full = correlation(tdms_array, prepro_para, timestamps)
        corrs.append(corr_full)

    plot_multiple_correlations(corrs, prepro_para, channels_range, 'channels', save_corr=False)


def frequency_experiment():
    corrs = []
    freq_range = [[0.001, 50.0], [0.001, 1.0], [1.0, 5.0], [5.0, 10.0], [10.0, 15.0], [15.0, 20.0], [20.0, 25.0], [25.0, 50.0]]
    
    for freqs in freq_range:
        print(f'Frequency experiment band: {freqs}')
        prepro_para = set_prepro_parameters(dir_path, task_t0, freqmin=freqs[0], freqmax=freqs[1], cha1=2000, cha2=7999)
        tdms_array, timestamps = get_tdms_array(dir_path)
        corr_full = correlation(tdms_array, prepro_para, timestamps)
        corrs.append(corr_full)

    plot_multiple_correlations(corrs, prepro_para, freq_range, 'frequencies', save_corr=False)


def stack_length_experiment():
    corrs = []
    stack_length_range = [10, 60, 120, 360]
    
    for stack_length in stack_length_range:
        print(f'Stack length experiment: {stack_length}')
        prepro_para = set_prepro_parameters(dir_path, task_t0, n_minute=stack_length)
        tdms_array, timestamps = get_tdms_array(dir_path)
        corr_full = correlation(tdms_array, prepro_para, timestamps)
        corrs.append(corr_full)

    plot_multiple_correlations(corrs, prepro_para, stack_length_range, 'stack_length', save_corr=False)


def spatial_res_experiment():
    corrs = []
    spatial_res_range = [10, 60, 120, 360]
    
    for spatial_res in spatial_res_range:
        print(f'Spatial resolution experiment: {spatial_res}')
        prepro_para = set_prepro_parameters(dir_path, task_t0, target_spatial_res=spatial_res)
        tdms_array, timestamps = get_tdms_array(dir_path)
        corr_full = correlation(tdms_array, prepro_para, timestamps)
        corrs.append(corr_full)

    plot_multiple_correlations(corrs, prepro_para, spatial_res_range, 'spatial_res', save_corr=False)

dir_path = "../../../../gpfs/data/DAS_data/Data/"
# dir_path = "../../temp_data_store/FirstData/"
task_t0 = datetime(year = 2024, month = 1, day = 19,
                   hour = 15, minute = 19, second = 7, microsecond = 0)
n_minute = 360

frequency_experiment()


## SINGLE RUN: 
# prepro_para = set_prepro_parameters(dir_path, task_t0, target_spatial_res=1)
# tdms_array, timestamps = get_tdms_array(dir_path)

# corr_full = correlation(tdms_array, prepro_para, timestamps, save_corr=False)
# plot_correlation(corr_full, prepro_para)
