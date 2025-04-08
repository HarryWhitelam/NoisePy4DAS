suppress_figures = True

# attach local resources
import sys
sys.path.append("./src")
sys.path.append("./DASstore")

from datetime import datetime, time as dt_time
from correlation_funcs import *
from visualisation import ts_spectrogram


def channels_experiment():
    corrs = []
    channels_range = [[2000, 3499], [3500, 4999], [5000, 6499], [6500, 7999]]
    
    for chas in channels_range:
        print(f'Channels experiment subsection: {chas}')
        prepro_para = set_prepro_parameters(dir_path, task_t0, cha1=chas[0], cha2=chas[1])
        corr_full = correlation(dir_path, prepro_para)
        corrs.append(corr_full)

    plot_multiple_correlations(corrs, prepro_para, channels_range, 'channels', save_corr=False)


def frequency_experiment():
    corrs = []
    freq_range = [[0.001, 50.0], [0.001, 1.0], [1.0, 5.0], [5.0, 10.0], [10.0, 15.0], [15.0, 20.0], [20.0, 25.0], [25.0, 50.0]]
    # freq_range = [[0.001, 50.0], [0.001, 1.0], [1.0, 5.0], [5.0, 10.0]]     # smaller batch for testing
    
    for freqs in freq_range:
        print(f'Frequency experiment band: {freqs}')
        prepro_para = set_prepro_parameters(dir_path, task_t0, freqmin=freqs[0], freqmax=freqs[1], cha1=3850, cha2=7999, n_minute=360, target_spatial_res=1)
        corr_full = correlation(dir_path, prepro_para)
        corrs.append(corr_full)

    plot_multiple_correlations(corrs, prepro_para, freq_range, 'frequencies', save_corr=True)


def stack_length_experiment():
    corrs = []
    stack_length_range = [10, 60, 120, 360]
    
    for stack_length in stack_length_range:
        print(f'Stack length experiment: {stack_length}')
        prepro_para = set_prepro_parameters(dir_path, task_t0, n_minute=stack_length)
        corr_full = correlation(dir_path, prepro_para)
        corrs.append(corr_full)

    plot_multiple_correlations(corrs, prepro_para, stack_length_range, 'stack_length', save_corr=False)


def spatial_res_experiment():
    corrs = []
    spatial_res_range = [10, 60, 120, 360]
    
    for spatial_res in spatial_res_range:
        print(f'Spatial resolution experiment: {spatial_res}')
        prepro_para = set_prepro_parameters(dir_path, task_t0, target_spatial_res=spatial_res)
        corr_full = correlation(dir_path, prepro_para)
        corrs.append(corr_full)

    plot_multiple_correlations(corrs, prepro_para, spatial_res_range, 'spatial_res', save_corr=False)

# dir_path = "../../temp_data_store/FirstData/"

# dir_path = "../../../../gpfs/data/DAS_data/Data/"
# task_t0 = datetime(year = 2024, month = 1, day = 19,
#                    hour = 15, minute = 19, second = 7, microsecond = 0)

# dir_path = "../../../../gpfs/data/DAS_data/30mins/segys/"               # RUNNING WITH SEGYS
# task_t0 = datetime(year = 2023, month = 11, day = 9, 
#                    hour = 13, minute = 39, second = 47, microsecond = 0)

dir_path = "../../../../gpfs/scratch/gfs19eku/20240508/"
task_t0 = datetime(year = 2024, month = 5, day = 8, 
                   hour = 12, minute = 7, second = 49, microsecond = 0)


# frequency_experiment()


## SINGLE RUN: 
# cha pairings: 3850, 5750 [straight section]
#               3850, 8050 [cable from out of forest] 
#               3300, 3750 [forest]

# IF PASSING ALLOWED_TIMES, n_minute is the number of days you'd like to span, n_minute in terms fo the data will be calculated within correlation() 
prepro_para = set_prepro_parameters(dir_path, task_t0, target_spatial_res=1, cha1=3850, cha2=5750, n_minute=4320, freqmin=0.01, freqmax=50.0)
# prepro_para = set_prepro_parameters(dir_path, task_t0, target_spatial_res=1, cha1=7280, cha2=7680, n_minute=4320, freqmin=0.01, freqmax=50.0)
# prepro_para = set_prepro_parameters(dir_path, task_t0, target_spatial_res=0.25, cha1=962, cha2=1437, n_minute=30)      # adapted for segy files at 1 m spacings therefore cha_num / 4

spec_prepro_para = prepro_para.copy()           # copy bc python dicts are mutable so (effectively) passed by ref
# spec_prepro_para.update({'freqmax':5.0})
# ts_spectrogram(dir_path, spec_prepro_para, task_t0)

# times = {dt_time(20, 0, 0): dt_time(2, 0, 0), 
#          dt_time(12, 0, 0): dt_time(18, 0, 0)}

corr_full = correlation(dir_path, prepro_para)
plot_correlation(corr_full, prepro_para, save_corr=True)
