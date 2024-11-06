suppress_figures = True

# attach local resources
import sys
sys.path.append("./src")
sys.path.append("./DASstore")
# sys.path.append("./scripts")

from datetime import datetime
from correlation_funcs import *



# dir_path = "../../../../gpfs/data/DAS_data/Data/"
dir_path = "../../temp_data_store/FirstData/"
task_t0 = datetime(year = 2024, month = 1, day = 19,
                   hour = 15, minute = 19, second = 7, microsecond = 0)
n_minute = 1

# corrs = []
# freq_range = [[1.0, 49.9], [1.0, 10.0], [10.0, 25.0], [25.0, 49.9]]
# freq_range = [[0.01, 0.04], [0.04, 0.1], [0.1, 1.0], [1.0, 49.9]]
# n_minute_range = [10, 60, 120, 360]
# target_spatial_res_range = [0.25, 1, 5, 10]
# channels_range = [[2000, 3499], [3500, 4999], [5000, 6499], [6500, 7999]]
# for chas in channels_range:
#     print(f'Starting channels experiment: {chas}')
#     prepro_para = set_prepro_parameters(dir_path, cha1=chas[0], cha2=chas[1], target_spatial_res=5)
#     tdms_array, timestamps = get_tdms_array(dir_path)
#     corr_full = correlation(tdms_array, prepro_para, timestamps, task_t0)
#     corrs.append(corr_full)

# plot_multiple_correlations(corrs, prepro_para, channels_range, save_corr=False)



## SINGLE RUN: 
prepro_para = set_prepro_parameters(dir_path, task_t0, target_spatial_res=1)
tdms_array, timestamps = get_tdms_array(dir_path)

corr_full = correlation(tdms_array, prepro_para, timestamps, task_t0, save_corr=False)
plot_correlation(corr_full, prepro_para)
