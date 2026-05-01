import daspy
import matplotlib.pyplot as plt
from daspy.basic_tools.visualization import plot
import numpy as np
from datetime import datetime, timedelta

from tdms_io import get_filepath_array, get_data_from_array
from correlation_funcs import set_prepro_parameters

# dir_path = "/data/QNAP1_Data/Data/"
dir_path = "/media/gfs19eku/Elements/DAS_Data/20241208/"
# task_t0 = datetime(year = 2024, month = 10, day = 22, 
#                         hour = 12, minute = 0, second = 0, microsecond = 0)
# 500hz_UTC_20241208_120736.240
task_t0 = datetime(year = 2024, month = 12, day = 8, 
                   hour = 12, minute = 8, second = 0, microsecond = 0)
# task_t0 = datetime(year = 2025, month = 3, day = 27,
#                 hour = 11, minute = 12, second = 39)
n_minute = 180
duration = timedelta(minutes=n_minute)

prepro_para = set_prepro_parameters(dir_path, task_t0, target_spatial_res=1, cha1=600, cha2=1600, n_minute=n_minute, freqmin=0.01, freqmax=50.0)

file_paths, timestamps = get_filepath_array(dir_path, task_t0, task_t0+duration)
task_t0 = timestamps[0].replace(microsecond=0)

data = get_data_from_array(file_paths, prepro_para, task_t0, duration, filter=False).T
print(data.shape)

sec = daspy.Section(data=data, dx=prepro_para['target_spatial_res'], fs=prepro_para['samp_freq'], start_channel=prepro_para['cha1'], start_distance=prepro_para['cha1'], start_time=0)
sec.spike_removal() # remove spikes
sec.channel_checking(use=True) # remove bad channels
# sec.bandpass(0.2, 20.0)

print(sec)

fig, ax = plt.subplots(1, 1)
sec.plot(ax=ax, obj='spectrum', tmode='time')
ax.set_yscale('log')
ax.set_ylim(0.01, 50)

# spec, f = sec.spectrum()
# spec = 10 * np.log10(abs(spec) ** 2)
# plot(spec, obj='spectrum', f=f)


# sec_out, sec_filt = sec.fk_filter(vmin=(400, 800), mode='decompose')
# fk, f, k = sec.fk_transform()
# fk_filt, f_filt, k_filt = sec_filt.fk_transform()
# fk_out, f_out, k_out = sec_out.fk_transform()

# fig, ax = plt.subplots(3, 2, figsize=(6,6), sharex='col', sharey='col', dpi=200)
# sec.plot(ax=ax[0,0], xlabel=False)
# sec_filt.plot(ax=ax[1,0])
# sec_out.plot(ax=ax[2,0], xlabel=False)
# vmin, vmax = 0, 2e5
# plot(fk, obj='fk', f=f, k=k, ax=ax[0,1], vmin=vmin, vmax=vmax, cmap='viridis')
# plot(fk_filt, obj='fk', f=f_filt, k=k_filt, ax=ax[1,1], vmin=vmin, vmax=vmax, cmap='viridis')
# plot(fk_out, obj='fk', f=f_out, k=k_out, ax=ax[2,1], vmin=vmin, vmax=vmax, cmap='viridis')
# for i in range(0,3):
#     ax[i,1].set_ylim(0, 10.0)
#     ax[i,1].set_xlim(-0.03, 0.03)
# plt.tight_layout()


plt.show()
