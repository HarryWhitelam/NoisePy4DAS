import os
import matplotlib.pyplot as plt
import numpy as np
from daspy import read
from pathlib import Path
from scipy.integrate import cumulative_trapezoid
from tdms_io import scale


n_files = 6
# dir_path = '../../temp_data_store/FirstData/'
dir_path = '../../../../gpfs/scratch/gfs19eku/20240508/'
file_list = [filename for filename in os.listdir(dir_path) if filename.endswith(('.tdms', '.segy'))]

sec = read(dir_path + file_list[0])
for i in range(1, 6):
    sec += read(dir_path + file_list[i])


data = sec.data
print(data.shape)
data = scale(data, {'SamplingFrequency[Hz]': sec.fs, 'GaugeLength': sec.gauge_length})
strain_data = cumulative_trapezoid(data, x=None, dx=1/sec.fs, axis=-1)

sec.data = strain_data
print(strain_data.shape)

t = np.arange(sec.data.shape[1]) / sec.fs

sec_fk = sec.copy().fk_rescaling(fmax=(5,6))
sec_cv = sec.copy().curvelet_conversion()

st_data = sec.channel_data(4000)
fk_data = sec_fk.channel_data(4000)
cv_data = sec_cv.channel_data(4000)

fig, axs = plt.subplots(3, 1, figsize=(7.0, 5.0))
axs[0].plot(t, st_data, c='#999999')
axs[1].plot(t, fk_data, c='#ED553B')
axs[2].plot(t, cv_data, c='#20639B')
plt.savefig(f'plot_velocity_conversion.png')
