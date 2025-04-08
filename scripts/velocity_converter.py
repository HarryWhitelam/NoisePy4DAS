import matplotlib.pyplot as plt
import numpy as np
from daspy import read

dir_path = '../../temp_data_store/FirstData/'
sec = read(dir_path + 'FirstData_UTC_20231109_134117.573.tdms')
sec += read(dir_path + 'FirstData_UTC_20231109_134127.573.tdms')

print(sec.data.shape[1] / sec.fs)
t = np.arange(sec.data.shape[1]) / sec.fs

sec_fk = sec.fk_rescaling()
fk_data = sec_fk.channel_data(4000)

fig, ax = plt.subplot()
ax.plot(t, fk_data)
