import json
import time
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import swprocess
from datetime import datetime, timedelta
import tdms_io


# Path (relative or full) to a folder containing the data files. Data files must be in either the SEG2 and/or SU data format.
dir_path = "/home/harry/Documents/0. PhD/DiSTANS/su_das/"
        
# fnames = []
# for file in os.listdir(dir_path):
#     # if file.lower().endswith(".sgy") | file.lower().endswith(".segy"):
#     if file.lower().endswith(".su"):
#         fnames.append(f'{dir_path}{file}')
#     else:
#         print(f"bad file :( not segy: {file}")

### importing data as ndarray wait this isn't what I want at all 
# dir_path = "../../temp_data_store/"
# properties = tdms_io.get_dir_properties(dir_path)
# prepro_para = {
#     'cha1': 6000,
#     'cha2': 7999,
#     'sps': properties.get('SamplingFrequency[Hz]'),
#     'spatial_ratio': int(1 / properties.get('SpatialResolution[m]')),          # int(target_spatial_res/spatial_res)
#     'duration': timedelta(seconds=360).total_seconds(),
# }
# task_t0 = datetime(year = 2023, month = 11, day = 9, 
#                    hour = 13, minute = 42, second = 57)
# reader_array, timestamps = tdms_io.get_reader_array(dir_path)
# tdata = tdms_io.get_data_from_array(reader_array, prepro_para, task_t0, timestamps)

### importing the stack data!
tdata = np.loadtxt('test_stack.txt', delimiter=',')
sps = 100


# Name for each fnames_set, if None, sets will be named according to the source position.
names = None

# Masw workflow {"time-domain", "frequency-domain", "single"}, time-domain is recommended
workflow = "time-domain"

# Trim record between the specified begin and end times (time in seconds). Trimming is recommended, however
# it must be done carefully to avoid accidentally trimming signal, particularly for far offsets.
trim, trim_begin, trim_end = True, 0, 0.5

# Mute portions of the time-domain record to isolate surface wave energy. No muting is recommended.
# Mute method {"interactive"} and window_kwargs (see documenation for details).
mute, method, window_kwargs = False, "interactive", {}

# Zero pad the time-domain record to achieve a desired frequency step. Padding with df=0.5 is recommended.
pad, df = True, 0.5

# Wavefield transform {"fk", "slantstack", "phaseshift", "fdbf"}, "fdbf" is recommended.
transform = "fdbf"

# Minimum and maximum frequencies of interest (frequency in Hertz).
fmin, fmax = 3, 100

# Selection of trial velocities (velocity in m/s) with minimum, maximum, number of steps, and space {"linear", "log"}.  
vmin, vmax, nvel, vspace = 100, 500, 400, "linear"

# Weighting for "fdbf" {"sqrt", "invamp", "none"} (ignored for all other wavefield transforms). "sqrt" is recommended. 
fdbf_weighting = "sqrt"

# Steering vector for "fdbf" {"cylindrical", "plane"} (ignored for all other wavefield transforms). "cylindrical" is recommended.
fdbf_steering = "cylindrical"

# Compute the records signal-to-noise ratio. 
snr = True

# Define noise and signal windows being and end times (time in seconds). Negative time refers to pre-trigger record.
noise_begin, noise_end =  -0.5, 0.
signal_begin, signal_end = 0., 0.5

# Zero pad the noise and signal records to achieve a specified frequency step. Padding with df=1 is recommended.
pad_snr, df_snr = True, 1

# Perform the selcted MASW workflow. No changes to this cell are required, however you may
# wish to change the variable `settings_fname` to a more specific name for later reference.
# This cell may take several seconds to run.
settings = swprocess.Masw.create_settings_dict(workflow=workflow,
                                               trim=trim, trim_begin=trim_begin, trim_end=trim_end,
                                               mute=mute, method=method, window_kwargs=window_kwargs,
                                               transform=transform, fmin=fmin, fmax=fmax, pad=pad, df=df,
                                               vmin=vmin, vmax=vmax, nvel=nvel, vspace=vspace,
                                               weighting=fdbf_weighting, steering=fdbf_steering,
                                               snr=snr, noise_begin=noise_begin, noise_end=noise_end,
                                               signal_begin=signal_begin, signal_end=signal_end,
                                               pad_snr = pad_snr, df_snr=df_snr)
start = time.perf_counter()

### Longshot to get this from an ndarray:
# make the source object
# source = swprocess.Source(-10.0, 0.0, 0.0)

# # make some noise data
# nstats = tdata.shape[1]
# npts = tdata.shape[0]

# x = np.linspace(0, 500, nstats)     # CURRENTLY LOCKED TO 500 m!!!
# y = np.zeros(nstats)
# z = np.zeros(nstats)

# # make the sensor object
# sensors = []
# for i in range(nstats): 
#     sensor = swprocess.Sensor1C(tdata[:,i], 1/sps, x[i], y[i], z[i])
#     sensors.append(sensor)

# data_array = swprocess.Array1D(sensors, source)
# data_array.to_file('test.su')

fnames = ['test.su']
data_array = swprocess.Array1D.from_files(fnames)
print(data_array)

fig, ax = data_array.plot()
print('finished plot')
print(type(fig), type(ax))
fig.savefig('test_fig.png')
fig.show()
# _ = data_array.waterfall()
### end of longshot
wavefieldtransform = swprocess.Masw.run(fnames=fnames, settings=settings)
end = time.perf_counter()
print(f"Elapsed Time (s): {round(end-start,2)}")
