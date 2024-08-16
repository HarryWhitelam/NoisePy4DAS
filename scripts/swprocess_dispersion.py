import json
import time
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

import swprocess

# Path (relative or full) to a folder containing the data files. Data files must be in either the SEG2 and/or SU data format.
path_to_folder = "/home/harry/Documents/0. PhD/DiSTANS/segy_das/"
        
fnames = []
for file in os.listdir(path_to_folder):
    if file.lower().endswith(".sgy") | file.lower().endswith(".segy"):
        fnames.append(f'{path_to_folder}{file}')
    else:
        print(f"bad file :( not segy: {file}")

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
wavefieldtransform = swprocess.Masw.run(fnames=fnames, settings=settings)
end = time.perf_counter()
print(f"Elapsed Time (s): {round(end-start,2)}")
