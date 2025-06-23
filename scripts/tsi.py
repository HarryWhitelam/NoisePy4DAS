### Implementing TSI for CWI, DWI and 'enhanced' TSI (Curtis 2010, Song 2022)

### Song's ouline:
# 4hr ambient noise record
# bandpassed 1 - 30 Hz
# cut into 30 s windows
# normalise (t & f domains)
# calculate CCF
# stack NCF

# for NCF to be enhanced (S_i and S_j) select third (S_k)
    # if k between i and j, convolve
    # if k outside i and j, cross-correlate
# loop above for all third channels and stack positive and negative lags
# loop above for each i and j
# iterate to increase SNR for stable result

import numpy as np
from tdms_io import load_xcorr

# load xcorr as 
xcorr = load_xcorr('./results/saved_corrs/2024-02-05 12:01:00_4320mins_f0.01:49.9__3850:8050_1m.txt', as_stream=True)
print(xcorr)

# xcorr.filter('highpass', freq=5, zerophase=True)
idx = np.linspace(0, len(xcorr)-1, 6, dtype=int)
for i in idx:
    xcorr[i].plot()
