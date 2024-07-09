import sys
sys.path.append("../src")
sys.path.append("../DASstore")

import os
import time
from datetime import datetime, timedelta
from dateutil.parser import parse

import h5py
import numpy as np
import DAS_module
import matplotlib.pyplot as plt
from tqdm import tqdm
from TDMS_Read import TdmsReader


def __main__():
    file_path = "../../../temp_data_store/FirstData_UTC_20231109_134117.573.tdms"
    tdms_file = TdmsReader(file_path)
    tdms_file._read_properties()

    n_channels = tdms_file.fileinfo['n_channels']
    cha1, cha2 = 0, n_channels-1

    alldata = tdms_file.get_data(cha1, cha2)
    properties = tdms_file.get_properties()

    # for prop in properties:
    #     print(prop)

    cha_spacing = properties.get('SpatialResolution[m]') * properties.get('Fibre Length Multiplier')
    start_dist, stop_dist = properties.get('Start Distance (m)'), properties.get('Stop Distance (m)')
    sps = properties.get('SamplingFrequency[Hz]')
    spatial_res = properties.get('SpatialResolution[m]')

    time_1 = datetime.strptime(file_path.split('/')[-1].split('UTC_')[-1].split('.')[0], '%Y%m%d_%H%M%S')
    time_delta = timedelta(seconds=tdms_file.channel_length / sps)
    time_2 = time_1 + time_delta
    print(time_1, time_2)


    print(type(alldata))
    print(f"data is of shape {alldata.shape}")

    plt.figure(figsize = (10, 6), dpi = 100)
    plt.imshow(alldata, aspect='auto', extent=[cha1, cha2, 0, time_delta.seconds],
            cmap='RdBu', vmax=1.5, vmin=-1.5, origin='lower')
    plt.xlabel("Channel number", fontsize=15)
    plt.ylabel("Time (s)", fontsize=15)
    plt.xlim([cha1, cha2])

    twinx = plt.gca().twiny()
    twinx.set_xticks(np.linspace(0, 2000, 9),
                    [int(i* cha_spacing) for i in np.linspace(0, 9000, 9)])
    twinx.set_xlabel("Distance along cable (m)", fontsize=15)
    plt.colorbar(pad = 0.15)