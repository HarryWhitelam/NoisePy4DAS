import sys
sys.path.append("../src")
sys.path.append("../DASstore")

import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.parser import parse

import obspy
from obspy.io.segy.core import _read_segy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import dascore as dc

def read_das_dir(dir_path):
    tdms_spool = dc.spool(dir_path).update()
    
    contents = tdms_spool.get_contents()
    print(contents)
    
    patch = tdms_spool[0]
    patch.viz.waterfall(show=True, scale=(-50, 50))


def read_das_file(file_path):
    tdms = dc.spool(file_path)
    contents = tdms.get_contents()
    pd.set_option('display.max_columns', None)
    print(contents)
    pd.reset_option('display.max_columns')
    # tdms[0].viz.waterfall(show=True, scale=(-50, 50))


def obspy_read_segy(file_path):
    # st = obspy.read(file_path)
    st = _read_segy(file_path)
    st[0].plot()
    print(st)


def tdms_folder_converter(dir_path, out_dir_path):
    tdms_spool = dc.spool(dir_path).update()
    
    file_names = []
    for file in os.listdir(dir_path):
        if file.endswith(".tdms"):
            file_names.append(file)
    
    for i, patch in enumerate(tdms_spool):
        file_name = file_names[i][:-8]
        out_path = out_dir_path + file_name + "su"
        # patch.io.write(out_path, "dasdae")
        st = patch.io.to_obspy()
        for tr in st:
            tr.data = np.require(tr.data, dtype=np.float32)
        st.write(out_path, format="SU", data_encoding=5)

# tdms_folder_converter("/home/harry/Documents/0. PhD/DiSTANS/temp_data_store/", "/home/harry/Documents/0. PhD/DiSTANS/su_das/")
# obspy_read_segy("/home/harry/Documents/0. PhD/DiSTANS/segy_das/FirstData_UTC_20231109_134257.segy")
# obspy_read_segy("/home/harry/Documents/0. PhD/DiSTANS/su_das/FirstData_UTC_20231109_134257.su")
# read_das_dir("/home/harry/Documents/0. PhD/DiSTANS/segy_das/")
# read_das_file("/home/harry/Documents/0. PhD/DiSTANS/temp_data_store/FirstData_UTC_20231109_134257.573.tdms")



fig, axs = plt.subplots(2, 4, figsize=(15, 10))
cha1 = 2000
cha2 = 7999
spatial_ratio = 4
cha_spacing = 0.25
maxlag = 4
data = [np.random.rand(1000, 800)] * 8
vars = ['1', '2', '3', '4', '5', '6', '7', '8']

for ax, corr, var in zip(axs.ravel(), data, vars):
    plt.sca(ax)
    plt.imshow(corr, aspect = 'auto',
            vmax = 2e-2, vmin = -2e-2, origin = 'lower', interpolation=None)      # vmax, vmin original values of 2e-2, -2e-2 respectively

    _ =plt.yticks((np.linspace(cha1, cha2, 4) - cha1)/spatial_ratio,
                [int(i) for i in np.linspace(cha1, cha2, 4)], fontsize = 12)
    plt.ylabel("Channel number", fontsize = 12)
    _ = plt.xticks(np.arange(0, maxlag*200+1, 200), np.arange(-maxlag, maxlag+1, 2), fontsize=12)
    plt.xlabel("Time lag (sec)", fontsize = 12)
    # bar = plt.colorbar(pad = 0.1, format = lambda x, pos: '{:.1f}'.format(x*100))
    # bar.set_label('Cross-correlation Coefficient ($\\times10^{-2}$)', fontsize = 8)
    ax.label_outer()

    twiny = plt.gca().twinx()
    twiny.set_yticks(np.linspace(0, cha2 - cha1, 4),
                                [int(i* cha_spacing) for i in np.linspace(cha1, cha2, 4)])
    twiny.set_ylabel("Distance along cable (m)", fontsize = 12)
    twiny.label_outer()

    plt.tight_layout()
    plt.title(f"test {var}")

plt.tight_layout()
plt.savefig('quick_test.png')
