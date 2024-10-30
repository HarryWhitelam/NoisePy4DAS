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
read_das_file("/home/harry/Documents/0. PhD/DiSTANS/temp_data_store/FirstData_UTC_20231109_134257.573.tdms")
