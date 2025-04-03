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


from obspy.clients.fdsn import Client
from obspy import UTCDateTime

client = Client('http://eida.bgs.ac.uk')

# Download station list
# inventory = client.get_stations(network="GB")
# print(inventory)

# Save and plot data
# t = UTCDateTime("2025-01-26T04:00:00.0")      # UK
# t = UTCDateTime("2025-02-08T23:30:00.0")      # Cayman
# t = UTCDateTime("2025-01-12T19:45:00.0")      # Norway
# t = UTCDateTime("2025-03-28T06:30:00.0")      # Myanmar
# t = UTCDateTime("2025-03-30T12:15:00.0")      # Tonga
# st = client.get_waveforms("GB", "BEDF", "00", "HH?", t - 2, t + 14400,
#                           attach_response=True)
# st.plot()


import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        # "10m_u_component_of_wind",
        # "10m_v_component_of_wind",
        # "2m_temperature",
        # "mean_sea_level_pressure",
        # "mean_wave_direction",
        # "mean_wave_period",
        # "sea_surface_temperature",
        "total_precipitation"
    ],
    "year": ["2023", "2024", "2025"],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "grib",
    "download_format": "unarchived",
    "area": [55, 0, 52, 3]
}
target = 'era5_data.grib'

client = cdsapi.Client()
client.retrieve(dataset, request, target)
