### Functions from: https://github.com/luan-th-nguyen/PyDispersion/blob/master/src/dispersion.py
import sys
sys.path.append("./src")
sys.path.append("./DASstore")

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
from obspy import Stream, Trace
from obspy.core.trace import Stats
import dascore
from datetime import datetime, timedelta
from TDMS_Read import TdmsReader


def get_closest_index(timestamps, time):
    """retrieves the index of the closest timestamp within timestamps to time

    Args:
        timestamps (ndarray): _description_
        time (timestamp): _description_

    Returns:
        _type_: _description_
    """    
    # array must be sorted
    idx = timestamps.searchsorted(time)
    idx = np.clip(idx, 1, len(timestamps)-1)
    idx -= time - timestamps[idx-1] < timestamps[idx] - time
    return idx


def get_tdms_array(dir_path):
    tdms_array = np.empty(int(len([filename for filename in os.listdir(dir_path) if filename.endswith(".tdms")])), dtype=TdmsReader)
    timestamps = np.empty(len(tdms_array), dtype=datetime)

    for count, file in enumerate([filename for filename in os.listdir(dir_path) if filename.endswith(".tdms")]):
        if file.endswith('.tdms'):
            tdms = TdmsReader(dir_path + file)
            tdms_array[count] = dir_path + file
            timestamps[count] = tdms.get_properties().get('GPSTimeStamp')
    timestamps.sort()
    print(f'{len(timestamps)} files available from {timestamps[0]} to {timestamps[-1]}')

    return [x for y, x in sorted(zip(np.array(timestamps), tdms_array))], timestamps


def get_time_subset(dir_path, start_time, tpf, delta, tolerance=300):
    tdms_array, timestamps = get_tdms_array(dir_path)
    
    # tolerence is the time in s that the closest timestamp can be away from the desired start_time
    start_idx = get_closest_index(timestamps, start_time)
    if abs((start_time - timestamps[start_idx]).total_seconds()) > tolerance:
        print(f"Error: first tdms is over {tolerance} seconds away from the given start time.")
        return
    
    end_time = timestamps[start_idx] + delta - timedelta(seconds=tpf)
    end_idx = get_closest_index(timestamps, end_time)
    if (end_time - timestamps[end_idx]).total_seconds() > tolerance:
        print(f"WARNING: end tdms is over {tolerance} seconds away from the calculated end time.")
    # print(f"Given t={start_time}, snippet selected from {timestamps[start_idx]} to {timestamps[end_idx]}!")
    
    if (end_idx - start_idx + 1) != (delta.seconds/tpf):
        print(f"WARNING: time subset not continuous; only {(end_idx - start_idx + 1)*tpf} seconds represented.")
    
    return tdms_array[start_idx:end_idx+1]


def load_data(file_paths):
    stream = Stream()
    for file_path in file_paths:
        spool = dascore.spool(file_path)
        for patch in spool:
            stream += patch.io.to_obspy()
    return stream


def load_data(dir_path):
    stream = Stream()    
    for file in os.listdir(dir_path):
        if file.endswith(".tdms"):
            file_path = os.path.join(dir_path, file)
            spool = dascore.spool(file_path)
            for patch in spool:
                stream += patch.io.to_obspy()
    return stream


def load_xcorr(file_path):
    stream = Stream()
    xdata = np.loadtxt(file_path, delimiter=',')
    stats = Stats()
    stats.delta = 1/100
    stats.npts = 1601
    for i in range(0, xdata.shape[1]):
        stream.append(Trace(xdata[:, i], stats))
    print(stream)
    return stream


def get_fft(traces, dt, nt):
    """ Get temporal Fourier transform for each of the traces
    """
    f = scipy.fftpack.fftfreq(nt,dt) #f = np.linspace(0.0, 1.0/(2.0*dt), nt//2)
    U = scipy.fftpack.fft(traces)
    if np.size(U.shape) > 1:
        return U[:,0:nt//2], f[0:nt//2]
    else:
        return U[0:nt//2], f[0:nt//2]


def get_dispersion(traces,dx,cmin,cmax,dc,fmax):
    """ calculate dispersion curves after Park et al. 1998
    INPUTS
    traces: SU traces
    dx: distance between stations (m)
    cmax: upper velocity limit (m/s)
    fmax: upper frequency limit (Hz)
    OUTPUTS
    f: 1d array frequency vector
    c: 1d array phase velocity vector
    img: 2d array (c x f) dispersion image
    fmax_idx: integer index corresponding to the given fmax
    U: 2d array (nr x npts//2) Fourier transform of traces
    t: 1d array time vector
    """
    nr = len(traces) 
    dt = traces[0].stats.delta
    print('dt: ', dt)
    nt = traces[0].stats.npts
    print('nt: ', nt)
    t = np.linspace(0.0, nt*dt, nt)
    traces.detrend()
    traces.taper(0.05,type='hann')
    U, f = get_fft(traces, dt, nt)
    #dc = 10.0 # phase velocity increment
    c = np.arange(cmin,cmax,dc) # set phase velocity range
    df = f[1] - f[0]
    fmax_idx = int(fmax//df)
    print('Frequency resolution up to %5.2f Hz: %i bins' % (fmax, fmax_idx))
    print('Phase velocity resolution up to %5.2f m/s: %i bins' % (cmax, len(c)))
    print(f'c: {c}')
    print(f'f: {f}')
    img = np.zeros((len(c),fmax_idx))
    x = np.linspace(0.0, (nr-1)*dx, nr)
    for fi in range(fmax_idx): # loop over frequency range
        for ci in range(len(c)): # loop over phase velocity range
            if ci==800 or fi==800:
                print(f'ci: {ci}, fi: {fi}')
            k = 2.0*np.pi*f[fi]/(c[ci])
            num_zeroes = np.count_nonzero(U[:,fi]==0)
            if num_zeroes != 0:
                print(f'num zeroes: {num_zeroes}')
            img[ci,fi] = np.abs(np.dot(dx * np.exp(1.0j*k*x), U[:,fi]/np.abs(U[:,fi])))

    return f,c,img,fmax_idx,U,t


# dir_path = "/home/harry/Documents/0. PhD/DiSTANS/temp_data_store/"
# dir_path = "../../../data/DAS_data/Data/"		# changed from "../../../../gpfs/data/DAS_data/Data", should work? 
dir_path = "../temp_data_store/"
start_time = '2024-01-19T15:19:07'

# prepro_para = {
#     'cha1': 2000,
#     'cha2': 7999,
#     'sps': 100, 
#     'spatial_ratio': int(1/0.25),      # int(target_spatial_res/spatial_res)
#     'cc_len': 60
# }


dx = 1
cmin = 50.0
cmax = 8000.0
dc = 10.0
fmax = 100.0

# stream = load_data(dir_path)
stream = load_xcorr('test_stack.txt')

# tdms_array = get_time_subset(dir_path, start_time, tpf=10, delta=timedelta(minutes=1))
f, c, img, fmax_idx, U, t = get_dispersion(stream, dx, cmin, cmax, dc, fmax)

im, ax = plt.subplots(figsize=(7.0,5.0))
ax.imshow(img[:,:],aspect='auto',origin='lower',extent=(f[0],f[fmax_idx],c[0],c[-1]),interpolation='bilinear')
im.savefig('./results/figures/test_dispersion.png')
