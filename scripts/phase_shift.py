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
# import dascore


# def load_data(file_paths:list):
#     stream = Stream()
#     for file_path in file_paths:
#         spool = dascore.spool(file_path)
#         for patch in spool:
#             stream += patch.io.to_obspy()
#     return stream


# def load_data(dir_path:str):
#     stream = Stream()    
#     for file in os.listdir(dir_path):
#         if file.endswith(".tdms"):
#             file_path = os.path.join(dir_path, file)
#             spool = dascore.spool(file_path)
#             for patch in spool:
#                 stream += patch.io.to_obspy()
#     return stream


def load_xcorr(file_path):
    stream = Stream()
    xdata = np.loadtxt(file_path, delimiter=',')
    stats = Stats()
    stats.delta = 1/100
    stats.npts = 1601
    for i in range(0, xdata.shape[1]):
        stream.append(Trace(xdata[:, i], stats))
    return stream


def get_fft(traces, dt, nt):
    """ Get temporal Fourier transform for each of the traces
    """
    f = scipy.fft.fftfreq(nt,dt) #f = np.linspace(0.0, 1.0/(2.0*dt), nt//2)
    U = scipy.fft.fft(traces)
    print(f'U: {U}')
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
    # print(f'c: {c}')
    # print(f'f: {f}')
    img = np.zeros((len(c),fmax_idx))
    x = np.linspace(0.0, (nr-1)*dx, nr)
    if fmax_idx > len(f):
        print(f'WARNING: maximum frequency too high. fmax_idx: {fmax_idx}; len(f): {len(f)}')
    epsilon = 1e-10
    for fi in range(fmax_idx): # loop over frequency range
        for ci in range(len(c)): # loop over phase velocity range
            k = 2.0*np.pi*f[fi]/(c[ci])
            num_zeroes = np.count_nonzero(U[:,fi]==0)
            if num_zeroes:
                print(f'num zeroes: {num_zeroes}')
            if np.any(np.isnan(U[:, fi])) or np.any(np.isinf(U[:, fi])):
                print(f"Warning: NaN or inf in U[:, {fi}]")
            if np.any(np.isnan(x)) or np.any(np.isinf(x)) or np.any(dx == 0):
                print("Warning: Invalid values in x or dx")
            if np.any(np.isnan(k * x)) or np.any(np.isinf(k * x)):
                print(f"Warning: Invalid values in k*x at frequency index {fi}")
            if np.any(np.abs(U[:, fi]) < epsilon):
                print(f"Warning: Small or zero values in U[:, {fi}]")
            img[ci,fi] = np.abs(np.dot(dx * np.exp(1.0j*k*x), U[:,fi]/np.abs(U[:,fi])))

    return f,c,img,fmax_idx,U,t


prepro_para = {
    'cha1': 2000,
    'cha2': 7999,
    'sps': 100, 
    'spatial_ratio': int(1/0.25),      # int(target_spatial_res/spatial_res)
    'cc_len': 60
}


dx = 1
cmin = 50.0
cmax = 8000.0
dc = 10.0
fmax = 50.0     # down from 100 for fmax testing


stream = load_xcorr('test_stack.txt')

f, c, img, fmax_idx, U, t = get_dispersion(stream, dx, cmin, cmax, dc, fmax)

im, ax = plt.subplots(figsize=(7.0,5.0))
ax.imshow(img[:,:],aspect='auto',origin='lower', extent=(f[0], f[fmax_idx-1], c[0], c[-1]), interpolation='bilinear')
im.savefig('./results/figures/test_dispersion.png')

im, ax = plt.subplots(figsize=(7.0,5.0))
ax.imshow(img[:,:],aspect='auto',origin='lower', interpolation='bilinear')
im.savefig('./results/figures/test_dispersion_no_extent.png')

img.tofile('./results/checkpoints/phase_shift_img.txt', sep=',')
f.tofile('./results/checkpoints/phase_shift_f.txt',   sep=',')
U.tofile('./results/checkpoints/phase_shift_U.txt',   sep=',')
c.tofile('./results/checkpoints/phase_shift_c.txt',   sep=',')
t.tofile('./results/checkpoints/phase_shift_t.txt',   sep=',')
