### Functions from: https://github.com/luan-th-nguyen/PyDispersion/blob/master/src/dispersion.py
import sys
sys.path.append("./src")
sys.path.append("./DASstore")

import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import scipy
from obspy import Stream, Trace
from obspy.core.trace import Stats
# import dascore


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
    if np.size(U.shape) > 1:
        return U[:,0:nt//2], f[0:nt//2]
    else:
        return U[0:nt//2], f[0:nt//2]


def get_dispersion(traces, dx, cmin, cmax, dc, fmax, f_norm=False):
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

        if f_norm:
            img[:, fi] /= np.max(img[:, fi])
    
    return f,c,img,fmax_idx,U,t


def print_freq_c_summaries(img, c, fs=None):
    if not fs:
        fs = [1, 5, 10, 15, 20, 30, 40, 50]
    for f in fs:
        max_c = c[np.argmax(img[:,f])]
        # min_c = c[np.argmin(img[:,f])]
        print(f'c responses at {f} Hz: max {max_c} m/s')


def get_max_cs(img, c, fmax_idx):
    max_cs = []
    for f in img[:, 0:fmax_idx].T:
        max_cs.append(c[np.argmax(f)])
    return max_cs


if __name__ == '__main__':
    prepro_para = {
        'cha1': 2000,
        'cha2': 7999,
        'sps': 100, 
        'spatial_ratio': int(1/0.25),      # int(target_spatial_res/spatial_res)
        'cc_len': 60
    }
    
    # corr_path = './results/saved_corrs/2024-01-19 09:19:07_360mins_f1:49.9__3850:7999_1m.txt'
    # corr_path = './results/saved_corrs/2024-01-19 09:19:07_360mins_f1:49.9__3850:5750_1m.txt'
    # corr_path = './results/saved_corrs/2024-01-19 09:19:07_360mins_f1:49.9__3850:7999_0.25m.txt'
    # corr_path = './results/saved_corrs/2024-01-19 09:19:07_360mins_f1:49.9__3850:5750_0.25m.txt'
    corr_path = './results/saved_corrs/2024-02-02 12:01:00_4320mins_f1:49.9__3850:5750_1m.txt'
    stream = load_xcorr(corr_path)
    
    corr_name = corr_path.split('/')[3][:-4]
    name_splits = corr_name.rsplit('_', 4)
    out_dir = f'./results/figures/{name_splits[0]}_{name_splits[3]}/'
    out_name = corr_name + '_dispersion'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dx = float(corr_name.split('_')[-1].strip('m'))      # 06/12 made modular on corr_path
    cmin = 50.0
    cmax = 1500.0   # 27/11 dropped from 4000.0 to 1500.0
    dc = 5.0       # 27/11 changed from 10.0 to 5.0
    fmax = 20.0     # down from 100 for fmax testing
    
    f, c, img, fmax_idx, U, t = get_dispersion(stream, dx, cmin, cmax, dc, fmax)
    
    fig, ax = plt.subplots(figsize=(7.0,5.0))
    im = ax.imshow(img[:,:],aspect='auto', origin='lower', extent=(f[0], f[fmax_idx-1], c[0], c[-1]), interpolation='bilinear')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase velocity (m/s)")
    bar = fig.colorbar(im, ax=ax, pad = 0.1) # if bad add in "format = format = lambda x, pos: '{:.1f}'.format(x*100)"
    plt.tight_layout()
    fig.savefig(f'{out_dir}{out_name}.png')
    
    
    ### max amplitude plot + line of best fit
    # max_cs = get_max_cs(img, c, fmax_idx)
    # ax.plot(f, max_cs, color='black')     # removed max amps for just a line of best fit
    # print(f'max_cs: {len(max_cs)}; f: {f.shape}')
    # coefs = poly.polyfit(f, max_cs, 4)
    # ffit = poly.polyval(f, coefs)
    # plt.plot(f, ffit, color='red')
    # plt.tight_layout()
    # fig.savefig(f'{out_dir}{out_name}_annotated.png')
    
    
    ### frequency normalisation
    for fi in range(fmax_idx):
        img[:, fi] /= np.max(img[:, fi])
    fig, ax = plt.subplots(figsize=(7.0,5.0))
    im = ax.imshow(img[:,:],aspect='auto', origin='lower', extent=(f[0], f[fmax_idx-1], c[0], c[-1]), interpolation='bilinear')
    plt.tight_layout()
    fig.savefig(f'{out_dir}{out_name}_f_norm.png')
    
    print_freq_c_summaries(img, c)
