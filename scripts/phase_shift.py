### Functions from: https://github.com/luan-th-nguyen/PyDispersion/blob/master/src/dispersion.py
import sys
sys.path.append("./src")
sys.path.append("./DASstore")

import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
from math import floor
import scipy
from obspy import Stream, Trace, UTCDateTime
from obspy.core.trace import Stats
# import dascore


def load_xcorr(file_path, normalise=False, chas=None):
    stream = Stream()
    xdata = np.loadtxt(file_path, delimiter=',')
    xdata = xdata[:, ~np.all(np.isnan(xdata), axis=0)]      # 06/01/25 added for Ni's data
    if normalise:
        xdata = xdata/np.sqrt(np.sum(xdata**2))
    stats = Stats()
    stats.delta = 1/50
    stats.npts = 801
    if chas: 
        for cha in chas:
            stream.append(Trace(xdata[:, cha], stats))
    else:
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


def get_dispersion(traces, dx, cmin, cmax, dc, fmin, fmax, f_norm=False, normalise=False):
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
    f = f[f >= (0 or fmin)]; f = f[f <= fmax]
    #dc = 10.0      # phase velocity increment
    c = np.arange(cmin,cmax,dc) # set phase velocity range
    df = f[1] - f[0]
    print(f'df: {df}')
    # fmax_idx = int((fmax-fmin)//df)         # This is stupid why have you called it fmax_idx if it's not the index it's the element count why have you done this fix this TODO: fix this please
    print('Frequency resolution up to %5.2f Hz: %i bins' % (fmax, len(f)))
    print('Phase velocity resolution up to %5.2f m/s: %i bins' % (cmax, len(c)))
    # print(f'c: {c}')
    # print(f'f: {f}')
    img = np.zeros((len(c),len(f)))
    x = np.linspace(0.0, (nr-1)*dx, nr)
    # if fmax_idx > len(f):
    #     print(f'WARNING: maximum frequency too high. fmax_idx: {fmax_idx}; len(f): {len(f)}. Setting fmax_idx to {len(f)}')
    #     fmax_idx = len(f)
    epsilon = 1e-10
    for fi in range(len(f)): # loop over frequency range
        for ci in range(len(c)): # loop over phase velocity range
            k = 2.0*np.pi*f[fi]/(c[ci])
            if np.any(np.isnan(U[:, fi])) or np.any(np.isinf(U[:, fi])):
                print(f"Warning: NaN or inf in U[:, {fi}]")
            if np.any(np.isnan(x)) or np.any(np.isinf(x)) or np.any(dx == 0):
                print("Warning: Invalid values in x or dx")
            if np.any(np.isnan(k * x)) or np.any(np.isinf(k * x)):
                print(f"Warning: Invalid values in k*x at frequency index {fi}")
            # if np.any(np.abs(U[:, fi]) < epsilon):
            #     print(f"Warning: Small or zero values in U[:, {fi}]")
            img[ci,fi] = 1/nr * np.abs(np.dot(np.exp(1.0j*k*x), U[:,fi]/np.abs(U[:,fi])))

        if f_norm:
            img[:, fi] /= np.max(img[:, fi])
        if normalise:
            img = img/np.sqrt(np.sum(img**2))
   
    
    return f,c,img,U,t


def print_freq_c_summaries(img, c, fs, step=5):
    round_fs = np.arange(0, floor(fs[-1])+1, step)
    # for f_idx in np.arange(0, len(fs), step):
    for f_idx in [np.argmin([abs(f-round_f) for f in fs]) for round_f in round_fs]:
        max_c = c[np.argmax(img[:,f_idx])]
        # min_c = c[np.argmin(img[:,f_idx])]
        print(f'c responses at {fs[f_idx]} Hz: max {max_c} m/s')


def get_max_cs(img, c, fmax_idx):
    max_cs = []
    for f in img[:, 0:fmax_idx].T:
        max_cs.append(c[np.argmax(f)])
    return max_cs


if __name__ == '__main__':    
    # corr_path = './results/saved_corrs/2024-02-05 12:01:00_4320mins_f0.01:49.9__3850:5750_1m.txt'
    # corr_path = './results/saved_corrs/2024-02-05 12:01:00_4320mins_f0.01:49.9__3300:3750_1m.txt'
    # corr_path = './results/saved_corrs/2024-02-05 12:01:00_4320mins_f0.01:49.9__3850:8050_1m.txt'
    # corr_path = './results/saved_corrs/2024-02-05 12:01:00_4320mins_f0.01:49.9__2000:3999_1m.txt'
    corr_path = './results/saved_corrs/2024-02-05 12:01:00_4320mins_100f0.01:50.0__4168:4568_1m.txt'
    # corr_path = './results/saved_corrs/2024-02-05 12:01:00_1440mins_100f0.01:50.0__3850:5750_1m.txt'
    # corr_path = './results/saved_corrs/2024-02-05 12:01:00_4320mins_20:00:00:02:00:00_12:00:00:18:00:00_100f0.01:50.0__3850:5750_1m.txt'
    # corr_path = './results/saved_corrs/SeaDAS_CCF.txt'
    
    stream = load_xcorr(corr_path)
    # stream.trim(UTCDateTime("19700101T00:00:08"))
    # for tr in stream: tr.data = np.flip(tr.data)
    
    if "SeaDAS_CCF" in corr_path:
        corr_name = 'SeaDAS_CCF'
        out_dir = './results/figures/'
        out_name = corr_name + '_dispersion'
        dx = 5.0
    else:    
        corr_name = corr_path.split('/')[3][:-4]
        name_splits = corr_name.rsplit('_', 4)
        out_dir = f'./results/figures/{name_splits[0]}_{name_splits[3]}/'
        out_name = corr_name + '_dispersion'
        dx = float(corr_name.split('_')[-1].strip('m'))      # 06/12 made modular on corr_path
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cmin = 50.0
    cmax = 1500.0   # 27/11 dropped from 4000.0 to 1500.0
    dc = 10.0       # 27/11 changed from 10.0 to 5.0
    fmin = 0.01
    fmax = 25.0     # down from 100 for fmax testing
    
    f, c, img, U, t = get_dispersion(stream, dx, cmin, cmax, dc, fmin, fmax, normalise=False)
    
    fig, ax = plt.subplots(figsize=(7.0,5.0))
    im = ax.imshow(img[:,:],aspect='auto', origin='lower', extent=(f[0], f[-1], c[0], c[-1]), interpolation='bilinear')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase velocity (m/s)")
    bar = fig.colorbar(im, ax=ax, pad = 0.1) # if bad add in "format = lambda x, pos: '{:.1f}'.format(x*100)"
    plt.tight_layout()
    plt.show()
    # fig.savefig(f'{out_dir}{out_name}.png')
    
    
    ### max amplitude plot + line of best fit
    # max_cs = get_max_cs(img, c, len(f))
    # ax.plot(f, max_cs, color='black')
    # print(f'max_cs: {len(max_cs)}; f: {f.shape}')
    # coefs = poly.polyfit(f, max_cs, 4)
    # ffit = poly.polyval(f, coefs)
    # plt.plot(f, ffit, color='red')
    # plt.tight_layout()
    # plt.show()
    # fig.savefig(f'{out_dir}{out_name}_annotated.png')
    
    
    ### frequency normalisation
    for fi in range(len(f)):
        img[:, fi] /= np.max(img[:, fi])
    fig, ax = plt.subplots(figsize=(7.0,5.0))
    im = ax.imshow(img[:,:],aspect='auto', origin='lower', extent=(f[0], f[-1], c[0], c[-1]), interpolation='bilinear')
    plt.tight_layout()
    plt.show()
    # fig.savefig(f'{out_dir}{out_name}_f_norm.png')

    # pcolormesh attempt
    # fig, ax = plt.subplots(figsize=(7.0, 5.0))
    # im = ax.pcolormesh(img**2, cmap='jet')
    # fig.colorbar(im, ax=ax)
    # ax.set_xlabel('Frequency (Hz)')
    # ax.set_ylabel('Phase velocity (m/s)')
    # plt.tight_layout()
    # fig.savefig(f'{out_dir}pcolormesh_attempt.png')

    print_freq_c_summaries(img, c, f, step=1)
