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
    stats.delta = 1/100
    stats.npts = xdata.shape[0]
    if chas is not None: 
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
    nt = traces[0].stats.npts
    # t = np.linspace(0.0, nt*dt, nt)
    # traces.detrend()
    # traces.taper(0.05,type='hann')
    # U, f = get_fft(traces, dt, nt)
    # f = f[f >= (0 or fmin)]; f = f[f <= fmax]
    t = np.arange(nt) * dt
    traces.detrend(type='linear')
    traces.taper(max_percentage=0.05, type='hann')
    traces_array = np.vstack([tr.data if hasattr(tr, 'data') else np.asarray(tr) for tr in traces])
    U_all, f_all = get_fft(traces_array, dt, nt)
    freq_mask = (f_all >= fmin) & (f_all <= fmax)
    f = f_all[freq_mask]
    U = U_all[:, freq_mask]
    if U.shape[1] == 0:
        raise RuntimeError(f"No frequency bins in [{fmin},{fmax}] Hz (f_all range: {f_all.min()}-{f_all.max()})")
    
    c = np.arange(cmin,cmax,dc) # set phase velocity range
    print('Frequency resolution up to %5.2f Hz: %i bins' % (fmax, len(f)))
    print('Phase velocity resolution up to %5.2f m/s: %i bins' % (cmax, len(c)))
    img = np.zeros((len(c),len(f)))
    x = np.linspace(0.0, (nr-1)*dx, nr)
    
    # epsilon = 1e-10
    # for fi in range(len(f)): # loop over frequency range
    #     for ci in range(len(c)): # loop over phase velocity range
    #         k = 2.0*np.pi*f[fi]/(c[ci])
    #         img[ci,fi] = 1/nr * np.abs(np.dot(np.exp(1.0j*k*x), U[:,fi]/np.abs(U[:,fi])))
    eps = 1e-12
    for fi in range(U.shape[1]):  # loop over selected frequency bins
        Uf = U[:, fi]
        # protect against zero amplitude
        amp = np.abs(Uf)
        amp[amp < eps] = eps
        phasor = Uf / amp
        for ci in range(len(c)):  # loop over phase velocity range
            k = 2.0 * np.pi * f[fi] / (c[ci])
            # phase steering vector
            steering = np.exp(1.0j * k * x)
            img[ci, fi] = (1.0 / nr) * np.abs(np.dot(steering, phasor))
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


def get_max_cs(img, c, f, fmin, fmax, f_freq=1):
    max_cs = []
    fs = [min(f, key=lambda x:abs(x-target_f)) for target_f in np.arange(fmin, fmax, f_freq)]
    f_idx = [np.where(f==f_val) for f_val in fs]
    
    # for f_val in img[:, 0:len(f)].T:
    for f_val in f_idx:
        max_cs.append(c[np.argmax(img[:, f_val].T)])
    return fs, max_cs


if __name__ == '__main__':
    # corr_path = '/data/localraid/saved_corrs/2025-02-16 00:00:00_4320mins_100f0.01:25.0__850:1650_10m.txt'
    corr_path = '/data/localraid/saved_corrs/2025-02-16 00:00:00_4320mins_100f0.01:25.0_800:1600_10m_pws.txt'
    # corr_path = '/data/localraid/saved_corrs/2024-11-04 00:00:00_1440mins_100f0.01:25.0__750:1750_10m.txt'
    
    stream = load_xcorr(corr_path)
    # stream.trim(UTCDateTime("19700101T00:00:08"))
    # for tr in stream: tr.data = np.flip(tr.data)
    
    if "SeaDAS_CCF" in corr_path:
        corr_name = 'SeaDAS_CCF'
        out_dir = './results/figures/'
        out_name = corr_name + '_dispersion'
        dx = 5.0
    else:    
        corr_name = corr_path.split('/')[-1][:-4]
        name_splits = corr_name.split('_')
        out_dir = f'./results/figures/{name_splits[0]}_{name_splits[1]}_{name_splits[3]}/'
        out_name = corr_name + '_dispersion'
        dx = float(corr_name.split('_')[4].strip('m'))      # 06/12 made modular on corr_path
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cmin = 150.0
    cmax = 2000.0
    dc = 10.0
    fmin = 0.01
    fmax = 25.0     # down from 100 for fmax testing
    
    f, c, img, U, t = get_dispersion(stream, dx, cmin, cmax, dc, fmin, fmax, normalise=False)
    
    fig, ax = plt.subplots(figsize=(7.0,5.0))
    im = ax.imshow(img[:,:],aspect='auto', origin='lower', extent=(f[0], f[-1], c[0], c[-1]), interpolation='bilinear')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase velocity (m/s)")
    bar = fig.colorbar(im, ax=ax, pad = 0.1) # if bad add in "format = lambda x, pos: '{:.1f}'.format(x*100)"
    fs, max_cs = get_max_cs(img, c, f, fmin, fmax)
    ax.scatter(fs, max_cs, facecolors='none', edgecolors='k')
    xs = [ci/10.0 for ci in c]
    ax.plot(xs, c, color='white', linestyle='dashed')
    ax.set_xlim(fmin, fmax)
    plt.tight_layout()
    plt.show()
    fig.savefig(f'{out_dir}{out_name}.eps')
    
    
    ### max amplitude plot + line of best fit
    # max_cs = get_max_cs(img, c, len(f))
    # ax.line(f, max_cs, facecolors='none', edgecolors='k')
    # print(f'max_cs: {len(max_cs)}; f: {f.shape}')
    # coefs = poly.polyfit(f, max_cs, 4)
    # ffit = poly.polyval(f, coefs)
    # plt.plot(f, ffit, color='red')
    # plt.tight_layout()
    # plt.show()
    # fig.savefig(f'{out_dir}{out_name}_annotated.eps')
    
    
    ### frequency normalisation
    for fi in range(len(f)):
        img[:, fi] /= np.max(img[:, fi])
    fig, ax = plt.subplots(figsize=(7.0,5.0))
    im = ax.imshow(img[:,:],aspect='auto', origin='lower', extent=(f[0], f[-1], c[0], c[-1]), interpolation='bilinear')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase velocity (m/s)")
    bar = fig.colorbar(im, ax=ax, pad = 0.1) # if bad add in "format = lambda x, pos: '{:.1f}'.format(x*100)"
    plt.tight_layout()
    plt.show()
    fig.savefig(f'{out_dir}{out_name}_f_norm.eps')

    # pcolormesh attempt
    # fig, ax = plt.subplots(figsize=(7.0, 5.0))
    # im = ax.pcolormesh(img**2, cmap='jet')
    # fig.colorbar(im, ax=ax)
    # ax.set_xlabel('Frequency (Hz)')
    # ax.set_ylabel('Phase velocity (m/s)')
    # plt.tight_layout()
    # fig.savefig(f'{out_dir}pcolormesh_attempt.eps')

    # print_freq_c_summaries(img, c, f, step=1)
