### Functions from: https://github.com/luan-th-nguyen/PyDispersion/blob/master/src/dispersion.py

import matplotlib.pyplot as plt
import numpy as np
import scipy
from obspy import Stream
import dascore

def load_data(dir_path, start_time):
    spool = dascore.spool(dir_path).update()
    print('Loaded directory!')
    spool = spool.select(time=(start_time, None))
    print(f'Selected all samples after {start_time}')
    
    stream = Stream()
    for patch in spool:
        stream += patch.io.to_obspy()
    
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
    print('Frequency resolution up to %5.2f kHz: %i bins' % (fmax, fmax_idx))
    print('Phase velocity resolution up to %5.2f m/s: %i bins' % (cmax, len(c)))
    img = np.zeros((len(c),fmax_idx))
    x = np.linspace(0.0, (nr-1)*dx, nr)
    for fi in range(fmax_idx): # loop over frequency range
        for ci in range(len(c)): # loop over phase velocity range
            k = 2.0*np.pi*f[fi]/(c[ci])
            img[ci,fi] = np.abs(np.dot(dx * np.exp(1.0j*k*x), U[:,fi]/np.abs(U[:,fi])))

    return f,c,img,fmax_idx,U,t

# dir_path = "/home/harry/Documents/0. PhD/DiSTANS/temp_data_store/"
dir_path = "../../../../gpfs/data/DAS_data/30mins/"
start_time = '2023-11-09T13:39:46'

dx = 0.25
cmin = 50.0
cmax = 8000.0
dc = 10.
fmax = 100.0
stream = load_data(dir_path, start_time)
f,c,img,fmax_idx,U,t = get_dispersion(stream, dx, cmin, cmax, dc, fmax)

im, ax = plt.subplots(figsize=(7.0,5.0))
ax.imshow(img[:,:],aspect='auto',origin='lower',extent=(f[0],f[fmax_idx],c[0],c[-1]),interpolation='bilinear')
im.savefig('figures/test_dispersion.png')
