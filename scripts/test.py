import matplotlib.pyplot as plt
import numpy as np

def masw_fc_trans(data, dt, dist, f1, f2, c1, c2, nc=201):
    m, n = data.shape
    fs = 1 / dt; fn1 = int(f1*n/fs); fn2 = int(f2*n/fs)
    c = np.linspace(c1, c2, nc)
    f = np.arange(n) * fs / (n-1)
    df = f[1] - f[0]
    w = 2 * np.pi * f
    fft_d = np.zeros((m, n), dtype=np.complex)
    for i in range(m):
        fft_d[i] = np.fft.fft(data[i])
    fft_d = fft_d / abs(fft_d)
    fft_d[np.isnan(fft_d)] = 0
    fc = np.zeros((len(c), len(w[fn1: fn2+1])))
    for ci, cc in enumerate(c):
        for fi in range(fn1, fn2+1) :
            fc[ci, fi-fn1] = abs(sum(np.exp(1j*w[fi]/cc*dist)*fft_d[:, fi]))
    return f[fn1: fn2+1], c, fc/abs(fc).max()