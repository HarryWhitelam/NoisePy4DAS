
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from correlation_funcs import set_prepro_parameters
from tdms_io import get_filepath_array, get_data_from_array


def one_bit_normalize(x, axis=-1, eps=1e-12):
    return np.sign(np.clip(x, -1, 1) + 0.0)


def whiten_spectrum(x, fs, fmin, fmax, axis=-1, eps=1e-12):
    X = np.fft.rfft(x, axis=axis)
    freqs = np.fft.rfftfreq(x.shape[axis], d=1/fs)
    band = (freqs >= fmin) & (freqs <= fmax)

    amp = np.abs(X)
    win = 21
    kernel = np.ones(win) / win
    envelope = np.apply_along_axis(lambda a: np.convolve(a, kernel, mode='same'),
                                   axis=axis, arr=amp) + eps

    Xw = np.copy(X)
    Xw[..., band] = X[..., band] / envelope[..., band]

    return Xw, freqs


def segment_generator(data, fs, win_len_s=60.0, overlap=0.5):
    """
    Yield overlapping time windows (segments) with a cosine taper.
    """
    ntr, nt = data.shape
    w = int(round(win_len_s * fs))
    step = int(round(w * (1 - overlap)))
    if w > nt:
        print(w, nt)
        raise ValueError("Window length longer than the trace length.")
    # Cosine taper (5% each end)
    taper = np.ones(w)
    m = max(1, int(0.05 * w))
    ramp = 0.5*(1 - np.cos(np.linspace(0, np.pi, m)))
    taper[:m] = ramp
    taper[-m:] = ramp[::-1]

    for s in range(0, nt - w + 1, step):
        seg = data[:, s:s+w] * taper
        yield seg


def fk_beamform_linear(data, fs, dx, fmin=1.0, fmax=25.0, win_len_s=60.0, overlap=0.5, onebit=False, whiten=True, detrend=True, demean=True, return_velocity=True, vmin=100.0, vmax=2000.0):
    """
    Bartlett (delay-and-sum) f-k beamforming for a straight DAS array.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
        Time series for each channel (uniformly spaced).
    fs : float
        Sampling rate (Hz).
    dx : float
        Channel spacing (m).
    fmin, fmax : float
        Analysis band (Hz).
    win_len_s : float
        Window length for Welch-style averaging (s).
    overlap : float
        Fractional overlap between windows [0..1).
    onebit : bool
        Apply one-bit normalization per channel before segmentation.
    whiten : bool
        Apply spectral whitening (between fmin,fmax) per segment.
    detrend, demean : bool
        Simple trend/mean removal per segment.
    return_velocity : bool
        If True, also compute (f, v) image (masking k≈0).
    vmin, vmax : float
        Velocity axis for plotting (m/s).

    Returns
    -------
    fk_power : ndarray, shape (nf, nk)
    freqs : ndarray, shape (nf,)
    kvec  : ndarray, shape (nk,)
    fv_power : ndarray (optional), freqs, vvec
    """
    data = np.asarray(data)
    ntr, nt = data.shape
    x = np.arange(ntr) * dx

    # Pre-normalize if requested
    if demean:
        data -= np.mean(data, axis=1, keepdims=True)
    if detrend:
        # simple linear detrend via least squares per channel
        t = np.arange(nt)
        G = np.vstack([t, np.ones_like(t)]).T
        GTG_inv = np.linalg.inv(G.T @ G)
        for i in range(ntr):
            m = GTG_inv @ (G.T @ data[i])
            data[i] -= G @ m
    if onebit:
        data = one_bit_normalize(data, axis=1)

    # Prepare wavenumber grid (anti-alias safe)
    kmax = 0.2  # np.pi / dx
    nk = int(2 * ntr)  # dense grid; adjust as you like
    kvec = np.linspace(-kmax, kmax, nk)

    # First pass over one segment to size arrays
    gen = segment_generator(data, fs, win_len_s, overlap)
    first = next(gen)
    w = first.shape[1]
    freqs = np.fft.rfftfreq(w, d=1.0/fs)
    fband = (freqs >= fmin) & (freqs <= fmax)
    nf = np.count_nonzero(fband)

    fk_power = np.zeros((nf, nk), dtype=float)
    nsegs = 0

    def process_segment(seg):
        nonlocal fk_power, nsegs
        # Optional whitening (works best per-segment)
        if whiten:
            Xw, freqs = whiten_spectrum(seg, fs, fmin, fmax, axis=1)
            U = Xw[:, fband]   # already whitened in freq domai
        # FFT in time for each channel
        else:
            U = np.fft.rfft(seg, axis=1)  # shape: (ntr, w//2+1)
            U = U[:, fband]               # limit to band, shape (ntr, nf)

        # Steering matrix for this array (constant across segments)
        # A[n, k] = exp(-i * k * x_n)
        A = np.exp(-1j * np.outer(x, kvec))  # (ntr, nk)

        # For each frequency, compute beam power:
        # B(f,k) = | sum_n U_n(f) * exp(-i k x_n) |^2
        # Vectorized across k using matrix multiply
        # U_f: (ntr,), A: (ntr, nk) -> beams: (nk,)
        # Stack across all f
        for fi in range(nf):
            U_f = U[:, fi]  # (ntr,)
            beams = U_f.conj().T @ A   # (nk,)
            fk_power[fi, :] += np.abs(beams) ** 2

        nsegs += 1

    # Process the first and the rest of segments
    process_segment(first)
    for seg in segment_generator(data, fs, win_len_s, overlap):
        process_segment(seg)

    if nsegs > 0:
        fk_power /= nsegs

    freqs = freqs[fband]

    results = (fk_power, freqs, kvec)

    if return_velocity:
        # Map (f, k) -> (f, v) using v = 2*pi*f / k
        vv = np.linspace(vmin, vmax, 400)
        kv = (2*np.pi*freqs[:, None]) / vv[None, :]  # (nf, nv)
        # Interpolate fk_power along k for each frequency onto kv
        # Build k grid for interpolation
        from numpy import interp
        fv_power = np.zeros_like(kv, dtype=float)
        for fi in range(nf):
            fv_power[fi, :] = interp(kv[fi, :], kvec, fk_power[fi, :], left=0.0, right=0.0)
        results = (fk_power, freqs, kvec, fv_power, freqs, vv)

    return results


def plot_fk(fk_power, freqs, kvec, title="f-k power"):
    plt.figure(figsize=(8, 5))
    extent = [kvec[0], kvec[-1], freqs[0], freqs[-1]]
    plt.imshow(fk_power, aspect='auto', origin='lower', extent=extent, cmap='viridis')
    plt.xlabel("Wavenumber k (rad/m)")
    plt.ylabel("Frequency f (Hz)")
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label("Power")
    plt.tight_layout()


def plot_fv(fv_power, freqs, vvec, title="Dispersion Curve (f-v)"):
    plt.figure(figsize=(8, 5))
    extent = [freqs[0], freqs[-1], vvec[0], vvec[-1]]
    plt.imshow(fv_power.T, aspect='auto', origin='lower', extent=extent, cmap='viridis')
    plt.ylabel("Phase velocity v (m/s)")
    plt.xlabel("Frequency f (Hz)")
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label("Power")
    plt.tight_layout()


if __name__ == '__main__':
    dir_path = "/data/QNAP1_Data/Data/"
    # for h in [8,9,10,11,12,13,14,15,16,17]:
    task_t0 = datetime(year = 2025, month = 2, day = 12, 
                           hour = 15, minute = 0, second = 0, microsecond = 0)
    # task_t0 = datetime(year = 2025, month = 3, day = 27,
    #                 hour = 11, minute = 12, second = 39)
    n_minute = 10080
    duration = timedelta(minutes=n_minute)

    prepro_para = set_prepro_parameters(dir_path, task_t0, target_spatial_res=10, cha1=1300, cha2=1600, n_minute=n_minute, freqmin=0.1, freqmax=25.0)

    file_paths, timestamps = get_filepath_array(dir_path, task_t0, task_t0+duration)
    task_t0 = timestamps[0].replace(microsecond=0)
    
    data = get_data_from_array(file_paths, prepro_para, task_t0, duration).T
    print(data.shape)
    print(np.count_nonzero(np.isnan(data)))
    data = data[:, ~np.isnan(data).any(axis=0)]
    print(np.count_nonzero(np.isnan(data)))

    fk_power, f_fk, kvec, fv_power, f_fv, vvec = fk_beamform_linear(
        data, fs=prepro_para.get('samp_freq'), dx=prepro_para.get('target_spatial_res'), fmin=1.5, fmax=25.0,
        win_len_s=60.0, overlap=0.5, onebit=False, whiten=True)
    
    plot_fk(fk_power, f_fk, kvec, title=f"DAS ambient noise: f-k power")
    plot_fv(fv_power, f_fv, vvec, title="DAS ambient noise: apparent dispersion")
    plt.show()
