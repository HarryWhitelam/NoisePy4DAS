##### Following Aquifer Monitoring..., Rodriguez Tribaldos, 2021
### Overview: 
# 1. Cross correlation (seperate)
    # This needs to be adapted, ccf between two channels
    # Stack ccfs for each day, then save as a 'row' in the output ccf
# 2. fk-filtering (OR bandpassing?)
# 3. Take coda window
# 4. Apply stretching technique
    # Stretch between -10% and 10%, applied with moving window of 0.25 s, shift of 0.02 s
    # Cross correlate previous and current (stretched) trace for that window
    # Remove outliers, and take median of values

import sys
sys.path.append("./src")
sys.path.append("./DASstore")
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from warnings import warn
from scipy.signal import savgol_filter, wiener, convolve2d
from scipy.interpolate import interp1d
import pywt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import Stream, Trace
from obspy.core.trace import Stats

from weather import plot_nwalsham_rain_csv


def weighted_corr(x, y, w):
    mask = (
        np.isfinite(x) &
        np.isfinite(y) &
        np.isfinite(w) &
        (w > 0)
    )

    if np.sum(mask) < 5:
        return np.nan

    x = x[mask]
    y = y[mask]
    w = w[mask]

    wsum = np.sum(w)

    x_mean = np.sum(w * x) / wsum
    y_mean = np.sum(w * y) / wsum

    cov_xy = np.sum(
        w * (x - x_mean) * (y - y_mean)
    )

    var_x = np.sum(
        w * (x - x_mean)**2
    )

    var_y = np.sum(
        w * (y - y_mean)**2
    )

    if var_x <= 0 or var_y <= 0:
        return np.nan

    return cov_xy / np.sqrt(var_x * var_y)


def stream_index_getter(day_idx, missed_days):
    if day_idx in missed_days:
        return None
    cnt = 0
    for k in range(1, day_idx + 1):
        if k not in missed_days:
            cnt += 1
    return cnt - 1


def stretching_dvv(ccf_ref, ccf, dt, tmin, tmax, t_win_size=0.25, t_win_step=0.02, eps_limit=0.05, side='causal', metric='median', norm=False):
    '''Stretching Technique following Sens-Schonfelder & Wegler
    For this, ccf_ref is the day before our 'current' ccf
    Stretching is applied to reference trace, therefore, dv/v = -max(eps)
    
    Side can be 'causal', 'acausal' or 'both' for average
    '''
    
    if norm:
        ccf_ref = ccf_ref / np.nanmax(ccf_ref)
        ccf = ccf / np.nanmax(ccf)
    
    mid = int(len(ccf_ref)/2)
    if side == 'causal':
        ccf_ref = ccf_ref[mid:]
        ccf = ccf[mid:]
    elif side == 'acausal':
        ccf_ref = ccf_ref[:mid+1][::-1]
        ccf = ccf[:mid+1][::-1]
    elif side == 'both':
        ccf_ref = (ccf_ref[mid:] + ccf_ref[:mid+1]) / 2
        ccf = (ccf[mid:] + ccf[:mid+1]) / 2
    
    ccf_ref = wiener(ccf_ref)
    ccf = wiener(ccf)
    
    n = len(ccf_ref)
    t = (np.arange(n) * dt)
    dvvs = []
    eps_best_corrs = []
    
    interp_ref = interp1d(t, ccf_ref, kind='cubic', bounds_error=False, fill_value=0.0)
    
    t_win_min = tmin; t_win_max = tmin + t_win_size
    while t_win_max <= tmax:
        mask = (t >= t_win_min) & (t <= t_win_max)
        t_win = t[mask]
        cur_win = ccf[mask]
        cur_win = cur_win - np.mean(cur_win)
        cur_win *= np.hanning(len(cur_win))

        eps_range = np.linspace(-eps_limit, eps_limit, 501)
        corr_coeff = []
        for eps in eps_range:
            t_stretch = (1 + eps) * t_win
            ref_stretched = interp_ref(t_stretch)

            num = np.sum(cur_win * ref_stretched)
            den = np.sqrt(np.sum(cur_win**2) * np.sum(ref_stretched**2))

            if den == 0:
                corr_coeff.append(0)
            else:
                corr_coeff.append(num / den)

        corr_coeff = np.array(corr_coeff)
        idx = np.argmax(corr_coeff)
        eps_best_corrs.append(corr_coeff[idx])
        dvvs.append(-eps_range[idx])
        
        t_win_min += t_win_step; t_win_max += t_win_step
    
    eps_best_corrs = np.array(eps_best_corrs); dvvs = np.array(dvvs)
    mask = (dvvs > np.percentile(dvvs, 10)) & (dvvs < np.percentile(dvvs, 90))
    if True in mask:
        eps_best_corrs = eps_best_corrs[mask]
        dvvs = dvvs[mask]
    
    match metric:
        case 'median':
            mask = eps_best_corrs >= 0.7
            if True in mask:
                dvvs = dvvs[mask]
                eps_best_corrs = eps_best_corrs[mask]
            idx = np.argsort(dvvs)[len(dvvs)//2]
        case 'max':
            idx = np.argmax(eps_best_corrs)
    eps_best_corr = eps_best_corrs[idx]
    # if eps_best_corr < 0.6: 
    #     print(f'\nSelected corr: {eps_best_corr}')
    #     for i in range(len(dvvs)):
    #         print(eps_best_corrs[i], dvvs[i])
    dvv = dvvs[idx]
    q25, q75 = np.percentile(dvvs, 25), np.percentile(dvvs, 75)

    return dvv, eps_best_corr, (q25, q75)


def smooth_cfs(cfs, scales, dt, nt=0.1, ns=3):
    N = cfs.shape[1]
    npad = int(2 ** np.ceil(np.log2(N)))

    omega_pos = np.arange(1, npad // 2 + 1) * (2 * np.pi / npad)
    if npad % 2 == 0:
        omega = np.concatenate(([0.0], omega_pos, -omega_pos[-2::-1]))
    else:
        omega = np.concatenate(([0.0], omega_pos, -omega_pos[::-1]))
    normscales = scales / dt

    for k in range(len(scales)):
        F = np.exp(-nt * (normscales[k] ** 2) * omega ** 2)
        smooth = np.fft.ifft(F * np.fft.fft(cfs[k, :], npad))
        cfs[k, :] = smooth[:N]

    H = np.ones((ns, 1)) / ns
    cfs = convolve2d(cfs, H, mode="same", boundary="symm")

    return cfs


def wavelet_method(ccf_ref, ccf, dt, tmin, tmax, fmin=1.0, fmax=15.0, side='causal'):
    mid = int(len(ccf_ref)/2)
    if side == 'causal':
        ccf_ref = ccf_ref[mid:]
        ccf = ccf[mid:]
    elif side == 'acausal':
        ccf_ref = ccf_ref[:mid+1][::-1]
        ccf = ccf[:mid+1][::-1]
    elif side == 'both':
        ccf_ref = (ccf_ref[mid:] + ccf_ref[:mid+1]) / 2
        ccf = (ccf[mid:] + ccf[:mid+1]) / 2
    
    # ccf_ref = wiener(ccf_ref)
    # ccf = wiener(ccf)
    
    # (i)   Choose an appropriate mother wavelet function.
    w = pywt.ContinuousWavelet('cmor1.5-1.0')
    # (ii)  Compute the continuous wavelet transforms (CWT) of the reference and the current waveforms (Fig. 1b).
    freqs_target = np.linspace(fmin, fmax, num=100)
    scales = pywt.central_frequency(w)/(freqs_target*dt)
    
    cwt_ref, freqs = pywt.cwt(ccf_ref, scales=scales, wavelet=w, sampling_period=dt)
    cwt_cur, _     = pywt.cwt(ccf, scales=scales, wavelet=w, sampling_period=dt)
    # (iii) Calculate the wavelet cross-spectrum WX Y from the two CWTs, and obtain its phase spectrum φX Y ( f, t) (Fig. 1c).
    xwt = cwt_ref * np.conj(cwt_cur)
    xwt_amp = np.abs(xwt)
    xwt_angle = np.angle(xwt)
    # (iv)  Divide φX Y ( f, t) by 2π f at each frequency and lapse-time to get δt( f, t) (Fig. 1d, assuming no cycle skipping).
    xwt_ft = xwt_angle / (2*np.pi*freqs[:,None])
    # (v)   Define a weighting function for δt( f, t).
    S_xy = smooth_cfs(xwt, scales, dt)
    S_x = smooth_cfs(np.abs(cwt_ref)**2, scales, dt)
    S_y = smooth_cfs(np.abs(cwt_cur)**2, scales, dt)
    
    xwt_coh = np.abs(S_xy)**2 / (S_x * S_y)
    xwt_coh = np.clip(xwt_coh.real, 0, 1)
    threshold = 0.2
    weights = np.log(xwt_amp) / np.max(np.log(xwt_amp))
    weights[xwt_coh < threshold] = 0
    
    # fig, axs = plt.subplots(1,1)
    # im0 = axs.imshow(xwt_ft)
    # plt.colorbar(im0)
    # # im1 = axs[1].plot(ccf_ref)
    # # # plt.colorbar(im1)
    # # im2 = axs[2].imshow(xwt_ft)
    # plt.show()
    
    # (vi)  Use δt( f, t) to derive velocity changes (beyond contents of the wavelet method).
    t = np.arange(len(ccf_ref)) * dt
    mask = (t >= tmin) & (t <= tmax)
    t = t[mask]
    
    # r2s = np.ndarray(shape=(100,))
    dvv_f = np.zeros(len(freqs))
    for i, f in enumerate(freqs):
        dt_vec = xwt_ft[i, :][mask]
        f_weights = weights[i, :][mask]
        
        # plt.hist(xwt_angle[i,:], bins=50)
        # plt.xlim(-np.pi, np.pi)
        # plt.show()
        
        valid = f_weights > 0
        if np.sum(valid) < 3:
            dvv_f[i] = np.nan
            continue
        
        slope = np.sum(f_weights * t * dt_vec) / np.sum(f_weights * t**2)
        
        # pred = slope * t
        # dt_mean = np.sum(f_weights * dt_vec) / np.sum(f_weights)
        # rss = np.sum(f_weights * (dt_vec - pred)**2)
        # tss = np.sum(f_weights * (dt_vec - dt_mean)**2)
        # r2 = 1 - (rss/tss)
        # r2s[i] = r2
        
        ### doing corrs instead of r2, can't be bothered to rename vars
        # corr = weighted_corr(t, dt_vec, f_weights)
        # r2s[i] = corr
        
        # if r2 < 0.5:
        #     # print(f, r2)
        #     # plt.scatter(t, dt_vec, s=2)
        #     # plt.plot(t, slope*t, c='r')
        #     # plt.show()
        #     bad_r2s += 1
        #     dvv_f[i] = np.nan
        #     continue

        dvv_f[i] = -(100 * slope)
    # print(f'bad r2s: {bad_r2s}/{len(freqs)}')
        
    return dvv_f, freqs


def plot_dvv(stream, task_t0, n_days, nlag_s=8, bandpass:list=None):
    ccf_stream = stream.copy()
    fig, ax = plt.subplots(1,1)
    if bandpass:
        ccf_stream.filter("bandpass", freqmin=bandpass[0], freqmax=bandpass[1])
    fig = ccf_stream.plot(type='section', recordstart=0, recordlength=nlag_s, fillcolors=('k', None), orientation='horizontal', fig=fig)
    ax.set_xticklabels([i-(nlag_s/2) for i in range(0,nlag_s+1)])
    ax.set_xlabel('Time lag (s)')
    ax.set_yticks([i*0.01 for i in range(0,n_days)])
    ax.set_yticklabels(np.arange(str(task_t0.date()), str((task_t0+timedelta(days=n_days)).date()), dtype='datetime64[D]'))
    ax.yaxis.minorticks_off()
    plt.show()

dir_path = '/data/localraid/dv_v_corrs'
task_t0 = datetime(year = 2024, month = 6, day = 1, 
                   hour = 0, minute = 0, second = 0, microsecond = 0)   # [2024,7], [2024,11], [2024,12], [2025,2], [2025,5]

stack_method = 'pws'
side = 'both'
metric = 'median'
months = 12

dates = np.arange(task_t0.date(), (task_t0+relativedelta(months=months)).date())

# get rainfall
# rain_df = plot_era5_csv(plot_daily=True, get_df=True)
rain_df = plot_nwalsham_rain_csv(get_df=True)
# rain_df['rainfall(mm)'].plot()
rain_df = rain_df[task_t0.date():(task_t0+relativedelta(months=months)-relativedelta(days=1)).date()]

# chas_list = [[800, 1000],[1000, 1200], [1200, 1400]]
chas = [1000, 1200]
f_bands = [[10, 15], [5, 10], [1, 5], [1, 15]]; n_fs = len(f_bands)
dvvs = [[] for _ in range(n_fs)]; eps_corrs = [[] for _ in range(n_fs)]
lower_errs = [[] for _ in range(n_fs)]; upper_errs = [[] for _ in range(n_fs)]
wavelet_dvvs = []

fig, axs = plt.subplots(n_fs, 1, sharex=True, sharey=True, figsize=(12,10))

for i in range(months):
    month_start = task_t0 + relativedelta(months=i)
    month_end = month_start + relativedelta(months=1)
    n_days = int((month_end - month_start).total_seconds() // (60*60*24))
    print(month_start)
    
    ccf_stream = Stream()
    missed_days = []
    cc = np.loadtxt(f'{dir_path}/{month_start.date()}_{month_end.date()}_100f0.01:25.0_{chas[0]}:{chas[1]}_10m_{stack_method}.txt', delimiter=',', dtype=np.float64)
    cc_ref = np.loadtxt(f'{dir_path}/ref_stacks/{month_start.date()}_{month_end.date()}_100f0.01:25.0_{chas[0]}:{chas[1]}_10m_{stack_method}.txt', delimiter=',', dtype=np.float64)
    for i in range(0,n_days):
        stats = Stats()
        stats.delta = 1/100; stats.npts = cc[:,i].shape[0]; stats.distance = i*10
        ccf_stream.append(Trace(cc[:,i], stats))
    ccf_ref = Trace(cc_ref, stats)
    # plot_dvv(ccf_stream, month_start, n_days, bandpass=[1.0, 15.0])

    ### BANDPASSING STREAM
    # ccf_stream_cp = ccf_stream.copy()
    # ccf_ref_cp = ccf_ref.copy()
    # for f_i, f_band in enumerate(f_bands):
    #     ccf_stream = ccf_stream_cp.copy()
    #     ccf_ref = ccf_ref_cp.copy()
    #     ccf_stream.filter("bandpass", freqmin=f_band[0], freqmax=f_band[1])
    #     ccf_ref.filter("bandpass", freqmin=f_band[0], freqmax=f_band[1])

    #     for i in range(0, n_days):
    #         if np.count_nonzero(ccf_stream[i].data) == 0:
    #             dvvs[f_i].append(0); eps_corrs[f_i].append(0); lower_errs[f_i].append(0); upper_errs[f_i].append(0)
    #         else:
    #             dvv, eps_best_corr, iqr = stretching_dvv(ccf_ref.data, ccf_stream[i].data, 0.01, 0.8, 1.8, eps_limit=0.05, side=side, metric=metric, norm=True)
    #             dvvs[f_i].append(dvv * 100)
    #             eps_corrs[f_i].append(eps_best_corr)
    #             lower_errs[f_i].append(np.abs(dvv - iqr[0]) * 100)
    #             upper_errs[f_i].append(np.abs(iqr[1] - dvv) * 100)
    
    # r2s_full = np.ndarray(shape=(n_days, 100))
    for i in range(0, n_days):
        dvv_f, freqs = wavelet_method(ccf_ref.data, ccf_stream[i].data, 0.01, 0.8, 2.8, side=side)
        wavelet_dvvs.append(dvv_f)
    #     r2s_full[i, :] = r2s
    # plt.plot(freqs, np.median(r2s_full, axis=0))
    # plt.show()

wavelet_dvvs = np.asarray(wavelet_dvvs)
print(wavelet_dvvs.shape)


# for f_i, f_band in enumerate(f_bands):
#     # dvvs = np.cumsum(dvvs)
#     f_eps_corrs = np.asarray(eps_corrs[f_i])
#     f_lower_errs = dvvs[f_i] - np.array(lower_errs[f_i])
#     f_upper_errs = dvvs[f_i] + np.array(upper_errs[f_i])

#     mask = f_eps_corrs >= 0.7
#     masked_dvvs = np.ma.array(dvvs[f_i])
#     masked_dvvs[~mask] = np.ma.masked

#     axs[f_i].plot(dates, masked_dvvs, label=f'{f_band[0]}-{f_band[1]} Hz') # c=cs[f_i], 
#     axs[f_i].plot(dates, savgol_filter(dvvs[f_i], 7, 2, mode='nearest'), linestyle='dashed')
#     axs[f_i].fill_between(dates, f_lower_errs, f_upper_errs, alpha=0.2, where=mask)# , color=cs[f_i], 
#     # axs[ax_idx].set_ylabel('Cumulative dv/v (%)')
#     axs[f_i].grid()

#     # label = f"({chr(ord('a') + ax_idx)})"
#     axs[f_i].text(0.01, 0.97, f'{f_band[0]}-{f_band[1]} Hz', transform=axs[f_i].transAxes,
#                   fontsize=10, va='top', ha='left')
#     axs[f_i].set_ylabel('dv/v (%)')
#     axs[f_i].margins(x=0)
#     axs[f_i].xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
#     axs[f_i].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[f_i].xaxis.get_major_locator()))
    

for i, [fmin, fmax] in enumerate(f_bands):

    idx = (freqs >= fmin) & (freqs < fmax)
    band_dvv = np.nanmean(wavelet_dvvs[:, idx], axis=1)
    # band_dvv = np.median(wavelet_dvvs[:, idx], axis=1)

    axs[i].plot(dates, band_dvv, label=f'{fmin}-{fmax} Hz')
    axs[i].plot(dates, savgol_filter(band_dvv, 7, 2, mode='nearest'), linestyle='dashed')       # week filter
    axs[i].plot(dates, savgol_filter(band_dvv, 28, 2), linestyle='dashdot')          # month (4 week) filter
    
    axs[i].text(0.01, 0.97, f'{fmin}-{fmax} Hz', transform=axs[i].transAxes,
                fontsize=10, va='top', ha='left')
    axs[i].set_ylabel('dv/v (%)')
    axs[i].margins(x=0)
    axs[i].xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    axs[i].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[i].xaxis.get_major_locator()))
    axs[i].grid()

# rainfall
# rain_ax = axs[-1].twinx()
# rain_ax.bar(dates, rain_df['value'], alpha=0.5)
# axs[-1].set_xlim(task_t0.date(), (task_t0+timedelta(days=n_days-1)).date())
# axs[-1].set_ylim(-8, 6)

# if ax_idx == int(len(chas_list)/2):
#     axs[ax_idx].set_ylabel('dv/v (%)')
    # rain_ax.set_ylabel('Rainfall (mm)')
    # if len(f_bands) > 1: axs[ax_idx].legend()
    # axs[ax_idx].legend()

fig.suptitle(f'wavelet dv/v [{stack_method}] [{side}]')
# fig.suptitle(f'dv/v [{stack_method}] [{side}] [{metric}]')
plt.tight_layout()
plt.show()
