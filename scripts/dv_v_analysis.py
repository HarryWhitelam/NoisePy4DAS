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
from tqdm import tqdm
from warnings import warn
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import Stream, Trace
from obspy.core.trace import Stats

from weather import plot_nwalsham_rain_csv


def stream_index_getter(day_idx, missed_days):
    if day_idx in missed_days:
        return None
    cnt = 0
    for k in range(1, day_idx + 1):
        if k not in missed_days:
            cnt += 1
    return cnt - 1


def stretching_dvv(ccf_ref, ccf, dt, tmin, tmax, t_win_size=0.25, t_win_step=0.02, eps_limit=0.05, side='causal', metric='median'):
    '''Stretching Technique following Sens-Schonfelder & Wegler
    For this, ccf_ref is the day before our 'current' ccf
    Stretching is applied to reference trace, therefore, dv/v = -max(eps)
    
    Side can be 'causal', 'acausal' or 'both' for average
    '''    
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
        case 'median':  idx = np.argsort(dvvs)[len(dvvs)//2]
        case 'max':     idx = np.argmax(eps_best_corrs)
    eps_best_corr = eps_best_corrs[idx]
    # if eps_best_corr < 0.6: 
    #     print(f'\nSelected corr: {eps_best_corr}')
    #     for i in range(len(dvvs)):
    #         print(eps_best_corrs[i], dvvs[i])
    dvv = dvvs[idx]
    q25, q75 = np.percentile(dvvs, 25), np.percentile(dvvs, 75)

    return dvv, eps_best_corr, (q25, q75)


def plot_dvv(stream, task_t0, n_days, nlag_s=8, bandpass:list=None):
    ccf_stream = stream.copy()
    fig, ax = plt.subplots(1,1)
    if bandpass:
        ccf_stream.filter("bandpass", freqmin=bandpass[0], freqmax=bandpass[1])
    fig = ccf_stream.plot(type='section', recordstart=0, recordlength=nlag_s, fillcolors=('k', None), orientation='horizontal', fig=fig)
    ax.set_xticklabels([i-(nlag_s/2) for i in range(0,nlag_s+1)])
    ax.set_xlabel('Time lag (s)')
    ax.set_yticks([i*0.1 for i in range(0,n_days)])
    ax.set_yticklabels(np.arange(str(task_t0.date()), str((task_t0+timedelta(days=n_days)).date()), dtype='datetime64[D]'))
    ax.yaxis.minorticks_off()
    plt.show()

dir_path = '/data/localraid/saved_corrs'
task_t0 = datetime(year = 2024, month = 11, day = 1, 
                   hour = 0, minute = 0, second = 0, microsecond = 0)   # [2024,7], [2024,11], [2024,12], [2025,2], [2025,5]

stack_method = 'robust'
side = 'both'
metric = 'median'
n_days = 93
ch = 20         # 10 m interval, so this is 200 m

# get rainfall
# rain_df = plot_era5_csv(plot_daily=True, get_df=True)
rain_df = plot_nwalsham_rain_csv(get_df=True)
# rain_df['rainfall(mm)'].plot()
rain_df = rain_df[task_t0.date():(task_t0+timedelta(days=n_days-1)).date()]

# chas_list = [[1000, 1200], [1200, 1400]]
chas_list = [[800, 1000], [1000, 1200], [1200, 1400]]   # [500, 700], 
fig, axs = plt.subplots(len(chas_list),1, sharex=True, figsize=(12,10))

for ax_idx, chas in enumerate(chas_list):
    ccf_stream = Stream()
    missed_days = []
    t = task_t0
    for i in range(1,n_days+1):
        try:
            cc = np.loadtxt(f'{dir_path}/rma/{t}_1440mins_100f0.01:25.0_{chas[0]}:{chas[1]}_10m_{chas[0]}src_{stack_method}.txt', delimiter=',', dtype=np.float64)
            stats = Stats()
            stats.delta = 1/100; stats.npts = cc.shape[0]; stats.distance = (i-1)*100
            ccf_stream.append(Trace(cc[:, ch], stats))
        except:
            missed_days.append(i)
        t += timedelta(days=1)
    # plot_dvv(ccf_stream, task_t0, n_days, bandpass=[4.0, 15.0])

    ### BANDPASSING STREAM
    ccf_stream_cp = ccf_stream.copy()
    f_bands = [[1.0, 5.0], [5.0, 10.0], [10.0, 15.0]]
    for f_i, f_band in enumerate(f_bands):
        ccf_stream = ccf_stream_cp.copy()     # uncomment the copying bits for different f band looping
        ccf_stream.filter("bandpass", freqmin=f_band[0], freqmax=f_band[1])

        dvvs = [0]; eps_corrs = [1]; lower_errs = [0]; upper_errs = [0]
        for i in range(1, n_days):
            idx_prev = stream_index_getter(i-1, missed_days)
            idx_curr = stream_index_getter(i, missed_days)
            if idx_prev is None or idx_curr is None:
                dvvs.append(0); eps_corrs.append(0); lower_errs.append(0); upper_errs.append(0)
            else:
                ccf_0, ccf_1 = ccf_stream[idx_prev].data, ccf_stream[idx_curr].data
                dvv, eps_best_corr, iqr = stretching_dvv(ccf_0, ccf_1, 0.01, 0.8, 1.8, eps_limit=0.05, side=side, metric=metric)
                dvvs.append(dvv * 100)
                eps_corrs.append(eps_best_corr)
                lower_errs.append(np.abs(dvv - iqr[0]) * 100)
                upper_errs.append(np.abs(iqr[1] - dvv) * 100)

        dvvs = np.cumsum(dvvs)
        eps_corrs = np.asarray(eps_corrs)
        lower_errs = dvvs - np.array(lower_errs)
        upper_errs = dvvs + np.array(upper_errs)

        dates = np.arange(task_t0.date(), (task_t0+timedelta(days=n_days)).date())
        mask = eps_corrs >= 0.6
        masked_dvvs = np.ma.array(dvvs)
        masked_dvvs[~mask] = np.ma.masked
        
        axs[ax_idx].plot(dates, masked_dvvs, c=f'C{f_i+1}', label=f'{f_band[0]}-{f_band[1]} Hz')
        axs[ax_idx].fill_between(dates, lower_errs, upper_errs, alpha=0.4, color=f'C{f_i+1}', where=mask)
        # axs[ax_idx].set_ylabel('Cumulative dv/v (%)')
    axs[ax_idx].grid()
    
    label = f"({chr(ord('a') + ax_idx)})"
    axs[ax_idx].text(0.01, 0.97, label, transform=axs[ax_idx].transAxes,
                    fontsize=10, va='top', ha='left')
    plt.xticks(rotation=90)
    
    # rainfall
    rain_ax = axs[ax_idx].twinx()
    rain_ax.bar(dates, rain_df['value'], alpha=0.5)
    axs[ax_idx].set_xlim(task_t0.date(), (task_t0+timedelta(days=n_days-1)).date())
    # axs[ax_idx].set_ylim(-8, 6)
    
    if ax_idx == int(len(chas_list)/2):
        axs[ax_idx].set_ylabel('Cumulative dv/v (%)')
        rain_ax.set_ylabel('Rainfall (mm)')
        if len(f_bands) > 1: axs[ax_idx].legend()
        # axs[ax_idx].legend()

fig.suptitle(f'dv/v [{stack_method}] [{side}] [{metric}]')
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0.25)
plt.show()
