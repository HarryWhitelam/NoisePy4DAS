import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from obspy import UTCDateTime
from math import pi, ceil

### plan:
# getting average value in certain frequency ranges

### read spec file, ensure format: 
dt = 0.5
df = 1.0
target_ch = 3500

dir_path = '/data/localraid/saved_specs/'
# if target_ch: 
#     ch_opts = [f'_{int(target_ch / 4)}_', f'_{target_ch}_']
#     files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and any(ch_opt in f for ch_opt in ch_opts)]
# else: 
#     files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
# files = sorted(files)

# fs = np.arange(0, 51, df)
# dates = [UTCDateTime(file.split('__')[0]) for file in files]
# arr = np.empty((len(files), len(fs)))
# for i, file in enumerate(files):
#     spec = np.loadtxt(dir_path + file, delimiter=',')
#     for j, f in enumerate(fs):
#         max = spec[j, :].max()
#         arr[i,j] = max

# df = pd.DataFrame(data=arr, index=dates)
# df.to_csv('./results/checkpoints/max_psd_series.csv')

df = pd.read_csv('./results/checkpoints/max_psd_series.csv', index_col=0, parse_dates=True)


# waverider_df.plot()
# plt.show()
# waverider_df.to_csv(f'./results/checkpoints/daily_max_wave_height.csv')

# get psd value and average wave height across 3 day spec
var = ['Hs(Hm0)(m)','Tp(s)','Tz(Tm)(s)','v_wind'][3]     # easy selector on index

mean_vars = []
var_df = pd.read_csv('./results/checkpoints/daily_windspeed.csv', parse_dates=True, index_col=0)
var_df['v_wind'] = np.sqrt(np.square(var_df['u10(m/s)']) + np.square(var_df['v10(m/s)']))
var_df['angle'] = np.mod(180 + 180/pi * np.arctan2(var_df['u10(m/s)'],var_df['v10(m/s)']), 360)

waverider_df = pd.read_csv('./results/checkpoints/Hpg_wave_all_years.csv', parse_dates=True, index_col=0)
waverider_df.drop(waverider_df[waverider_df['Hs(Hm0)(m)'] == 9999].index, inplace=True)
waverider_df = waverider_df.groupby(pd.to_datetime(waverider_df.index).date).agg({'Hs(Hm0)(m)':'mean', 'Tp(s)':'mean','Tz(Tm)(s)':'mean'})
waverider_df.index.name = 'Date/Time(GMT)'
var_df[['Hs(Hm0)(m)','Tp(s)','Tz(Tm)(s)']] = waverider_df[['Hs(Hm0)(m)','Tp(s)','Tz(Tm)(s)']].astype(float)
var_df = var_df.loc[datetime(year=2024, month=2, day=1):datetime(year=2025, month=3, day=1)]

var_df.index = pd.to_datetime(var_df.index)
for date in df.index:
    date = date.date()
    var_slice = var_df.loc[date:date+timedelta(days=3)]
    mean_var = var_slice.mean()
    mean_vars.append(mean_var[var])
df[f'mean_{var.split("(")[0]}'] = mean_vars


# print(df)

target_fs = [0, 1, 2, 5, 10, 15]
fig, axs = plt.subplots(2, ceil(len(target_fs) / 2))
for f, ax in zip(target_fs, axs.ravel()):
    plt.sca(ax)
    plt.scatter(df[f'mean_{var.split("(")[0]}'], df[f'{f}'], c=mdates.date2num(df.index), cmap='twilight_shifted_r')
    plt.title(f'{f} Hz')
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
cb = plt.colorbar()
loc = mdates.AutoDateLocator()
cb.ax.yaxis.set_major_locator(loc)
cb.ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
plt.tight_layout()
plt.savefig(f'./results/figures/psd_{var.split("(")[0]}_hysteresis.png')
# plt.show()

# target_fs = np.arange(0, 51, 1)
# for f in target_fs:
#     fig, ax = plt.subplots()
#     plt.scatter(df[f'mean_{var.split("(")[0]}'], df[f'{f}'], c=mdates.date2num(df.index), cmap='twilight_shifted_r')
#     plt.title(f'{f} Hz')
#     # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
#     cb = plt.colorbar()
#     loc = mdates.AutoDateLocator()
#     cb.ax.yaxis.set_major_locator(loc)
#     cb.ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
#     for i, date in enumerate(df.index):
#         ax.annotate(date.month, (df[f'mean_{var.split("(")[0]}'][i], df[f'{f}'][i]))
#     plt.tight_layout()
#     plt.show()


# windspeed
# 2 Hz v slight link
# 

# fig, axs = plt.subplots(2, 2)
# for ax, label in zip(axs.ravel(), ['Hs(Hm0)(m)','Tz(Tm)(s)','v_wind','Tp(s)']):
#     ax.plot(mdates.date2num(var_df.index), var_df[label])
#     ax.set_title(label)
#     fig.autofmt_xdate()
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
# plt.tight_layout()
# plt.savefig(f'./results/figures/weather_timeseries.png')
# plt.show()
