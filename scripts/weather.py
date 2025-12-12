import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
from math import pi



def dms_to_dd(degrees, minutes=0, seconds=0):
    return degrees + (minutes/60) + (seconds/3600)


def plot_weather():
    weather_data = pd.read_csv('./results/checkpoints/weather.csv', sep=',', index_col=[0, 1], comment='#', na_values=['   --- ', '   ---'])
    weather_data.index = [np.datetime64(f'{date[0]}-{date[1] if date[1] > 9 else f"0{date[1]}"}', 'D') for date in weather_data.index]
    # print(weather_data)
    deployment_data = weather_data.loc[np.datetime64('2023-09-01'):]
    axs = deployment_data.plot.line(None, subplots=True, legend=False, grid=True, figsize=(12, 12))
    for ax, label in zip(axs, ['Max temp (degC)', 'Min temp (degC)', 'AF (days)', 'Rainfall (mm)', 'Sun (hours)']):
        ax.set_ylabel(label)
    plt.tight_layout()
    plt.savefig('./results/figures/weather_data.png')
    plt.show()


def plot_rain_storms():
    weather_data = pd.read_csv('./results/checkpoints/weather.csv', sep=',', index_col=[0, 1], comment='#', na_values=['   --- ', '   ---'], skipinitialspace=True)
    weather_data.index = [np.datetime64(f'{date[0]}-{date[1] if date[1] > 9 else f"0{date[1]}"}', 'D') for date in weather_data.index]
    rain_data = weather_data.loc[np.datetime64('2023-09-01'):, ['rain']]
    rain_data['rain'] = rain_data['rain'].astype(float)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(rain_data.index, rain_data['rain'])
    ax.set_ylabel('Rainfall (mm)')
    ax.grid(which='both') 
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,4,7,10)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=()))
    
    storms_data = pd.read_csv('./results/checkpoints/storms.csv', sep=',', index_col=0, comment='#')
    prev_storm_end = 0
    for storm in storms_data.index:
        dates = storms_data.loc[storm, ['start_date', 'end_date']]
        ax.axvspan(np.datetime64(dates['start_date']), np.datetime64(dates['end_date'])+1, label=storm, facecolor='r', alpha=0.5)
        if prev_storm_end and np.datetime64(dates['end_date']) - prev_storm_end < 10:
            ax.text(np.datetime64(dates['start_date']), 30, storm, rotation=90)
        else:
            ax.text(np.datetime64(dates['start_date']), 20, storm, rotation=90)
        prev_storm_end = np.datetime64(dates['end_date'])
    plt.tight_layout()
    # plt.savefig('./results/figures/rainfall_storms.png')
    plt.show()


def plot_era5_data(file_path:str):
    import xarray as xr
    import cartopy.crs as ccrs
    
    if file_path.split('.')[-1 == 'grib']:
        data = xr.open_dataset(file_path, engine='cfgrib')
        print(data)
        times = data.time.values
        steps = data.step.values
        print(times)
        
        df = pd.DataFrame(columns=['timestamp','V','rainfall'])
        for time in times:
            arr = []
            hour_data = data.sel(time=time, longitude=1.50, latitude=53.0)
            print(hour_data)
            V = np.sqrt(np.square(hour_data.v10.values) + np.square(hour_data.u10.values))
            print(time, V)
            for step in steps:
                rain_data = data.sel(time=time, step=step, longitude=1.50, latitude=53.0)
                print(rain_data.tp.values)

            # arr.append([time, V, rainfall])
        df = df.set_index('timestamp')
        print(df)
        df.to_csv(f'./results/checkpoints/hourly_era5.csv')
        
        plot_time = 'daily'
        # df = df.groupby(pd.to_datetime(df.index).date).agg({'rainfall(mm)': 'sum'}).reset_index()
        df = df.groupby(pd.to_datetime(df.index).date).agg({'u10(m/s)': 'mean', 'v10(m/s)': 'mean'})
        df.index.name = 'timestamp'
        print(df)
        df.to_csv(f'./results/checkpoints/daily_{var}.csv')
    else: 
        var = file_path.split('.')[-2].split('_')[-1]
        plot_time = file_path.split('/')[-1].split('_')[0]
        df = pd.read_csv(file_path, parse_dates=['timestamp'], header=0)
        df = df.set_index('timestamp')        
        if 'windspeed' in file_path:
            df['V'] = np.sqrt(np.square(df['u10(m/s)']) + np.square(df['v10(m/s)']))
            df['angle'] = np.mod(180 + 180/pi * np.arctan2(df['u10(m/s)'],df['v10(m/s)']), 360)
        print(df)
    
    start_date = np.datetime64('2023-09-01')
    
    fig, ax = plt.subplots()
    df = df[start_date:]
    # ax.bar(mdates.date2num(df.index), df['rainfall(mm)'])
    # ax.set_ylabel(f'{plot_time.capitalize()} rainfall (mm)');
    # ax.plot(mdates.date2num(df.index), df['u10(m/s)'], label='u10')
    # ax.plot(mdates.date2num(df.index), df['v10(m/s)'], label='v10')
    ax.plot(mdates.date2num(df.index), df['V'], label='V')
    # ax2 = ax.twinx()
    # ax2.plot(mdates.date2num(df.index), df['angle'], label='angle', color=(0.761, 0.455, 0, 0.5))
    # plt.legend()
    ax.set_ylabel(f'Average {plot_time} wind speed (m/s)')
    # ax2.set_ylabel(f'{plot_time.capitalize()} wind angle')
    # ax2.yaxis.label.set_color((0.761, 0.455, 0))
    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    
    storms_data = pd.read_csv('./results/checkpoints/storms.csv', sep=',', index_col=0, comment='#')
    prev_storm_end = 0
    for storm in storms_data.index:
        dates = storms_data.loc[storm, ['start_date', 'end_date']]
        ax.axvspan(np.datetime64(dates['start_date']), np.datetime64(dates['end_date'])+1, label=storm, facecolor='r', alpha=0.3)
        # if prev_storm_end and np.datetime64(dates['end_date']) - prev_storm_end < 10:
        #     ax.text(np.datetime64(dates['start_date']), 30, storm, rotation=90)
        # else:
        #     ax.text(np.datetime64(dates['start_date']), 20, storm, rotation=90)
        # prev_storm_end = np.datetime64(dates['end_date'])
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'./results/figures/{plot_time}_{var}.png')
    
    
def era5_data_to_csv(file_path:str):
    try:
        wind_data = xr.open_dataset(file_path, engine='cfgrib', 
                                    filter_by_keys={'shortName': ['10u', '10v']})
        wind_data = wind_data.sel(longitude=1.50, latitude=53.0, method='nearest')
    except Exception as e:
        print(f"Error reading wind data: {e}")
        return
    
    try:
        precip_data = xr.open_dataset(file_path, engine='cfgrib', 
                                    filter_by_keys={'shortName': 'tp'})
        precip_data = precip_data.sel(longitude=1.50, latitude=53.0, method='nearest')
    except Exception as e:
        print(f"Error reading precipitation data: {e}")
        return
    
    df_list = []
    for time in precip_data.time.values:
        for step in precip_data.step.values:
            wind_time = pd.to_datetime(time) + pd.to_timedelta(step)
            try:
                wind_slice = wind_data.sel(time=wind_time)
                u10 = float(wind_slice.u10.values) if 'u10' in wind_slice else 0
                v10 = float(wind_slice.v10.values) if 'v10' in wind_slice else 0
                V = np.sqrt(u10**2 + v10**2)
                V_angle = np.mod(180 + 180/pi * np.arctan2(u10,v10), 360)
            except:
                print(f"WIND SLICE NOT FOUND FOR {wind_time}")
                V = 0
                V_angle = 0
            
            precip_slice = precip_data.sel(time=time, step=step)
            tp = float(precip_slice.tp.values) if 'tp' in precip_slice else 0
            
            df_list.append({
                'timestamp': wind_time,
                'V(m/s)': V,
                'V_angle': V_angle,
                'rainfall(mm)':tp*1000
            })
    df = pd.DataFrame(df_list)
    df = df.set_index('timestamp').sort_index()
    df = df.dropna(axis=0)
    mask = (df.index >= pd.to_datetime('20231109')) & (df.index <= pd.to_datetime('20250904'))
    df = df.loc[mask]
    
    print(df)
    print(df.shape)
    df.to_csv('./results/combined_weather.csv')
    

def plot_era5_csv(file_path, plot_daily=False, plot_storms=False, get_df=False):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    if plot_daily:
        df = df.groupby(pd.to_datetime(df.index).date).agg({'V(m/s)': 'mean', 'rainfall(mm)': 'mean'})
    if get_df: return df
    
    fig, ax = plt.subplots()
    # ax.plot(mdates.date2num(df.index), df['V(m/s)'], label='V(m/s)', color=(0.8, 0.0, 0.0, 0.8))
    ax.bar(mdates.date2num(df.index), df['rainfall(mm)'], label='Rainfall(mm)', width=1.6, color=(0.0, 0.0, 0.8, 0.8))
    # ax2 = ax.twinx()
    # ax2.bar(mdates.date2num(df.index), df['rainfall(mm)'], label='Rainfall(mm)', width=1.6, color=(0.0, 0.0, 0.8, 0.8))
    
    # ax.set_ylabel(f"{'Mean daily' if plot_daily else 'Hourly'} wind speed (m/s)", color=(0.8, 0.0, 0.0))
    ax.set_ylabel(f"{'Mean daily' if plot_daily else 'Hourly'} rainfall (mm)")
    # ax2.set_ylabel(f"{'Mean daily' if plot_daily else 'Hourly'} rainfall (mm)", color=(0.0, 0.0, 0.8))
    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(2, 3, 5, 6, 8, 9, 11, 12)))
    
    if plot_storms:
        storms_data = pd.read_csv('./results/checkpoints/storms.csv', sep=',', index_col=0, comment='#')
        prev_storm_end = 0
        for storm in storms_data.index:
            dates = storms_data.loc[storm, ['start_date', 'end_date']]
            ax.axvspan(np.datetime64(dates['start_date']), np.datetime64(dates['end_date'])+1, label=storm, facecolor='r', alpha=0.3)
            # if prev_storm_end and np.datetime64(dates['end_date']) - prev_storm_end < 10:
            #     ax.text(np.datetime64(dates['start_date']), 30, storm, rotation=90)
            # else:
            #     ax.text(np.datetime64(dates['start_date']), 20, storm, rotation=90)
            # prev_storm_end = np.datetime64(dates['end_date'])
    
    plt.tight_layout()
    plt.savefig(f"./results/figures/{'daily' if plot_daily else 'hourly'}_rainfall.png")


def plot_tidal_data(file_path, t_start:datetime, t_end):
    # df = pd.read_csv(file_path, sep='\s+', skiprows=[0,1,2,3,4,5,6,7,8,10])
    # df.drop('Cycle', axis=1, inplace=True)
    # df.loc[df['ASLVBG02'].str.contains('N'), ['ASLVBG02', 'Residual']] = '-1.0'
    # df[['ASLVBG02', 'Residual']] = df[['ASLVBG02', 'Residual']].apply(lambda x: x.str.strip('M'))
    # df[['ASLVBG02', 'Residual']] = df[['ASLVBG02', 'Residual']].apply(pd.to_numeric)
    # # df['ASLVBG02'] = pd.to_numeric(df['ASLVBG02'].str.strip('M'))
    # df.set_index(pd.to_datetime(df['Date'] + df['Time'].astype(str), format = '%Y/%m/%d%H:%M:%S'), inplace=True)
    
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    
    if t_end is timedelta:
        t_end = t_start + t_end
    plot_df = df[t_start:t_end]
    
    fig, ax = plt.subplots()
    ax.plot(plot_df.index, plot_df['ASLVBG02'])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(12))
    
    ax.set_title('Cromer Tidal Gauge (National Oceanographic Centre)')
    ax.set_ylabel('Tidal Height (m)')
    
    plt.tight_layout()
    # plt.savefig(f'./results/figures/Tidal_Plots/{t_start.date()}_{t_end.date()}_tidal.png')
    plt.show()


def plot_waverider_csv(file_path:str, plot_daily=False, get_df=False):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    mask = (df.index >= pd.to_datetime('20231109')) & (df.index <= pd.to_datetime('20250904'))
    df = df.loc[mask]
    
    # replacing flagged error values with NaN for later removal/handling (5 unused)
    df.loc[df.Flag == 1, ['Hs(Hm0)(m)','Hmax(m)','Tp(s)','Tz(Tm)(s)','Dirp(degrees)','Spread(deg)']] = np.NaN
    df.loc[df.Flag == 2, ['Tp(s)','Dirp(degrees)','Spread(deg)']] = np.NaN
    df.loc[df.Flag == 3, ['Dirp(degrees)','Spread(deg)']] = np.NaN
    df.loc[df.Flag == 4, ['Spread(deg)']] = np.NaN
    df.loc[df.Flag == 6, ['Dirp(degrees)','Spread(deg)']] = np.NaN
    df.loc[df.Flag == 7, ['Hs(Hm0)(m)','Hmax(m)','Tp(s)','Tz(Tm)(s)','Dirp(degrees)','Spread(deg)','SST(degC)']] = np.NaN
    df.loc[df.Flag == 8, ['SST(degC)']] = np.NaN
    df.loc[df.Flag == 9, ['Hs(Hm0)(m)','Hmax(m)','Tp(s)','Tz(Tm)(s)','Dirp(degrees)','Spread(deg)','SST(degC)']] = np.NaN
    df = df.dropna(axis=0)
    
    if plot_daily:
        df = df.groupby(pd.to_datetime(df.index).date).agg({'Hs(Hm0)(m)': 'mean', 'Hmax(m)': 'mean', 'Tp(s)': 'mean', 'Tz(Tm)(s)': 'mean', 'Dirp(degrees)': 'mean', 'Spread(deg)': 'mean', 'SST(degC)': 'mean'})
    if get_df: return df
    
    ### Wave height
    fig = plt.figure(); ax = fig.add_subplot(111)
    ax.plot(mdates.date2num(df.index), df['Hs(Hm0)(m)'], label='Hs(m)', color=(0.8, 0.0, 0.0))
    ax.set_ylabel(f"{'Mean daily' if plot_daily else 'Hourly'} wave height (m/s)", color=(0.8, 0.0, 0.0))
    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(2, 3, 5, 6, 8, 9, 11, 12)))
    plt.tight_layout()
    plt.savefig(f"./results/figures/{'daily' if plot_daily else 'hourly'}_wave_height.png")
    
    ### Wave period
    fig = plt.figure(); ax = fig.add_subplot(111)
    ax.plot(mdates.date2num(df.index), df['Tp(s)'], label='Peak wave period (s)', color=(0.8, 0.0, 0.0, 0.4))
    ax2 = ax.twinx()
    ax2.plot(mdates.date2num(df.index), df['Tz(Tm)(s)'], label='Zero-crossing wave period (s)', color=(0.0, 0.0, 0.8, 0.4))
    ax.set_ylabel(f"{'Mean daily' if plot_daily else 'Hourly'} peak wave period (s)", color=(0.8, 0.0, 0.0))
    ax2.set_ylabel(f"{'Mean daily' if plot_daily else 'Hourly'} zero-crossing wave period (s)", color=(0.0, 0.0, 0.8))
    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(2, 3, 5, 6, 8, 9, 11, 12)))
    plt.tight_layout()
    plt.savefig(f"./results/figures/{'daily' if plot_daily else 'hourly'}_wave_period.png")


def plot_met_csv(file_path:str, plot_daily=False, get_df=False):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    mask = (df.index >= pd.to_datetime('20231109')) & (df.index <= pd.to_datetime('20250904'))
    df = df.loc[mask]
    
    # replacing flagged error values with NaN for later removal/handling
    df.loc[df.Flag == 1, ['Wind(m/s)']] = np.NaN
    df.loc[df.Flag == 2, ['WindDir(deg)']] = np.NaN
    df.loc[df.Flag == 3, ['Baro(hPa)']] = np.NaN
    df.loc[df.Flag == 4, ['Tair(degC)']] = np.NaN
    df.loc[df.Flag == 5, ['Gust(m/s)']] = np.NaN
    df.loc[df.Flag == 6, ['Rainfall(mm)']] = np.NaN
    df.loc[df.Flag == 7, ['Solar(W/m^2)', 'UV(W/m^2)']] = np.NaN
    df.loc[df.Flag == 8, ['RH(%)']] = np.NaN
    df.loc[df.Flag == 9, ['Wind(m/s)', 'WindDir(deg)', 'Gust(m/s)', 'Baro(hPa)', 'Tair(degC)', 'Rainfall(mm)', 'Solar(W/m^2)', 'UV(W/m^2)', 'RH(%)']] = np.NaN
    # below are all undrecoreded parameters
    df = df.drop(columns=['Rainfall(mm)', 'Solar(W/m^2)', 'UV(W/m^2)', 'RH(%)'], axis=1)
    df = df.dropna(axis=0)
    
    if plot_daily:
        df = df.groupby(pd.to_datetime(df.index).date).agg({'Wind(m/s)': 'mean', 'WindDir(deg)': 'mean', 'Gust(m/s)': 'mean', 'Baro(hPa)': 'mean', 'Tair(degC)': 'mean'})
    if get_df: return df
        
    ### Wind & rainfall
    fig = plt.figure(); ax = fig.add_subplot(111)
    ax.plot(mdates.date2num(df.index), df['Wind(m/s)'], label='Wind speed (m/s)', color=(0.8, 0.0, 0.0))
    ax.set_ylabel(f"{'Mean daily' if plot_daily else 'Hourly'} wind speed (m/s)", color=(0.8, 0.0, 0.0))
    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(2, 3, 5, 6, 8, 9, 11, 12)))
    plt.tight_layout()
    plt.savefig(f"./results/figures/{'daily' if plot_daily else 'hourly'}_windspeed.png")


def plot_combined_weather(plot_daily=False, plot_storms=False, t_start=None, t_end=None): 
    df_wave = plot_waverider_csv('./results/checkpoints/hpg_wave.csv', plot_daily=plot_daily, get_df=True)
    df_met = plot_met_csv('./results/checkpoints/hpg_met.csv', plot_daily=plot_daily, get_df=True)
    df_era5 = plot_era5_csv('./results/checkpoints/combined_weather.csv', plot_daily=plot_daily, get_df=True)
    df = pd.concat([df_wave, df_met, df_era5], axis=1)
    
    if t_end is timedelta:
        t_end = t_start + t_end
    df = df[t_start:t_end]
    df = df[[('00:00' or '30:00') in str(s) for s in df.index]]
    
    vars = ['Wind(m/s)','Hs(Hm0)(m)','Tp(s)','Tz(Tm)(s)','rainfall(mm)']
    labels = ['Wind speed (m/s)','Wave height (m)','Peak wave period (s)','Zero-crossing period (s)','Rainfall (mm)']
    
    fig, axs = plt.subplots(len(vars), 1, figsize=(12, 6), sharex=True)
    for ax, col, label in zip(axs, vars, labels):
        if col=='rainfall(mm)':
            ax.bar(mdates.date2num(df.index), df[col])
        else:
            ax.plot(mdates.date2num(df.index), df[col])
        ax.set_ylabel(label)
        ax.grid(axis='x', which='both')
    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    # ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    # ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(2, 3, 5, 6, 8, 9, 11, 12)))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    
    if plot_storms:
        storms_data = pd.read_csv('./results/checkpoints/storms.csv', sep=',', index_col=0, comment='#')
        for storm in storms_data.index:
            dates = storms_data.loc[storm, ['start_date', 'end_date']]
            axs[-1].axvspan(np.datetime64(dates['start_date']), np.datetime64(dates['end_date'])+1, label=storm, facecolor='r', alpha=0.3)
    
    plt.subplots_adjust(hspace=0)
    # plt.savefig(f"./results/figures/{'daily' if plot_daily else 'hourly'}_combined.png")
    plt.show()


if __name__ == "__main__":
    daily = False
    m = 5
    d0 = datetime(year=2025, month=m, day=1)
    d1 = datetime(year=2025, month=(m+1)%12, day=1)
    d0 = datetime(year=2024, month=12, day=8); d1 = datetime(year=2024, month=12, day=14)
    # plot_weather()
    # plot_rain_storms()
    # era5_data_to_csv('era5_final.grib')
    # plot_era5_csv('./results/checkpoints/combined_weather.csv', plot_daily=daily, plot_storms=True)
    # plot_tidal_data('./results/checkpoints/CRO_final.csv', d0, d1)
    
    # plot_waverider_csv('./results/checkpoints/hpg_wave.csv', plot_daily=daily)
    # plot_met_csv('./results/checkpoints/hpg_met.csv', plot_daily=daily)
    plot_combined_weather(plot_daily=daily, plot_storms=False, t_start=d0, t_end=d1)
