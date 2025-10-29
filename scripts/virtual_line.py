import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from math import sin, cos, asin, sqrt

from phase_shift import load_xcorr, get_dispersion, get_max_cs


def haversine_distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(hav))


def get_closest_points(line_pts, track_pts):
    closest_pts = []
    last_used_idx = 0
    
    for point in line_pts.geometry:
        candidates = track_pts.iloc[last_used_idx + 1:]
        
        if len(candidates) == 0:
            closest_pts.append(track_pts.index[-1])
            continue
            
        distances = candidates.geometry.distance(point)
        
        distances = candidates.geometry.distance(point)
        min_idx = distances.idxmin()
        closest_pts.append(min_idx)
        last_used_idx = track_pts.index.get_loc(min_idx)
    
    return closest_pts


### GENERATE VIRTUAL LINE ###
def generate_virtual_line(track_pts_path, plot=False, save=False):
    track_pts = pd.read_csv(track_pts_path, index_col='channel_no')

    track_pts = gpd.GeoDataFrame(track_pts, 
        geometry=gpd.points_from_xy(track_pts['lon'], track_pts['lat']), 
        crs='EPSG:4326'
    ).to_crs('EPSG:32630')

    coefs = poly.polyfit(track_pts['lon'], track_pts['lat'], 1)
    ffit = poly.polyval(track_pts['lon'], coefs)
    line_pts = pd.DataFrame({'lon': track_pts['lon'], 'lat': ffit})

    line_gdf = gpd.GeoDataFrame(
        line_pts[['lat', 'lon']], geometry=gpd.points_from_xy(line_pts['lon'], line_pts['lat'], crs='EPSG:4326')
    )
    coords = [(row.geometry.x, row.geometry.y) for _, row in line_gdf.iterrows()]
    line_gs = gpd.GeoSeries(LineString(coords), crs='EPSG:4326')
    line_gs = line_gs.to_crs(crs='EPSG:32630')

    interp_gs = line_gs.geometry[0].interpolate(np.arange(0, line_gs.length[0], 1))
    interp_gs = gpd.GeoSeries(interp_gs, crs='EPSG:32630')

    closest_pts = get_closest_points(interp_gs, track_pts)

    interp_gs = interp_gs.to_crs('EPSG:4326')
    track_pts = track_pts.to_crs('EPSG:4326')


    if plot:
        fig, ax = plt.subplots()
        # ax.plot(track_pts['lon'], track_pts['lat'])
        track_pts.plot(ax=ax, markersize=5)
        interp_gs.plot(ax=ax, markersize=5)
        for point, ch_no in zip(interp_gs.geometry, closest_pts):
            track_pt = track_pts.loc[ch_no]
            ax.plot([point.x, track_pt.lon], [point.y, track_pt.lat], '--r')
        plt.tight_layout()
        plt.show()

    final_df = track_pts.loc[closest_pts]
    if save:
        final_df.to_csv('results/checkpoints/interp_virtual_line.csv', columns=['lat', 'lon'])
    return final_df


### SLICE CROSS CORRELATION & DISPERSION ###




if __name__ == '__main__':
    old_cha_nos = True
    
    # vl_df = generate_virtual_line('./results/checkpoints/interp_ch_pts.csv')
    vl_df = pd.read_csv('./results/checkpoints/interp_virtual_line.csv')
    
    xcorr_path = './results/saved_corrs/2024-02-05 12:01:00_4320mins_f0.01:49.9__3850:5750_1m.txt'
    chas = vl_df['channel_no'].to_list()
    chas = list(map(int, chas))
    xcorr_chas = xcorr_path.split('__')[-1].split('_')[0].split(':'); xcorr_chas = [int(cha) for cha in xcorr_chas]
    if old_cha_nos: 
        xcorr_chas = [int(xcorr_cha / 4) for xcorr_cha in xcorr_chas]
    chas = [cha - xcorr_chas[0] for cha in chas if xcorr_chas[0] <= cha <= xcorr_chas[1]]

    stream = load_xcorr(xcorr_path, chas=chas)
    
    cmin = 50.0
    cmax = 1500.0   # 27/11 dropped from 4000.0 to 1500.0
    dc = 10.0       # 27/11 changed from 10.0 to 5.0
    dx = 1
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
    plt.tight_layout()
    plt.show()
    
    for fi in range(len(f)):
        img[:, fi] /= np.max(img[:, fi])
    fig, ax = plt.subplots(figsize=(7.0,5.0))
    im = ax.imshow(img[:,:],aspect='auto', origin='lower', extent=(f[0], f[-1], c[0], c[-1]), interpolation='bilinear')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase velocity (m/s)")
    bar = fig.colorbar(im, ax=ax, pad = 0.1) # if bad add in "format = lambda x, pos: '{:.1f}'.format(x*100)"
    plt.tight_layout()
    plt.show()


### OLD ATTEMPTS ###
# closest_pts = [min(
#         [[track_pt.name, haversine_distance(point.y, point.x, track_pt.lat, track_pt.lon)] for _, track_pt in track_pts.iterrows()], 
#         key=lambda x: x[1]
#     )[0] for point in interp_gs.geometry]

# closest_pts = []
# track_pts_cp = track_pts.copy()
# for i, point in enumerate(interp_gs.geometry):
#     prev_ch = max([prev_ch, closest_pts[-1]]) if len(closest_pts) != 0 else track_pts_cp.iloc[0].name
#     print(i, prev_ch)
#     ch_no = min([[track_pt.name, haversine_distance(point.y, point.x, track_pt.lat, track_pt.lon)] for _, track_pt in track_pts_cp.loc[:prev_ch+10].iterrows()], key=lambda x: x[1])[0]
#     closest_pts.append(ch_no)
#     track_pts_cp.drop(ch_no, inplace=True)

# pts_dict = {i:closest_pts.count(i) for i in closest_pts}
# print(sorted(pts_dict.items(), key=lambda x: x[1]))

# i = 0
# print(len(closest_pts))
# while i < len(closest_pts)-2:
#     end_i = i
#     while end_i < len(closest_pts)-1 and closest_pts[i] == closest_pts[end_i+1]:
#         end_i += 1
#     if i != end_i:
#         print(f'value {closest_pts[i]} from {i} to {end_i} ---- {i}: {closest_pts[i]}; {end_i}: {closest_pts[end_i]}')
#         i = end_i + 1
#     else:
#         i += 1