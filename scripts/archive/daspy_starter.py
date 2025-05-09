import daspy
from daspy.basic_tools.visualization import plot
from daspy.advanced_tools.channel import location_interpolation, turning_points
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# dir_path = "../../../../gpfs/data/DAS_data/Data/"
dir_path = '../../temp_data_store/'
gps_path = '../../Deployment/gps_coords.csv'

# tdata = daspy.read(dir_path + 'FirstData_UTC_20231109_134707.573.tdms')
# tdata = daspy.read()


# spec, f = tdata.spectrum()
# Zxx, f, t = tdata.spectrogram()
# fk, f, k = tdata.fk_transform()


# tdata.plot(obj='fk')


# fig, ax = plt.subplots(2, 1, figsize=(6,6))
# tdata.plot(ax=ax[0], obj='waveform', xmode='channel', tmode='origin', xlabel=False, transpose=True, vmax=0.05) # set the spatial axis to the channel number, the time axis to the time after the event occurred, do not draw the x-axis label, invert the default x/y axis, and set the data range to -0.05~0.05
# tdata.plot(ax=ax[1], obj='spectrogram', tmode='origin') # overlap between two windows is 156 points
# plt.tight_layout()
# plt.show()


# spec, f = tdata.spectrum()
# spec = 10 * np.log10(abs(spec) ** 2) # convert the spectrum to units of decibels (dB), using 1 as the reference value
# plot(spec, obj='spectrum', f=f, xmode='channel') # set the spatial axis to the channel number, the time axis to the time after the event occurred, and invert the default x/y axis



### Location Interpolation
# txt_url = 'http://piweb.ooirsn.uw.edu/das/processed/metadata/Geometry/OOI_RCA_DAS_channel_location/north_cable_latlon.txt'
# track_pt = np.loadtxt(txt_url)[:, ::-1]
# known_pt = np.array([[*track_pt[0], 942], [*track_pt[-1], 32459]])
# print(track_pt)
# print(known_pt)

track_pt = np.loadtxt('results/checkpoints/track_pts.csv', delimiter=',', skiprows=1, usecols=(2,1), comments='#') # read in the track points and swap the two columns (let longitude precede latitude)
known_pt = np.loadtxt('results/checkpoints/known_pts.csv', delimiter=',', skiprows=1, usecols=(1,0,2), comments='#')
# print(track_pt)
# print(known_pt)

# known_pt = known_pt[[2], :]
track_pt = track_pt[::-1]       # upside down :(
# print(test_track_pt.shape)

interp_ch = location_interpolation(known_pt, track_pt=track_pt, dx=0.25)
# print(interp_ch) # longitude, latitude, and channel number
interp_ch_df = pd.DataFrame(interp_ch, columns=['lon', 'lat', 'channel_no'])[['lat', 'lon', 'channel_no']]      # rearranged for lat, lon, ch_no
# interp_ch_df.to_csv('results/interp_ch_pts.csv', index=False)

plt.scatter(interp_ch[:, 0], interp_ch[:, 1], c=interp_ch[:, 2], cmap='bone')
plt.scatter(track_pt[:, 0], track_pt[:, 1], c='k', s=1)

### HIGHLIGHTS
highlight_mask = np.isin(interp_ch[:, 2], [4300, 4000, 5000, 6000, 7000, 8000])
highlight_ch = interp_ch[highlight_mask, :]
plt.scatter(highlight_ch[:, 0], highlight_ch[:, 1], c='b', s=30)
print(highlight_ch)

### distance highlight attempt - see above for less messy layout
# cha_spacing = 0.25
# distances = [1042, 1142, 1172, 1272, 1312, 1412, 1820, 1920]        # after forest section start at 962 ish
# d_as_ch = interp_ch[np.isin(interp_ch[:, 2], [d/cha_spacing for d in distances]), :]
# plt.scatter(d_as_ch[:, 0], d_as_ch[:, 1], c='r', s=10)

plt.gca().set_aspect('equal')
plt.title('Interpolated channel location')
plt.tight_layout()
plt.show()



# turning_h, turning_v = turning_points(interp_ch, depth_info=True) # the data contains depth information, detect turning points both horizontally and vertically

# plt.scatter(interp_ch[:, 0], interp_ch[:, 1], c='y', s=5)
# plt.scatter(interp_ch[turning_v, 0], interp_ch[turning_v, 1], c='g', s=5, label='vertical')
# plt.scatter(interp_ch[turning_h, 0], interp_ch[turning_h, 1], c='r', s=5, label='horizontal')
# plt.gca().set_aspect('equal')
# plt.title('Turning points')
# plt.legend()
# plt.show()
