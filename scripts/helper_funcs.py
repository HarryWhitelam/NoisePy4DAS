from math import cos, asin, sqrt
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(hav))

def closest(points, v):
    return min(points, key=lambda p: haversine_distance(v[0],v[1],p[1],p[0]))


track_pt = np.loadtxt('track_gps.csv', delimiter=',', skiprows=1, usecols=(1,2), comments='#')[:, ::-1] # read in the track points and swap the two columns (let longitude precede latitude)
known_pt = np.loadtxt('known_gps.csv', delimiter=',', skiprows=1, usecols=(1,0,2), comments='#')


uinta_gps_points = [
    # [52.89939421, 1.38699213],      # cable start
    [52.89970906, 1.38548287],      # fall one
    [52.90026837, 1.38414773],      # forest entrance
    [52.90091812, 1.38306587],      # forest exit
    [52.90232053, 1.38018163],      # fall path
    [52.90307644, 1.37774255],      # marlings sign
    [52.90438791, 1.37468231],      # fall two question mark
    [52.90499186, 1.37081761],      # inner curve
    [52.90613392, 1.36821678],      # field divider
]

gearth_gps_points = [
    [52.899706, 1.385486],      # fall one
    [52.900269, 1.384068],      # forest entrance
    [52.901046, 1.383027],      # forest exit
    [52.902339, 1.380199],      # fall path
    [52.903058, 1.377689],      # marlings sign
    [52.904434, 1.374667],      # fall two question mark
    [52.904997, 1.370794],      # inner curve
    [52.906136, 1.368230],      # field divider
]

# get_closest_gps(track_pt, [52.9030722, 1.3777083333333333])
gps_list = []
for i in range(len(uinta_gps_points)):
    ref_1 = closest(track_pt, uinta_gps_points[i])
    ref_2 = closest(track_pt, gearth_gps_points[i])
    gps_list.append([ref_1, ref_2])

for gps in gps_list: print(gps)
