import os
from obspy import read, read_inventory, UTCDateTime
from obspy.signal import PPSD

file_path = '/media/harry/Elements/UoE Node Data/UEA_nodes_mseed_mV/453003479.1.2025.03.31.13.01.50.000.N.miniseed'
st_in = read(file_path)
print(st_in)
# print(len(st_in))

t0 = UTCDateTime('2025-03-31T17:00:00.00')
t1 = t0 + (60*60*24)

st = st_in.copy()
# st = st.slice(t0, t1)
for tr in st: tr.data *= 0.001      # convert from mV to V
st.decimate(16)     # decimate to 500 Hz (factor of 8)
print(st)
# st.plot()

inv_path = '/media/harry/Elements/UoE Node Data/RESP.5J.03479..DPN.DTSOLO.5.1850.43000.76_6'
# inv_path = '/media/harry/Elements/UoE Node Data/iris_resp_file'
inv = read_inventory(inv_path)
# print(inv)
# inv.plot_response(min_freq=0.001)

st.remove_response(inventory=inv, pre_filt=[0.001, 0.005, 95, 100], plot=True)
# st.plot()
# st.spectrogram()

ppsd = PPSD(st[0].stats, metadata=inv)
ppsd.add(st)

ppsd.plot(xaxis_frequency=True)
ppsd.plot_spectrogram()
