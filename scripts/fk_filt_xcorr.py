import numpy as np
import matplotlib.pyplot as plt
import dascore as dc

xcorr_file = '2025-02-16 00:00:00_1440mins_100f0.01:25.0_1000:1400_1m_1000src_pws.txt'
# xcorr_file = '2025-02-04 00:00:00_20160mins_100f0.1:25.0_1000:1400_10m_1000src_pws.txt'
xcorr = np.loadtxt('/data/localraid/saved_corrs/' + xcorr_file, delimiter=',')
nt, nx = xcorr.shape

xcorr_file_splits = xcorr_file.split('_')
cha1, cha2 = [int(x) for x in xcorr_file_splits[3].split(':')]
dx = float(xcorr_file_splits[4].strip('m'))
dt = 1/float(xcorr_file_splits[2].split('f')[0])
maxlag = int(nt*dt/2)

vmin, vmax = 600, 1500      # in m/s

fk = np.fft.fft2(xcorr)     # 2D fft
freqs = np.fft.fftfreq(nt, dt)
ks = np.fft.fftfreq(nx, dx)

F, K = np.meshgrid(freqs, ks, indexing='ij')
v_apparent = np.abs(F / K)
v_mask = (v_apparent > vmin) & (v_apparent < vmax)

fk_filt = fk * v_mask
xcorr_filt = np.fft.ifft2(fk_filt).real
# print(np.nanpercentile(xcorr_filt, 100))
# print(np.nanpercentile(xcorr_filt, 99.9))
# print(np.nanpercentile(xcorr_filt, 99))
# print(np.nanpercentile(xcorr_filt, 95))
xcorr_filt = xcorr_filt / np.nanpercentile(np.abs(xcorr_filt), 99.9)

fig, axs = plt.subplots(1,2, width_ratios=[1, 0.5])
ax = axs[0]
# plt.imshow(xcorr_filt.T, vmax=2e-2, vmin=-2e-2, cmap='bwr', aspect='auto', origin='lower')
ax.imshow(xcorr_filt.T, vmax=1, vmin=-1, cmap='bwr', aspect='auto', origin='lower')

x = np.linspace(int(xcorr_filt.shape[0]/2),xcorr_filt.shape[0]-1,maxlag+1)
for velocity in [600, 1500]:
    y = [velocity*(xi*dt-maxlag) for xi in x]
    y = [yi/dx for yi in y]
    ax.plot(x, y, linestyle='--', color='k', alpha=0.6)
# ax.set_xlim(0, xcorr_filt.shape[0]-1/2)
ax.set_ylim(0, xcorr_filt.shape[1])

# highlight 200m mark
ax.plot([0, 801], [int(200/dx)]*2, color='k', alpha=0.2)

_ = ax.set_yticks((np.linspace(cha1, cha2-dx, 6) - cha1)/dx, [int(i) for i in np.linspace(cha1, cha2, 6)], fontsize = 10)
ax.set_ylabel("Channel number", fontsize = 10)
_ = ax.set_xticks(np.arange(0, maxlag*(1/dt)*2+1, 1/dt), np.arange(-maxlag, maxlag+1, 1), fontsize=10)
ax.set_xlabel("Time lag (sec)", fontsize = 10)

twiny = ax.twinx()
cha_spacing = 1.0209523838714072
twiny.set_yticks((np.linspace(cha1, cha2, 6) - cha1)/dx, [int(i* cha_spacing) for i in np.linspace(cha1, cha2, 6)], fontsize = 10)
twiny.set_ylabel("Distance along cable (m)", fontsize = 10)

# ax.set_xlim(200, 601)

### DASCore attempt
# patch = dc.Patch(data=xcorr, coords={'time': np.arange(nt)*dt, 'distance': np.arange(nx)*dx}, dims=('time', 'distance'))

# vfilt = np.array([360, 400, 2000, 2200])
# patch_filt = patch.slope_filter(vfilt)
# plt.figure()
# plt.imshow(patch_filt.data.T, vmax=2e-2, vmin=-2e-2, cmap='bwr', aspect='auto', origin='lower')


# show coda arrivals
far_channel = xcorr_filt[:, int(200/dx)]
axs[1].plot(far_channel)
_ = axs[1].set_xticks(np.arange(0, maxlag*(1/dt)*2+1, 1/dt), np.arange(-maxlag, maxlag+1, 1), fontsize=10)
axs[1].set_xlabel("Time lag (sec)", fontsize = 10)

plt.show()
