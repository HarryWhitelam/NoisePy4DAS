###### ADAPTED FROM NOISEPY: https://github.com/noisepy/NoisePy/blob/9fd81c361ec0f86157f99608f04c986e898c7eb9/src/noisepy/imaging/dispersion_analysis.py#L22
# all identifying comments left as original 

import os

import matplotlib.pyplot as plt
import numpy as np
import pyasdf
import pycwt
import pywt
import scipy

# from . import noise_module

"""
this application script of NoisePy is to measure group velocity on the resulted cross-correlation
functions from S2. It uses the wavelet transform to trace the wave energy on multiple frequencies.
Based on our tests, it generates very similar results to those from Frequency-Time Analysis (FTAN).

Authors: Chengxin Jiang (chengxin_jiang@fas.harvard.edu)

NOTE:
    According to Bensen et al., (2007), the central frequency of
    each narrowband filters (equivalent to wavelet tranformed signal
    at each scale) would be different from the instaneous frequency calculated
    using instaneous phase due to spectral linkage. We do not
    correct this effect in this script. Phase velocity is not calculated here,
    but could be expaneded using the phase info of wavelet transformed signal.
"""

def extract_dispersion(amp, per, vel):
    """
    this function takes the dispersion image from CWT as input, tracks the global maxinum on
    the wavelet spectrum amplitude and extract the sections with continous and high quality data

    PARAMETERS:
    ----------------
    amp: 2D amplitude matrix of the wavelet spectrum
    phase: 2D phase matrix of the wavelet spectrum
    per:  period vector for the 2D matrix
    vel:  vel vector of the 2D matrix
    RETURNS:
    ----------------
    per:  central frequency of each wavelet scale with good data
    gv:   group velocity vector at each frequency
    """
    maxgap = 5
    nper = amp.shape[0]
    gv = np.zeros(nper, dtype=np.float32)
    dvel = vel[1] - vel[0]

    # find global maximum
    for ii in range(nper):
        maxvalue = np.max(amp[ii], axis=0)
        indx = list(amp[ii]).index(maxvalue)
        gv[ii] = vel[indx]

    # check the continuous of the dispersion
    for ii in range(1, nper - 15):
        # 15 is the minumum length needed for output
        for jj in range(15):
            if np.abs(gv[ii + jj] - gv[ii + 1 + jj]) > maxgap * dvel:
                gv[ii] = 0
                break

    # remove the bad ones
    indx = np.where(gv > 0)[0]

    return per[indx], gv[indx]

############################################
############ PARAMETER SECTION ##############
############################################

# input file info
rootpath = os.path.join(os.path.expanduser("~"), "distans/noisePy4DAS-SeaDAS")    # root path for this data processing
# rootpath = os.path.join(os.path.expanduser("~"), "Documents/0. PhD/DiSTANS/Toolkit/NoisePy4DAS-SeaDAS/")
sfile = os.path.join(rootpath, "test_stack.txt")                                   # ASDF file containing stacked data
outdir = os.path.join(rootpath, "results/figures/dispersion")                       # dir where to output dispersive image and extracted dispersion

# data type and cross-component
stack_method = "linear"  # which stacked data to measure dispersion info
lag_type = "sym"  # options to do measurements on the 'neg', 'pos' or 'sym' lag (average of neg and pos)

# targeted freq bands for dispersion analysis
fmin = 0.03
fmax = 1
per = np.arange(int(1 / fmax), int(1 / fmin), 0.02)

# set time window for disperion analysis
vmin = 0.5
vmax = 4.5
vel = np.arange(vmin, vmax, 0.02)

# basic parameters for wavelet transform
dj = 1 / 12
s0 = -1
J = 255
wvn = "morl"

# get station-pair name ready for output
tmp = sfile.split("/")[-1].split("_")
spair = tmp[0] + "_" + tmp[1][:-3]

# TODO: below constants need to be parameterised - not constants
maxlag = 8
dist = 1       # in km i think
dt = 1/100

# initialize the plotting procedure
plt.figure(figsize=(4, 3))

##################################################
############ MEASURE GROUP VELOCITY ##############
##################################################

# loop through each component
comp = 'ZZ'

# load cross-correlation functions
tdata = np.loadtxt(sfile, delimiter=',')
print(f'tdata shape: {tdata.shape}')

# stack positive and negative lags
npts = int(1 / dt) * 2 * maxlag + 1
indx = npts // 2

if lag_type == "neg":
    data = tdata[: indx + 1]
elif lag_type == "pos":
    data = tdata[indx:]
elif lag_type == "sym":
    data = 0.5 * tdata[indx:] + 0.5 * np.flip(tdata[: indx + 1], axis=0)
else:
    raise ValueError("parameter of lag_type (L35) is not right! please double check")

# trim the data according to vel window
pt1 = int(dist / vmax / dt)
pt2 = int(dist / vmin / dt)
if pt1 == 0:
    pt1 = 10
if pt2 > (npts // 2):
    pt2 = npts // 2
indx = np.arange(pt1, pt2)
tvec = indx * dt
data = data[indx]

# wavelet transformation - below is deprecated :(
# cwt, sj, freq, coi, _, _ = pycwt.cwt(data, dt, dj, s0, J, wvn)

# ATTEMPT: using pywavelet
scales = np.arange(1, 50)
cwt_frequencies = pywt.scale2frequency(wvn, scales) / dt
print(cwt_frequencies)
cwt, freq = pywt.cwt(data, scales, wvn) 
print(f'cwt shape: {cwt.shape}')
print(f'freq shape: {freq.shape}')
del data

# do filtering here
print('checkpoint 0')
freq_ind = np.where((freq >= fmin) & (freq <= fmax))[0]
cwt = cwt[freq_ind]
freq = freq[freq_ind]
print(f'cwt shape after filter: {cwt.shape}')
print(f'freq shape after filter: {freq.shape}')

# use amplitude of the cwt
print('checkpoint 1')
period = 1 / freq
rcwt, pcwt = np.abs(cwt) ** 2, np.angle(cwt)

# interpolation to grids of freq-vel
print('checkpoint 2')
print(f'dist/tvec shape: {(dist/tvec).shape}')
print(f'period shape: {period.shape}')
print(f'rcwt shape: {rcwt.shape}')
print(f'rcwt[:,:,0] shape: {rcwt[:,:,0].shape}')
d_vec = dist / tvec
# fc = scipy.interpolate.interp2d(dist / tvec, period, rcwt)
# fc = scipy.interpolate.LinearNDInterpolator(d_vec, rcwt)
# fc = scipy.interpolate.RectBivariateSpline(dist / tvec, period, rcwt.T)
fc = scipy.interpolate.RegularGridInterpolator((dist / tvec, period), rcwt[:,:,0].T, method='cubic')
rcwt_new = fc((vel, per))		# changed to tuple for RGI

# do normalization for each frequency
print('checkpoint 3')
for ii in range(len(per)):
    rcwt_new[ii] /= np.max(rcwt_new[ii])

# extract dispersion curves for ZZ, RR and TT   FIXME: This doesn't exist in NoisePyDAS (namely extract_dispersion)
print('checkpoint 4')
if comp == "ZZ" or comp == "RR" or comp == "TT":
    nper, gv = extract_dispersion(rcwt_new, per, vel)
    fphase = open(os.path.join(outdir, spair + "_group_" + comp + ".csv"), "w")
    for iii in range(len(nper)):
        fphase.write("%5.1f %5.2f\n" % (nper[iii], gv[iii]))
    fphase.close()

# plot wavelet spectrum
print('checkpoint 5')
plt.imshow(
    np.transpose(rcwt_new),
    cmap="jet",
    extent=[per[0], per[-1], vel[0], vel[-1]],
    aspect="auto",
    origin="lower",
)
# extracted disperison curves
# plt.plot(nper, gv, "w--")
plt.xlabel("Period [s]")
plt.ylabel("U [km/s]")
plt.title("%s %5.2fkm linear" % (spair, dist))
font = {"family": "serif", "color": "green", "weight": "bold", "size": 16}
plt.text(int(per[-1] * 0.85), vel[-1] - 0.5, comp, fontdict=font)
plt.tight_layout()

# save figures
print('checkpoint 6')
outfname = outdir + "/{0:s}_{1:s}.pdf".format(spair, lag_type)
plt.savefig(outfname, format="pdf", dpi=400)
# plt.show()
plt.close()
