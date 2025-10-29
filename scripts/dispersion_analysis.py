import os

import matplotlib.pyplot as plt
import numpy as np
import pyasdf
import pycwt
import scipy


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
############ PAMAETER SECTION ##############
############################################

# input file info
rootpath = os.path.join(os.path.expanduser("~"), "/home/harry/Documents/0. PhD/DiSTANS/Toolkit/NoisePy4DAS-SeaDAS/")  # root path for this data processing
sfile = os.path.join(rootpath, "./CCF.h5")  # ASDF file containing stacked data
outdir = os.path.join(rootpath, "results/figures/")  # dir where to output dispersive image and extracted dispersion

# data type and cross-component
stack_method = "linear"  # which stacked data to measure dispersion info
lag_type = "sym"  # options to do measurements on the 'neg', 'pos' or 'sym' lag (average of neg and pos)
ncomp = 1
if ncomp == 1:
    rtz_system = ["ZZ"]
else:
    rtz_system = ["ZR", "ZT", "ZZ", "RR", "RT", "RZ", "TR", "TT", "TZ"]
# index for plotting the figures
post1 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
post2 = [0, 1, 2, 0, 1, 2, 0, 1, 2]

# targeted freq bands for dispersion analysis
fmin = 0.01
fmax = 10
per = np.arange(int(1 / fmax), int(1 / fmin), 0.02)

# set time window for disperion analysis
vmin = 0.5
vmax = 4.5
vel = np.arange(vmin, vmax, 0.02)

# basic parameters for wavelet transform
dj = 1 / 12
s0 = -1
J = -1
wvn = "morlet"

# get station-pair name ready for output
tmp = sfile.split("/")[-1].split("_")
spair = tmp[0]

# load basic data information including dt, dist and maxlag
with pyasdf.ASDFDataSet(sfile, mode="r") as ds:
    source = "500"
    try:
        slist = ds.auxiliary_data.list()
        rlist = ds.auxiliary_data[source].list()
        maxlag = ds.auxiliary_data[source][rlist[0]].parameters["maxlag"]
        dist = 500
        # dist = ds.auxiliary_data[dtype]["500_500"].parameters["dist"]
        dt = ds.auxiliary_data[source][rlist[0]].parameters["dt"]
    except Exception as e:
        raise ValueError(e)

# initialize the plotting procedure
if ncomp == 3:
    fig, ax = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
else:
    plt.figure(figsize=(4, 3))

##################################################
############ MEASURE GROUP VELOCITY ##############
##################################################

# loop through each component
for comp in rtz_system:
    cindx = rtz_system.index(comp)
    pos1 = post1[cindx]
    pos2 = post2[cindx]

    # load cross-correlation functions
    with pyasdf.ASDFDataSet(sfile, mode="r") as ds:
        try:
            tdata = ds.auxiliary_data[source][comp].data[:]
        except Exception as e:
            raise ValueError(e)

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

    # wavelet transformation
    cwt, sj, freq, coi, _, _ = pycwt.cwt(data, dt, dj, s0, J, wvn)

    # do filtering here
    if (fmax > np.max(freq)) | (fmax <= fmin):
        raise ValueError("Abort: frequency out of limits!")
    freq_ind = np.where((freq >= fmin) & (freq <= fmax))[0]
    cwt = cwt[freq_ind]
    freq = freq[freq_ind]

    # use amplitude of the cwt
    period = 1 / freq
    rcwt, pcwt = np.abs(cwt) ** 2, np.angle(cwt)

    # interpolation to grids of freq-vel
    fc = scipy.interpolate.interp2d(dist / tvec, period, rcwt)
    rcwt_new = fc(vel, per)

    # do normalization for each frequency
    for ii in range(len(per)):
        rcwt_new[ii] /= np.max(rcwt_new[ii])

    # extract dispersion curves for ZZ, RR and TT
    if comp == "ZZ" or comp == "RR" or comp == "TT":
        nper, gv = extract_dispersion(rcwt_new, per, vel)
        fphase = open(os.path.join(outdir, spair + "_group_" + comp + ".csv"), "w")
        for iii in range(len(nper)):
            fphase.write("%5.1f %5.2f\n" % (nper[iii], gv[iii]))
        fphase.close()

    # plot wavelet spectrum
    if ncomp == 3:
        # dispersive image
        im = ax[pos1, pos2].imshow(
            np.transpose(rcwt_new),
            cmap="jet",
            extent=[per[0], per[-1], vel[0], vel[-1]],
            aspect="auto",
            origin="lower",
        )
        # extracted dispersion curves
        if comp == "ZZ" or comp == "RR" or comp == "TT":
            ax[pos1, pos2].plot(nper, gv, "w--")
        ax[pos1, pos2].set_xlabel("Period [s]")
        ax[pos1, pos2].set_ylabel("U [km/s]")
        if cindx == 1:
            ax[pos1, pos2].set_title("%s %5.2fkm linear" % (spair, dist))
        ax[pos1, pos2].xaxis.set_ticks_position("bottom")
        cbar = fig.colorbar(im, ax=ax[pos1, pos2])
        font = {"family": "serif", "color": "green", "weight": "bold", "size": 16}
        ax[pos1, pos2].text(int(per[-1] * 0.85), vel[-1] - 0.5, comp, fontdict=font)
    else:
        plt.imshow(
            np.transpose(rcwt_new),
            cmap="jet",
            extent=[per[0], per[-1], vel[0], vel[-1]],
            aspect="auto",
            origin="lower",
        )
        # extracted disperison curves
        plt.plot(nper, gv, "w--")
        plt.xlabel("Period [s]")
        plt.ylabel("U [km/s]")
        plt.title("%s %5.2fkm linear" % (spair, dist))
        font = {"family": "serif", "color": "green", "weight": "bold", "size": 16}
        plt.text(int(per[-1] * 0.85), vel[-1] - 0.5, comp, fontdict=font)
        plt.tight_layout()

# save figures
outfname = outdir + "/{0:s}_{1:s}.pdf".format(spair, lag_type)
if ncomp == 3:
    fig.tight_layout()
    fig.savefig(outfname, format="pdf", dpi=400)
    plt.close()
else:
    plt.savefig(outfname, format="pdf", dpi=400)
    plt.close()
