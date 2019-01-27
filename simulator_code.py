#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:27:21 2018

@author: mbrogi
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import interpolate, constants, stats
import sys
import time
import emcee, corner

############################################
## FUNCTIONS FIRST - SCROLL DOWN FOR MAIN ##
############################################

# COMPUTING BLACK-BODY RADIATION IN W m-2 m-1
def blackbody(T,wl):
    # Define constants used in calculation
    h = 6.626E-34
    c = constants.c
    k = 1.38E-23
    c1 = 2.0 * np.pi * h * c * c
    c2 = h * c / k
    val = c2 / wl / T
    return c1 / (wl**5.0 * (np.exp(val) - 1.0))

## SCALING MODEL SPECTRUM FROM W m-2 m-1 to Fplanet / Fstar
def scale_model(fMod,wlen):
    flux = fMod.copy()
    Rp = 1.38           # Southworth et al. (2010)
    Rstar = 1.162       # Torres et al. (2008)
    RR = (Rp * 6.99E7 / (Rstar * 6.957E8) )**2.0
    Fstar = np.pi*blackbody(6065.0, wlen * 1E-9)
    flux /= Fstar
    flux *= RR
    return flux

### Just for plotting
def plot_matrix(wlen, mat, ph):
    no,nf,nx = mat.shape
    v0 = mat.min()
    v1 = mat.max()
    for io in range(no):
        plt.figure(figsize=(15,3))
        lims = [wlen[io,0]/1E3,wlen[io,-1]/1E3,ph[0],ph[-1]]
        plt.imshow(mat[io,], origin='lower', extent=lims, aspect='auto', vmin=v0, vmax=v1)
        plt.xlabel('Wavelength ($\mu$m)')
        plt.ylabel('Orbital phase')
        plt.show()
    return 0

## CREATING FAKE SPECTRAL SEQUENCE from following input parameters:
# - wMod: model wavelength in nm
# - fMod: model fluxes in W m-2 m-1
# - scale: scaling parameter to control the strength of the injected model
# - xJitter: jitter in alignment (pixels, default=0.01)
# - aJitter: jitter in airmass dependence of telluric lines (relative to 1)
# - subFolder: one of the folders containing the partial night of observations

def create_planet_sequence(wMod, fMod1, ph, rvel, Kp):
    wlen = np.load('wlen_regr.npy')
    spec = np.load('algn.npy')
    no, nf, nx = spec.shape
    Vpl = rvel + Kp * np.sin(2.0 * np.pi * ph)
    cs = interpolate.splrep(wMod, fMod1, s=0)
    spcFake = np.ones((no,nf,nx))
    for io in range(no):
        for j in range(nf):
            wMod1 = wlen[io,] * (1.0 - Vpl[j]/2.998E5)
            spcFake[io,j,] += interpolate.splev(wMod1,cs,der=0)
    np.save('output/ffake.npy', spcFake)
    return wlen, spcFake

def ally_extra_jitter(spcFake,xJitter):
    no, nf, nx = spcFake.shape
    xx = np.range(nx)
    for io in range(no):
        dx = np.random.normal(0.0, xJitter, nf)
        for j in range(nf):
            xx1 = xx + dx[j]
            cs1 = interpolate.splrep(xx,spcFake[io,j,].flatten(),s=0.0)
            spcFake[io,j,] = interpolate.splev(xx1,cs1,der=0)
    return spcFake
        
def add_noise(spcFake):
    spec = np.load('algn.npy')
    expTime = 150.0 * 2.0       # Twice the actual exposure time
    detGain = np.array([6.97, 6.89, 6.92, 7.27])
    no, nf, nx = spec.shape     # Getting dimensions of arrays
    spcNorm = spec.copy()               # Median-normalised spec
    cnt = np.zeros((no,nf))             # Median counts per spec    
    for io in range(no):
        spcNorm[io,] = expTime * detGain[io] * spec[io,]    # Counts -> e-
        for j in range(nf):
            # Populate median count vector with median of brightest 300 pixels
            spcTemp = spcNorm[io,j,]
            isort = np.argsort(spcTemp)
            cnt[io,j] = np.nanmedian(spcTemp[isort[700:-10]])    
            spcFake[io,j,] *= cnt[io,j]
    # Adding Gaussian noise
    phNoiseAmp = np.sqrt(spcFake)
    noise = np.random.normal(0.0, 1.0, size=(no,nf,nx))
    noise *= phNoiseAmp
    spcFake += noise
    for io in range(no): spcFake[io,] /= (expTime * detGain[io])
    np.save('output/ffake_with_noise.npy',spcFake)
    return spcFake

# Version 1 uses ESO Skycalc to pre-compute the Earth's transmission spectrum as
# function of airmass and PWV. The model is ran in the sub-folder skymod/ and 
# output as 'atm_trans.fits'

## TELLURIC REMOVAL ##
# This part is done in the tutorial directly to show people how things work
# Here there is a faster version working with matrices
    
# Version 1 - Airmass decorrelation, linear space
def rm_tell_airmass():
    spec = np.load('output/ffake_with_noise.npy')
    no, nf, nx = spec.shape
    tsub = spec.copy()
    # Load and prepare the airmass matrix for fast polynomial fitting
    air = np.load('air.npy')
    air = np.exp(air)
    airMat = np.ones((3,nf))
    airMat[1,] = air 
    airMat[0,] = air**2
    for io in range(no):
        spc = spec[io,].copy()
        for j in range(nf):
            isort = np.argsort(spc[j,])
            medCnt = np.nanmedian(spc[j,isort[700:1000]])
            spc[j,] /= medCnt
        pfit_coef = np.polyfit(air,spc,2)
        airFit = airMat.T @ pfit_coef
        spc /= airFit
        for j in range(nf): spc[j,] -= np.nanmean(spc[j,])  # /!\ Mean-subtracted spectra
        tsub[io,] = spc
    np.save('output/tsub.npy', tsub)
    return 1


## Cross correlation code
def xcorr(fVec, gVec, N):
    # Initialise identity matrix for fast computation
    I = np.ones(N)
    # The data (fVec) is already mean-subtracted but the model (gVec) is not
    gVec -= (gVec @ I) / N
    # Compute variances of model and data (will skip dividing by N because it
    # will cancels out when computing the cross-covariance)
    sf2 = (fVec @ fVec)
    sg2 = (gVec @ gVec)
    R = (fVec @ gVec)            # Data-model cross-covariance
    return R / np.sqrt(sf2*sg2)  # Data-model cross-correlation


# Code for the t-test
def do_ttest(din, dout):
    # Getting dimensions
    din = din.flatten()
    dout = dout.flatten()
    nin = len(din)
    nout = len(dout)
    # Computing means and variances
    mOut = np.mean(dout)
    mIn = np.mean(din)
    sOut = ((dout[:] - mOut)**2.0).sum() / (nout-1.0)
    sIn = ((din[:] - mIn)**2.0).sum() / (nin-1.0)
    # Computing the t statistic and degrees of freedom
    t = (mOut-mIn) / np.sqrt(sOut/nout + sIn/nin)
    nu = (sOut/nout + sIn/nin)**2.0 / ( ((sOut/nout)**2.0 / (nout-1)) + 
          ((sIn/nin)**2.0 / (nin-1)) )
    p2t = stats.t.cdf(t,nu)
    return p2t, stats.norm.ppf(p2t)


#### MCMC code
    
# Priors definition
def lnprior(theta):
    K, V = theta
    if 105.0 < K < 185.0 and -60.0 < V < 60.0:
        return 0.0
    return -np.inf

# Merging priors with the log-L calculation
def lnprob(theta, wMod, fMod, wData, fData, rvel, ph):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    logL = get_logL(theta, wMod, fMod, wData, fData, rvel, ph)
    return lp + logL

def process_model(spec, air, tell):
    nf = len(air)
    airMat = np.ones((3,nf))
    airMat[1,] = air 
    airMat[0,] = air**2
    spec1 = spec.copy() * tell
    pfit_coef = np.polyfit(air,spec1,2)
    airFit = airMat.T @ pfit_coef
    spec1 /= airFit
    spec1 -= 1.0
    stdVec = np.std(spec1,axis=0)
    nx = len(stdVec)
    for i in range(nx): spec1[:,i] /= stdVec[i]
    return spec1
          
    
# Actual CCF to logL mapping
def get_logL(pars, wMod, fMod, wObs, F, RV, ph):
    air = np.exp(np.load('air.npy'))
    tell = np.exp(np.load('atm_trans.npy'))
    fMod1 = scale_model(fMod*np.pi*7, wMod)
    K, V = pars
    # Initialising log-likelihood
    logLZ = 0
    # Scaling the tested model spectrum and pre-computing spline fit
    cs = interpolate.splrep(wMod,fMod1,s=0.0)
    # Looping over nights
    RVs = RV + V + K * np.sin(2.0 * np.pi * ph)
    dl_l = (1.0 - RVs / 2.9978E5)
    # Fake data and wavelengths
    no, nf, nx = F.shape
    # Additional vectors
    I = np.ones(nx)                     # Identity vector for fast computing
    tempMod = np.ones((nf,nx))
    for io in range(no):
        wSub = wObs[io,].copy()
        for j in range(nf):
            wShift = wSub * dl_l[j]
            tempMod[j,] += interpolate.splev(wShift, cs, der=0)
        #outMod = process_model(tempMod, air, tell[io,])
        for j in range(nf):
            gVec = tempMod[j,].copy()
            gVec -= (gVec @ I) / nx        # Mean-subtracted, shifted model
            sg2 = (gVec @ gVec) / float(nx)
            fVec = F[io,j,].copy()         # Already mean-subtracted
            sf2 = (fVec @ fVec)/float(nx)  # Data variance
            R = (fVec @ gVec) / float(nx)  # Data-model cross-covariance
            CC = R / np.sqrt(sf2*sg2)      # Data-model cross-correlation
            # Updating the log-likelihoods
            logLZ += (-0.5*nx * np.log (1.0 - CC**2.0))
    return logLZ

def run_mcmc(kp0, vr0, wMod, fMod, wData, fData, rvel, ph):
    initPos = (kp0, vr0)
    initDeltas = (5.0, 1.5)
    nPar = len(initPos)
    nWalkers = 10
    chainLen = 500
    nBurn = 50
    # Stuff for progress bar
    barW = 30
    n = 0
    sys.stdout.write("\r[{0}{1}]".format('#'*n, ' '*(barW-n)))
    tstart = time.time()
    # Initialisation of MCMC
    pos = [initPos + initDeltas*np.random.randn(nPar) for i in range(nWalkers)]
    sampler = emcee.EnsembleSampler(nWalkers, nPar, lnprob, args=(wMod, fMod, wData, fData, rvel, ph))
    
    for iRun in range(25):
        pos, prob, state = sampler.run_mcmc(pos, chainLen//25)
        n = int((barW+1)*float(iRun+1)/25)
        sys.stdout.write("\r[{0}{1}]".format('#'*n, ' '*(barW-n)))
    tstop = time.time()
    sys.stdout.write('\n')
    print(tstop - tstart, 's elapsed')
    samples = sampler.flatchain[nBurn:,]
    np.save('output/samples.npy',samples)
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
    outname = "emcee_results.png"
    fig = corner.corner(samples, labels=["$K_\mathrm{P}$", "$V_\mathrm{rest}$", "Scaling factor"],
                      truths=[kp0, vr0])
    fig.savefig(outname)
    plt.show()

    

