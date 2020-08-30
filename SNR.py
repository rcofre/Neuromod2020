#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:55:35 2020

@author: Carlos Coronel

The function computes the signal to noise ratio (SNR) of a time series using its
Power Spectral Density (PSD) function. 

[1] Schultz, S. R. (2007). Signal-to-noise ratio in neuroscience. 
Scholarpedia, 2(6), 2046.
"""

import numpy as np
from scipy.integrate import simps

def calculateSNR(PSD, freq_vector, freqs, nsig, n_harm):
    """
    Estimates the signal-to-noise ratio, 
    given the in-band bins of a PSD function and 
    the location of the input signal.
    Each increment in nsig adds a bin to either side.
    The SNR is expressed in dB. The function excludes the
    first n_harm harmonics (except the fundamental).
    
    Parameters
    ----------
    PSD :   wxN numpy array.
            Windowed PSD function obtained, for example, with the Welch method.
            w -> frequency range.
            N -> nodes.
    freq_vector : numpy array. 
                  vector of frequencies related to the PSD functions.
    freqs : numpy array.
            each value contains the desired frequency for each node, that is,
            the center values within the window for calculating the signal power.
    nsig : integer > 0.
           length (in frequency bins) of the signal power window centered at the
           desired frequency. Any increase in nsig add a bin to either side:
           nsig = 1 -> window of 3 points.
           nsig = 2 -> window of 5 points.
           nsig = K -> window of 2 * K + 1 points.
     nharm : integer >= 0.
             number of harmonics to supress (except the fundamental).
    Returns
    -------
    snr : numpy array.
          signal to noise ratio for each node.           
    """
    
    freq_point = np.diff(freq_vector)[0] #frequency resolution
    snr = np.zeros(PSD.shape[1])

    for i in range(0, PSD.shape[1]):
        #This part set to zero the harmonics' power
        freqs_harm = freqs[i] * np.arange(2, n_harm + 1, 1)
        points_harm = np.ndarray.astype(freqs_harm / freq_point, int)
        PSD[points_harm-1,i] = 0
        PSD[points_harm,i] = 0
        PSD[points_harm+1,i] = 0
        
        f = np.argmin(np.abs(freqs[i] - freq_vector))
        
        #Bins of the signal (within the window)
        signalBins = np.arange(f - nsig, f + nsig + 1, 1)
        signalBins = signalBins[signalBins > 0]
        signalBins = signalBins[signalBins <= len(PSD[:,i])]
        #Bins of the noise (outside the window)
        noiseBins = np.arange(0, len(PSD[:,i]), 1)
        noiseBins = np.delete(noiseBins, signalBins)
        
        #Computing signal and noise power as the area under the curve
        s = simps(PSD[signalBins,i], dx = freq_point)
        n = simps(PSD[noiseBins,i], dx = freq_point)
        
        #Computing signal to noise ratio in dB
        if n == 0:
            snr[i] = np.inf
        else:
            snr[i] = 20 * np.log10(s / n)
        
    return(snr)




