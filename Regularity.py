#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 17:27:09 2019

@author: Carlos Coronel

Functions to calculate the Regularity index as described by [1].

[1] Malagarriga, D., Villa, A. E., Garcia-Ojalvo, J., & Pons, A. J. (2015). 
Mesoscopic segregation of excitation and inhibition in a brain network model. 
PLoS Comput Biol, 11(2), e1004007.
"""

import numpy as np


def autocorr(x):
    """
    Computes the autocorrelation function of a time series x.
    
    Parameters
    ----------
    x : numpy array.
        time series.
    Returns
    -------
    corr : numpy array.
           autocorrelation function of x.
    """
    corr = np.correlate(x, x, mode = 'full')
    return(corr)


def find_second_maxima(y):
    """
    This function calculates the value of the second absolute maxima of the
    autocorrelation function y. That maxima corresponds to the Regularity index [1],
    a measure of signal periodicity -> 0 for noisy/chaotic signals, 1 for perfectly
    periodic signals (e.g., sines or cosines).
    
    Parameters
    ----------
    y : numpy array.
        autocorrelation function.
    Returns
    -------
    maxima : float.
           : Regularity index.
           
    """

    N = len(y)
    yc = autocorr(y - np.mean(y))
    yc = (yc / np.max(yc))[N:]
    
    idx = np.argwhere(np.diff((yc > 0) * 1) != 0)
    if len(idx) != 0:
            maxima = np.max(yc[idx[0][0]:])
    else:
        maxima = 0
    return(maxima)




