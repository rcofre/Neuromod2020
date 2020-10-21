# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 15:38:25 2020

@author: Carlos Coronel

Some auxiliary functions.

[1] Lancaster, G., Iatsenko, D., Pidde, A., Ticcinelli, V., & Stefanovska, A. 
(2018). Surrogate data for hypothesis testing of physical systems. 
Physics Reports, 748, 1-60.
"""

import numpy as np
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests


def get_uptri(x):
    """
    Gets the vectorized upper triangle of a matrix x.
    
    Parameters
    ----------
    x : numpy array.
        connectivity matrix.
    Returns
    -------
    vector : numpy array.
             upper triangle of x in vector form.
    """
    nnodes = x.shape[0]
    npairs = (nnodes**2 - nnodes) // 2
    vector = np.zeros(npairs)
    
    idx = 0
    for row in range(0, nnodes - 1):
        for col in range(row + 1, nnodes):
            vector[idx] = x[row, col]
            idx = idx + 1
    
    return(vector)


def matrix_recon(x):
    """
    Reconstructs a connectivity matrix from its upper triangle.
    
    Parameters
    ----------
    x : numpy array.
        upper triangle of the original matrix.
    Returns
    -------
    matrix : numpy array.
             original connectivity matrix.
    """
    npairs = len(x)
    nnodes = int((1 + np.sqrt(1 + 8 * npairs)) // 2)
    
    matrix = np.zeros((nnodes, nnodes))
    idx = 0
    for row in range(0, nnodes - 1):
        for col in range(row + 1, nnodes):
            matrix[row, col] = x[idx]
            idx = idx + 1
    matrix = matrix + matrix.T
   
    return(matrix)   
    


def probabilistic_thresholding(y, surr_number = 50, alpha = 0.05):
    """
    Threshold for neglecting spurious connectivity values in functional fata.
    It uses a phase randomization to destroy the pairwise correlations, while 
    preserving the spectral properties of the original signals. Type I error
    (multiple comparisons) corrected by the Benjamini-Hochberg procedure.
    
    Parameters
    ----------
    y : txN numpy array.
        time series.
        t -> time.
        N -> nodes.
    surr_number : integer.
                  number of surrogates.
    alpha : float (between 0 and 1).
            critical p_value for statistical inference. By default is 0.05.
    Returns
    -------
    FC_adjusted : NxN numpy array.
                  thresholded functional connectivity matrix.
    p_matrix : NxN numpy array.
               p_values for each connectivity pair.
    """

    y = y - np.mean(y, axis = 0)
    yf = np.fft.fft(y, axis = 0)

    nnodes = y.shape[1]
    npairs = (nnodes**2 - nnodes) // 2
    matrix_surr = np.zeros((npairs, surr_number))

    FC_real = np.corrcoef(y.T)
    FC_real = FC_real[np.triu_indices(n = nnodes, k = 1)]
    
    for i in range(surr_number):
        np.random.seed(i + 1)
        random_vector = np.random.uniform(0, 2 * np.pi, ((yf.shape[0] // 2), yf.shape[1]))
        random_vector = np.row_stack((random_vector, random_vector[::-1,:]))
        yfR = yf * np.exp(1j * random_vector)
        surrogate = np.fft.ifft(yfR, axis = 0)
        surrogate = surrogate.real
        FC_surr = np.corrcoef(surrogate.T)
        matrix_surr[:,i] = FC_surr[np.triu_indices(n = nnodes, k = 1)]

    p_vector = np.zeros(npairs)
    for i in range(npairs):
        mu, sigma = stats.norm.fit(matrix_surr[i,:])
        p_vector[i] = 1 - stats.norm.cdf(FC_real[i], mu, sigma)
    
    
    reject, p_adjust, alphacSidak, alphacBonf = multipletests(p_vector, alpha = alpha, method='fdr_bh')
    
    FC_adjusted = FC_real * ((p_adjust < alpha) * 1)
    FC_adjusted = matrix_recon(FC_adjusted)
    p_matrix = matrix_recon(p_adjust)
    
    return([FC_adjusted, p_matrix])
 
    
#Kuramoto order parameter
def simple_order_parameter(data):
    """
    Calculates the global phase synchronization over time as the Kuramoto order
    parameter.
    
    Parameters
    ----------
    data : phixN numpy array.
           phi -> phases.
           N -> nodes.
    Returns
    -------
    PhaseSynch : 1xphi numpy array.
                 Kuramoto order parameter.
    """
    phases = data
    PhaseSynch = np.abs(np.mean(np.exp(1j * phases), axis = 1))
    return(PhaseSynch)    
    
    