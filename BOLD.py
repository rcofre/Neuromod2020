#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:18:02 2019

@author: Carlos Coronel

Generalized Hemodynamic Model to reproduce fMRI BOLD-like signals.

[1] Stephan, K. E., Weiskopf, N., Drysdale, P. M., Robinson, P. A., & Friston, K. J. 
(2007). Comparing hemodynamic models with DCM. Neuroimage, 38(3), 387-401.

[2] Deco, Gustavo, et al. "Whole-brain multimodal neuroimaging model using serotonin 
receptor maps explains non-linear functional effects of LSD." Current Biology 
28.19 (2018): 3065-3074.
"""

import numpy as np

#PARAMETERS 
taus = 0.65 #time constant for signal decay
tauf = 0.41 #time constant for feedback regulation
tauo = 0.98 #time constant for volume and deoxyhemoglobin content change
#itauX = inverse of tauX
itaus = 1 / taus
itauf = 1 / tauf
itauo = 1 / tauo
nu = 40.3 #frequency offset at the outer surface of the magnetized
          #vessel for fully deoxygenated blood at 1.5 Tesla (s^-1)
r0 = 25 #slope of the relation between the intravascular relaxation 
        #rate and oxygen saturation (s^-1)         
alpha = 0.32 #resistance of the veins; stiffness constant
ialpha = 1 / alpha
epsilon = 0.5 #ratio of intra and extravascular signal

E0 = 0.4 #resting oxygen extraction fraction
TE = 0.04 #echo time (!!determined by the experiment)
V0 = 0.04 #resting venous blood volume fraction

#Kinetics constants
k1 = 4.3 * nu * E0 * TE
k2 = epsilon * r0 * E0 * TE
k3 = 1 - epsilon


def deoxyhemoglobin(y, rE, t):
    """
    This function generates BOLD-like signals using the firing rates rE.
    
    Parameters
    ----------
    y : numpy array.
        Contains the following variables:
        s: vasodilatory signal. If it increases, the blood vessel experiments
        vasodilatation.
        f: blood inflow. Increases with the vasodilatation.
        v: blood volumen. Increases with blood inflow.
        q: deoxyhemoglobin content.
    rE: numpy array.
        Firing rates of neural populations/neurons.
    t : float.
        Current simulation time point.
    Returns
    -------
    Numpy array with s, f, v and q derivatives at time t.
    """
    s, f, v, q = y
    
    s_dot = 1 * rE + 0 - itaus * s - itauf * (f - 1)
    f_dot = s
    v_dot = (f - v ** ialpha) * itauo
    q_dot = (f * (1 - (1 - E0) ** (1 / f)) / E0 - q * v ** ialpha / v) * itauo
    
    return(np.array((s_dot, f_dot, v_dot, q_dot)))
  
    
def BOLD(q, v):
    """
    This function returns the BOLD signal using deoxyhemoglobin content and
    blood volumen as inputs.
   
    Parameters
    ----------
    q: numpy array.
       deoxyhemoglobin content over time.
    v: numpy array.
       blood volumen over time.
    Returns
    -------
    Numpy array with BOLD-like signals.
    """
    return(V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v)))



