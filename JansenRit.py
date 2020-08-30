# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:57:25 2018

@author: Carlos Coronel

Modified version of the Jansen and Rit Neural Mass Model [1]. We included an extra
local connection from inhibitory interneurons to excitatory interneurons [2,3], scaled by
a connectivity constant 'beta'. Long-range connections are only excitatory (pyramidal to
pyramidal). The script runs the model for generate EEG-like and BOLD-like signals.

[1] Jansen, B. H., & Rit, V. G. (1995). Electroencephalogram and visual evoked 
potential generation in a mathematical model of coupled cortical columns. 
Biological cybernetics, 73(4), 357-366.

[2] Silberberg, G., & Markram, H. (2007). Disynaptic inhibition between neocortical
pyramidal cells mediated by Martinotti cells. Neuron, 53(5), 735-746.

[3] Fino, E., Packer, A. M., & Yuste, R. (2013). The logic of inhibitory 
connectivity in the neocortex. The Neuroscientist, 19(3), 228-237. 
"""

import numpy as np
from scipy import signal
import time
import SNR
import Regularity
import BOLD
import graph_utils
import FCD
import matplotlib.pyplot as plt

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


init1 = time.time()

#Simulation parameters
dt = 1E-3 #Integration step
teq = 60 #Simulation time for stabilizing the system
tmax = 600 #Simulation time
treg = 20 #Time window length (in seconds) for computing the regularity index
dws = 10 #Downsampling to reduce the number of points        
Neq = int(teq / dt / dws) 
Nmax = int(tmax/dt / dws)
Nreg = Neq + int(treg / dt / dws)
Ntotal = Neq + Nmax #Total number of points
ttotal = teq + tmax #Total simulation time

nnodes = 90 #number of nodes
D = np.loadtxt('speed_matrix_AAL.txt') #Speed constants matrix for long-range connections

seed = 0 #Random seed
np.random.seed(seed) #Set the random state

a = 100 #Velocity constant for EPSPs (1/sec)
b = 50 #Velocity constasnt for IPSPs (1/sec)

p = 2 #Basal input to pyramidal population

sigma = 2.25 * np.sqrt(dt) #Input standard deviation scaled by np.sqrt(dt)

C = 135 #Global synaptic connectivity
C1 = C * 1 #Connectivity between pyramidal pop. and excitatory pop.
C2 = C * 0.8 #Connectivity between excitatory pop. and pyramidal pop.
C3 = C * 0.25 #Connectivity between pyramidal pop. and inhibitory pop.
C4 = C * 0.25 #Connectivity between inhibitory pop. and pyramidal pop.

A = 3.25 #Amplitude of EPSPs
B = 22 #Amplitude of IPSPs

#Both as multiples of C
alpha = 0  #Long-range pyramidal-pyramidal coupling
beta = 0 #Connectivity between inhibitory pop. and excitatory interneuron pop. (short-range)

#Sigmoid function parameters
e0 = 2.5 #Half of the maximum firing rate
v0 = 6 #V1/2
r0, r1, r2 = 0.56, 0.56, 0.56 #Slopes of sigmoid functions

def sde_euler_maruyama(f1, f2, f3, f4, y0, y02, y03, tf, h, dws = 1, params=()):
    """
    Solve the SDE system using the Euler-Maruyama method. 
    This function was adapted specifically for the Jansen and Rit model.
    Parameters
    ----------
    f1 : function.
         Function describing the deterministic part of the SDE.
    f2, f3 : functions.
             Auxiliary functions for the time-delays.
    y0, y02, y03 : numpy array.
                   Initial conditions.
    y4 : function.
         Function describing the stochastic part of the SDE.
    tf : float.
         Final time.
    h  : float.
         Time step.
    dws : integer.
         downsampling of the results.
    params : tuple, optional
             Optional list of parameters to be passed to `f`.
    Returns
    -------
    y : numpy array
        Solution to the SDE.
    t : numpy array.
        Time vector.
    z : numpy array.
        Total intercolumn inputs.     
    """
    
    def dW(delta_t, nnodes): 
        """Sample a random number at each call."""
        return(np.random.normal(0.0, np.sqrt(delta_t), nnodes))

    #Time vector
    ts = np.linspace(0, tf, int(tf / h))
    #Downsampled time vector
    ts_dws = np.linspace(0, tf, int(tf / h / dws))
    
    row = y0.shape[0] #Number of variables of the Jansen & Rit model
    col = y0.shape[1] #Number of nodes
    y_temp = y0 #Temporal vector to update values
    y = np.zeros((ts_dws.size, row, col)) #Matrix to store values
    z = np.zeros((ts_dws.size, col)) #Matrix to store z values
    y[0,:,:] = y0 #First set of initial conditions
    y_delay = y02 #Second set of inital conditions
    y_aux = y03 #Third set of initial conditions
    for t, i in zip(ts[1:], range(ts.size - 1)):
        y_temp[4,:] +=  dW(h,col) * f4(t, *params)
        evaluate = f1(y_temp, y_delay, t, *params)
        y_temp += h * evaluate[0] 
        z_temp = evaluate[1]
        y_delay += h * f2(y_aux, t)
        y_aux += h * f3(y_temp[[1,2],:], y_delay, y_aux, t)
        #This line is for store values each dws points
        if ((i+1) % dws) == 0:
            y[(i+1)//dws,:,:] = y_temp
            z[(i+1)//dws,:] = z_temp
        if ((i+1) % (10 / h)) == 0:
            print(t) #this is for impatient people
        
    return(y, ts_dws, z)
 
    
#Sigmoid function
def s(v, e0 = e0, r = 0.56, v0 = v0):
    return (2 * e0) / (1 + np.exp(r * (v0 - v)))

#Jansen & Rit multicolumn model
def f1(y, y_delay, t):
    x0, x1, x2, y0, y1, y2 = y
    x3 = y_delay

    z = np.sum(M / norm * x3, axis = 0)
    
    x0_dot = y0
    y0_dot = A * a * (s(C2 * x1 - C4 * x2 + C * alpha * z, r = r0)) - \
             2 * a * y0 - a**2 * x0 
    x1_dot = y1
    y1_dot = A * a * (p + s(C1 * x0 - C * beta * x2, r = r1)) - \
             2 * a * y1 - a**2 * x1
    x2_dot = y2
    y2_dot = B * b * (s(C3 * x0, r = r2)) - \
             2 * b * y2 - b**2 * x2

    return([np.array((x0_dot, x1_dot, x2_dot, y0_dot, y1_dot, y2_dot)),z])

#This function takes the trasmision speed (for long-range outputs)
#and returns the output increment.
def f2(y_aux, t):
    y3 = y_aux
    
    x3_dot = y3
    
    return(np.array((x3_dot)))

#This function receives the instantaneous outputs of cortical columns
#(C2 * x2 - C1 * x1 + C * alpha * z), the delayed outputs between all
#cortical columns (x3), and the speed of transmission between all the columns
#(y3). It returns the trasmision speed increment.
def f3(y_inputs, y_delay, y_aux, t):
    x1, x2 = y_inputs
    x3 = y_delay
    y3 = y_aux
    z = np.sum(M / norm * x3, axis = 0)
       
    inputs_exc = s(C2 * x1 - C4 * x2 + C * alpha * z, r = r0)
    
    y3_dot = A * D * inputs_exc[:,None] - 2 * D * y3 - D**2 * x3
    
    return(np.array((y3_dot)))

#This is the stochastic part of the system
def f4(t):

    y1_dot = A * a * sigma

    return(np.array((y1_dot)))

 
#NETWORKS
#M: Structural connectivity matrix

#Real Human Connectivity Matrix
M = np.loadtxt('structural_Deco_AAL.txt')


#Normalization factor
norm = np.sqrt(np.sum(M + 1E-6, axis = 1) * np.sum(M + 1E-6, axis = 0))   
#norm = 1 

#Initial conditions
ic = np.ones((1, nnodes)) * np.array([0.131,  0.171, 0.343,
                                      3.07, 2.96,  25.36])[:, None] 
ic2 = np.ones((nnodes, nnodes)) * 0.2
ic3 = np.ones((nnodes, nnodes)) * 3.5
#Randomizing initial conditions
ic *= np.random.uniform(0.01, 2, ((6, nnodes)))
ic2 *= np.random.uniform(0.9, 1, ((nnodes, nnodes)))
ic3 *= np.random.uniform(0.9, 1, ((nnodes, nnodes)))
      
#Solve the SDE using Euler-Maruyama
res, t, z = sde_euler_maruyama(f1, f2, f3, f4, ic, ic2, ic3, ttotal, dt, dws) #Integration
pyrm = C2 * res[:,1] - C4 * res[:,2] + C * alpha * z #EEG-like output of the model
t2 = t[Neq:] - teq


end1 = time.time()

print([end1 - init1])


#%%
#Plot EEG-like signals

plt.figure(1)
plt.clf()
plt.plot(t2, pyrm[Neq:,:])
plt.tight_layout()


#%%
#This part calculates several measures over the EEG-like signals: global phase synchronization,
#frequency of oscillation, signal to noise ratio, and regularity.

init2 = time.time()

#Welch method to stimate power spectal density (PSD)
#Remember: dt = original integration step, dws = downsampling           
window_length = 20 #in seconds
PSD_window = int(window_length / dt / dws) #Welch time window
PSD = signal.welch(pyrm[Neq:,:] - np.mean(pyrm[Neq:,:], axis = 0), fs = 1 / dt / dws, 
                   nperseg = PSD_window, noverlap = PSD_window // 2, 
                   scaling = 'density', axis = 0)
freq_vector = PSD[0] #Frequency values
PSD_curves =  PSD[1] #PSD curves for each node
freq_min_point = int(1 / np.diff(freq_vector)[0]) #Frequency steps (in points)

#Position (pos) of the frequency (freqs) with max power, for freqs > 1
pos = np.argmax(PSD_curves[freq_min_point:,:], axis = 0)
freqs = freq_vector[freq_min_point:][pos]
     
#Mean and variance of the frequency for all the oscillators
Mfreq = np.mean(freqs)
Varfreq = np.var(freqs)

#Signal to noise ratio (SNR), discarting n_harm first harmonics
SNRs = SNR.calculateSNR(PSD = PSD_curves, freq_vector = freq_vector, 
                        freqs = freqs, nsig = freq_min_point * 2, 
                        n_harm = 4)
SNR_val = np.mean(SNRs) #Averaged signal to noise ratio

#This is for avoiding negative values of the minimum frequency of the filter 
if Mfreq <= 3.5:
    Mfilt = 3.5
else:
    Mfilt = Mfreq

#Filtering signals
Fmin, Fmax = Mfilt - 3, Mfilt + 3
a0, b0 = signal.bessel(3, [2 * dt * Fmin * dws, 2 * dt * Fmax * dws], btype = 'bandpass')
Vfilt = signal.filtfilt(a0, b0, pyrm, axis = 0)
     
#Synchronization
phases_signal = np.angle(signal.hilbert(Vfilt[Neq:,:], axis = 0)) #Phases
phaseSynch = simple_order_parameter(phases_signal) #Kuramoto order parameter
meanSynch = np.mean(phaseSynch) #Averaged Kuramoto order parameter
varSynch = np.var(phaseSynch) #Variance of the Kuramoto order parameter (Metastability)

#Computing the regularity index
Nreg_windows = int(tmax // treg) #Number of time windows for computing the Reg index
Reg = []
for i in range(0,nnodes):
    Reg_t = []
    for j in range(0,Nreg_windows):
        idx1, idx2 = int(Neq + j * treg / dt / dws), int(Neq + (j+1) * treg / dt / dws)
        Reg_t.append(Regularity.find_second_maxima(pyrm[idx1:idx2,i]))
    Reg_t = np.mean(Reg_t)
    Reg.append(Reg_t)
Reg = np.mean(Reg) #Average Reg index


end2 = time.time()

print([end2 - init2])


#%%

#Power spectral density functions
plt.figure(3)
plt.clf()
plt.plot(freq_vector, PSD_curves)
plt.tight_layout()

#Kuramoto order parameter
plt.figure(4)
plt.clf()
plt.plot(phaseSynch)
plt.ylim(0,1)
plt.tight_layout()

    
#%%
#fMRI-BOLD response
init3 = time.time()

if r0 == 0:
    rE = s(pyrm, r = 0.00001)
else:
    rE = s(pyrm, r = r0)

ic_BOLD = np.ones((1, nnodes)) * np.array([0.1, 1, 1, 1])[:, None] #initial conditions
BOLD_vars = np.zeros((Ntotal,4,nnodes))
BOLD_vars[0,:,:] = ic_BOLD
  
#Solve the ODEs with Euler
for i in range(1,Ntotal):
    BOLD_vars[i,:,:] = BOLD_vars[i - 1,:,:] + dt * dws * BOLD.deoxyhemoglobin(BOLD_vars[i - 1,:,:], rE[i - 1,:], i - 1)
    
#Get the BOLD-like signal using deoxyhemoglobin content (q) and blood volumen (v)
BOLD_signals = BOLD.BOLD(BOLD_vars[:,3,:], BOLD_vars[:,2,:])   
BOLD_signals = BOLD_signals[Neq:,:]
dws_BOLD = 100 #Second downsampling for the BOLD signals
#Remember that the total downsampling fot the BOLD signals is:
#real_dws = dws * dws_BOLD
real_dws = dws * dws_BOLD
BOLD_signals = BOLD_signals[::dws_BOLD,:]

#Filter the BOLD-like signal between 0.01 and 0.1 Hz
Fmin, Fmax = 0.01, 0.1
a_filt, b_filt = signal.bessel(3, [2 * dt * Fmin *  real_dws, 2 * dt * Fmax *  real_dws], btype = 'bandpass')
BOLDfilt = signal.filtfilt(a_filt, b_filt, BOLD_signals[:,:], axis = 0)

#Surrogate thresholding
#sFC: static Functional Connectivity (sFC) matrix
sFC_BOLD = graph_utils.probabilistic_thresholding(BOLDfilt, 500, 0.05)[0]

end3 = time.time()

print([end3 - init3])


#%%

#Filtered BOLD-like signals
plt.figure(5)
plt.clf()
plt.plot(BOLDfilt)
plt.tight_layout()

#sFC matrix
plt.figure(6)
plt.clf()
plt.imshow(sFC_BOLD, cmap = 'RdBu', vmin = -1, vmax = 1)
plt.tight_layout()

#%%

#Functional Connectivity Dynamics (FCD)

init4 = time.time()

L = 100 #Window length (in seconds)
mode = 2 #Mode of the FCD. 1: Pearson, 2: Clarkson
BOLDdt = dt * real_dws

FCD_matrix, L_points, steps = FCD.extract_FCD(BOLDfilt, L = L, mode = mode, dt = BOLDdt, steps = 2)

#Calculate the typical FCD speed (dtyp) and FCD variance (varFCD, multistability)
dtyp, varFCD = FCD.FCD_vars(FCD_matrix, L_points, steps, bins = 20, vmin = 0, vmax = 1)

#Plot the FCD
plt.figure(7)
plt.clf()
plt.imshow(FCD_matrix, vmin = 0, vmax = 1, cmap = 'jet')

end4 = time.time()

print([end4 - init4])



