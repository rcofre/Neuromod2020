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
import BOLDModel as BD
import JansenRitModel as JR
import Utils
import FCD
import matplotlib.pyplot as plt
import importlib
importlib.reload(JR)

#Simulation parameters
JR.dt = 1E-3 #Integration step
JR.teq = 60 #Simulation time for stabilizing the system
JR.tmax = 600 #Simulation time
treg = 20 #Time window length (in seconds) for computing the regularity index
JR.downsamp = 10 #Downsampling to reduce the number of points        
Neq = int(JR.teq / JR.dt / JR.downsamp) 
Nmax = int(JR.tmax / JR.dt / JR.downsamp)
Nreg = Neq + int(treg / JR.dt / JR.downsamp)
Ntotal = Neq + Nmax #Total number of points
ttotal = JR.teq + JR.tmax #Total simulation time
seed = 0

#Node parameters
JR.sigma = 2.25
JR.alpha = 0
JR.beta = 0
JR.r0 = 0.56
JR.p = 2

#Network parameters
JR.nnodes = 90 #number of nodes
nnodes = JR.nnodes
JR.std_speed = 7.76
JR.mean_speed = 54

#np.random.seed(seed)
JR.seed = seed
init1 = time.time()
y, time_vector, z = JR.Sim(verbose = True)
pyrm = JR.C2 * y[:,1] - JR.C4 * y[:,2] + JR.C * JR.alpha * z #EEG-like output of the model
end1 = time.time()

print([end1 - init1,np.mean(pyrm)])


#%%
#Plot EEG-like signals

plt.figure(1)
plt.clf()
plt.plot(time_vector[Neq:], pyrm[Neq:,:])
plt.tight_layout()


#%%
#This part calculates several measures over the EEG-like signals: global phase synchronization,
#frequency of oscillation, signal to noise ratio, and regularity.

init2 = time.time()

#Welch method to stimate power spectal density (PSD)
#Remember: dt = original integration step, dws = downsampling           
window_length = 20 #in seconds
PSD_window = int(window_length / JR.dt / JR.downsamp) #Welch time window
PSD = signal.welch(pyrm[Neq:,:] - np.mean(pyrm[Neq:,:], axis = 0), fs = 1 / JR.dt / JR.downsamp, 
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
SNRs = SNR.calculateSNR(PSD = np.copy(PSD_curves), freq_vector = freq_vector, 
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
a0, b0 = signal.bessel(3, [2 * JR.dt * Fmin * JR.downsamp, 
                           2 * JR.dt * Fmax * JR.downsamp], btype = 'bandpass')
Vfilt = signal.filtfilt(a0, b0, pyrm, axis = 0)
     
#Synchronization
phases_signal = np.angle(signal.hilbert(Vfilt[Neq:,:], axis = 0)) #Phases
phaseSynch = Utils.simple_order_parameter(phases_signal) #Kuramoto order parameter
meanSynch = np.mean(phaseSynch) #Averaged Kuramoto order parameter
varSynch = np.var(phaseSynch) #Variance of the Kuramoto order parameter (Metastability)

#Computing the regularity index
Nreg_windows = int(JR.tmax // treg) #Number of time windows for computing the Reg index
Reg = []
for i in range(0,nnodes):
    Reg_t = []
    for j in range(0,Nreg_windows):
        idx1 = int(Neq + j * treg / JR.dt / JR.downsamp)
        idx2 = int(Neq + (j+1) * treg / JR.dt / JR.downsamp)
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
plt.plot(freq_vector[1:-2], 10 * np.log10 (PSD_curves[1:-2,:]))
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

if np.any(JR.r0 == 0):
    rE = JR.s(pyrm, JR.r0 + 1E-4)
else:
    rE = JR.s(pyrm, JR.r0)

BOLD_signals = BD.Sim(rE, nnodes, JR.dt * JR.downsamp)
BOLD_signals = BOLD_signals[Neq:,:]

BOLD_downsamp = 100
BOLD_dt = JR.dt * JR.downsamp * BOLD_downsamp
BOLD_signals = BOLD_signals[::BOLD_downsamp,:]

#Filter the BOLD-like signal between 0.01 and 0.1 Hz
Fmin, Fmax = 0.01, 0.1
a0, b0 = signal.bessel(3, [2 * BOLD_dt * Fmin, 2 * BOLD_dt * Fmax], btype = 'bandpass')
BOLDfilt = signal.filtfilt(a0, b0, BOLD_signals[:,:], axis = 0)

#Surrogate thresholding
#sFC: static Functional Connectivity (sFC) matrix
sFC_BOLD = Utils.probabilistic_thresholding(BOLDfilt, 500, 0.05)[0]

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

FCD_matrix, L_points, steps = FCD.extract_FCD(BOLDfilt, L = L, mode = mode, dt = BOLD_dt, steps = 2)
FCD_matrix *= np.sqrt(2)

#Calculate the typical FCD speed (dtyp) and FCD variance (varFCD, multistability)
dtyp, varFCD = FCD.FCD_vars(FCD_matrix, L_points, steps, bins = 20, vmin = 0, vmax = 1)

#Plot the FCD
plt.figure(7)
plt.clf()
plt.imshow(FCD_matrix, vmin = 0, vmax = 1, cmap = 'jet')

end4 = time.time()

print([end4 - init4])


