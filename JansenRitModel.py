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
from numba import jit,float64, vectorize
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


#Simulation parameters
dt = 1E-3 #Integration step
teq = 60 #Simulation time for stabilizing the system
tmax = 600 #Simulation time
downsamp = 10 #Downsampling to reduce the number of points        
seed = 0 #Random seed

#Networks parameters

#Structural connectivity

nnodes = 90 #number of nodes
M = np.loadtxt('structural_Deco_AAL.txt')

#Speed constants matrix for long-range connections
D = np.loadtxt('speed_matrix_AAL.txt') #Base matrix

#Normalization factor
norm = np.sqrt(np.sum(M, axis = 1) * np.sum(M, axis = 0))   

mean_speed = 54 #mean speed of D for M>0
std_speed = 7.76 #std of D for M>0

#Node parameters
a = 100 #Velocity constant for EPSPs (1/sec)
b = 50 #Velocity constasnt for IPSPs (1/sec)
p = 2 #Basal input to pyramidal population
sigma = 2.25 #Input standard deviation

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

#Initial conditions
ic = np.ones((1, nnodes)) * np.array([0.131,  0.171, 0.343,
                                      3.07, 2.96,  25.36])[:, None] 
ic2 = np.stack((np.ones((nnodes, nnodes)) * 0.2, np.ones((nnodes, nnodes)) * 3.5), 0)


@vectorize([float64(float64,float64)],nopython=True)
#Sigmoid function
def s(v,r0):
    return (2 * e0) / (1 + np.exp(r0 * (v0 - v)))


@jit(float64[:](float64[:,:],float64),nopython=True)
#Long-range inputs function
def Z(y_delay,t):
    x3 = y_delay
    return(np.sum(M / norm * x3, axis = 0))


@jit(float64[:,:](float64[:,:],float64[:],float64),nopython=True)
#Jansen & Rit multicolumn model (intra-columnar outputs)
def f1(y,z,t):
    x0, x1, x2, y0, y1, y2 = y

    noise = np.random.normal(0,sigma,nnodes)

    x0_dot = y0
    y0_dot = A * a * (s(C2 * x1 - C4 * x2 + C * alpha * z, r0)) - \
             2 * a * y0 - a**2 * x0 
    x1_dot = y1
    y1_dot = A * a * (p + noise + s(C1 * x0 - C * beta * x2, r1)) - \
             2 * a * y1 - a**2 * x1
    x2_dot = y2
    y2_dot = B * b * (s(C3 * x0, r2)) - \
             2 * b * y2 - b**2 * x2

    return(np.vstack((x0_dot, x1_dot, x2_dot, y0_dot, y1_dot, y2_dot)))

@jit(float64[:,:,:](float64[:,:],float64[:,:],float64[:,:],float64[:],float64),nopython=True)
#Set of equations for inter-columnar outputs
def f2(y_inputs, y_delay, y_aux, z, t):
    x1, x2 = y_inputs
    x3 = y_delay
    y3 = y_aux
    
    inputs_exc = s(C2 * x1 - C4 * x2 + C * alpha * z, r0)
     
    x3_dot = y3
    y3_dot = A * D * np.repeat(inputs_exc,nnodes).reshape((nnodes,nnodes)) - 2 * D * y3 - D**2 * x3   
    
    return(np.stack((x3_dot,y3_dot)))

#This function is just for setting the random seed
@jit(float64(float64),nopython=True)
def set_seed(seed):
    np.random.seed(seed)
    return(seed)


def Sim(verbose = True):
    """
    Run a network simulation with the current parameter values.
    
    Note that the time unit in this model is seconds.

    Parameters
    ----------
    verbose : Boolean, optional
        If True, some intermediate messages are shown.
        The default is False.

    Raises
    ------
    ValueError
        An error raises if the dimensions of M and the number of nodes
        do not match.

    Returns
    -------
    y : ndarray
        Time trajectory for the six variables of each node.
    time_vector : numpy array (vector)
        Values of time.
    z : numpy array (matrix)
        long-range inputs over time to each node.

    """
    global teq,tmax,ttotal,downsamp,M,D,seed
         
    #Changing properties of D matrix
#    D[M == 0] = 0
#    D[M > 0] = D[M > 0] *  std_speed / np.std(D[M > 0])
#    D[M > 0] = D[M > 0] + (mean_speed - np.mean(D[M > 0]))    

    if M.shape[0]!=M.shape[1] or M.shape[0]!=nnodes:
        raise ValueError("check M dimensions (",M.shape,") and number of nodes (",nnodes,")")
    
    if M.dtype is not np.dtype('float64'):
        try:
            M=M.astype(np.float64)
        except:
            raise TypeError("M must be of numeric type, preferred float")    
    
    
    ttotal = teq + tmax #Total simulation time
    Nsim = int(ttotal / dt) #Total simulation time points
    Neq = int(teq / dt / downsamp) #Number of points to discard
    Nmax = int(tmax/dt / downsamp) #Number of points of final simulated recordings
    Ntotal = Neq + Nmax #Total number of points of total simulated recordings
    
    #Time vector
    time_vector = np.linspace(0, ttotal, Ntotal)
    
    row = 6 #Number of variables of the Jansen & Rit model
    col = nnodes #Number of nodes
    y_temp = np.copy(ic) #Temporal vector to update y values
    y = np.zeros((Ntotal, row, col)) #Matrix to store values
    z = np.zeros((Ntotal,col)) #Long-range inputs without scaling
    y[0,:,:] = np.copy(ic) #First set of initial conditions
    y_inter = np.copy(ic2)  #Second set of inital conditions
    y_delay = y_inter[0,:,:] #Temporal vector for updating long-range outputs
    y_aux = y_inter[1,:,:]  #Temporal vector for updating long-range outputs' speed
    
    Z.recompile()
    f1.recompile()
    f2.recompile() 
    
    z[0,:] = Z(y_delay,0) #z initial conditions
    z_temp = z[0,:] #Temporal vector to update z values
    
    set_seed(seed) #Set the random seed
    
    if verbose == True:
        for i in range(1,Nsim):
            y_temp += dt * f1(y_temp, z_temp, i)       
            y_inter += dt * f2(y_temp[[1,2],:], y_delay, y_aux, z_temp, i)
            y_delay, y_aux = y_inter[0,:,:], y_inter[1,:,:]
            z_temp = Z(y_delay, i)
            #This line is for store values each dws points
            if (i % downsamp) == 0:
                y[i//downsamp,:,:] = y_temp
                z[i//downsamp,:] = z_temp
            if (i % (10 / dt)) == 0:
                print('Elapsed time: %i seconds'%(i * dt)) #this is for impatient people
    else:
        for i in range(1,Nsim):
            y_temp += dt * f1(y_temp, z_temp, i)       
            y_inter += dt * f2(y_temp[[1,2],:], y_delay, y_aux, z_temp, i)
            y_delay, y_aux = y_inter[0,:,:], y_inter[1,:,:]
            z_temp = Z(y_delay, i)
            #This line is for store values each dws points
            if (i % downsamp) == 0:
                y[i//downsamp,:,:] = y_temp
                z[i//downsamp,:] = z_temp        
       
    return(y, time_vector, z)


def ParamsNode():
    pardict={}
    for var in ('a','b','A','B','r0',
                'r1','r2','e0','v0','C','C1','C2','C3',
                'C4','alpha','beta','p','sigma'):
        pardict[var]=eval(var)
        
    return pardict

def ParamsNet():
    pardict={}
    for var in ('nnodes','mean_speed','std_speed'):
        pardict[var]=eval(var)
        
    return pardict

def ParamsSim():
    pardict={}
    for var in ('tmax','teq','dt','downsamp'):
        pardict[var]=eval(var)
        
    return pardict


