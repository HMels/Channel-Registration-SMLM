# main.py
"""
Created on Thu Sep  9 14:55:12 2021

@author: Mels
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from dataset import dataset
import time
import numpy as np
import copy

import sys
sys.path.insert(0, 'C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment')

def ErrorDistribution_r(dataset, nbins=30, error=None):
    if not dataset.linked: raise Exception('Dataset should first be linked before registration errors can be derived!')
    pos1=dataset.ch1.pos_all()
    pos2=dataset.ch2.pos_all()
        
    # Calculating the error
    dist, avg, r = dataset.ErrorDist(pos1, pos2)
    p0 = np.std(dist)
    return p0, np.average(dist)#popt


## fit bar plot data using curve_fit
def func(r, sigma):
    # from Churchman et al 2006
    sigma2=sigma**2
    return r/sigma2*np.exp(-r**2/2/sigma2)
    #return (r/sigma2)/(2*np.pi)*np.exp(-(mu**2+r**2)/2/sigma2)*scpspc.jv(0, r*mu/sigma2)


#%%
NN=20
learning_rates1 = 5*np.logspace(-5, 4, NN)
learning_rates2 = 5*np.logspace(-5, 4, NN)
learning_rates3 = np.logspace(-8, -1, NN)

epochs = [100, 500, 1000]
gridsize=3000
t1,t2,t3=([],[],[])
sigma1,sigma2,sigma3=([],[],[])
mu1,mu2,mu3=([],[],[])

DS0 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv',
              linked=False, pix_size=1, loc_error=1.4, FrameLinking=True, FrameOptimization=False)
DS0.load_dataset_excel()
DS0.link_dataset()

DS0shift = copy.deepcopy(DS0)
DS0shift.Train_Shift(lr=1000, epochs=100, opt_fn=tf.optimizers.Adagrad)
DS0shift.Transform_Shift()

for i in range(NN):
    DS1=copy.deepcopy(DS0)
    DS2=copy.deepcopy(DS0shift)
    DS3=copy.deepcopy(DS0shift)
    
    
    
    #%% DS1
    try:
        start=time.time()
        DS1.Train_Shift(lr=learning_rates1[i], epochs=epochs[0])
        DS1.Transform_Shift()
        t1.append(round((time.time()-start)/60,1))
        popt = ErrorDistribution_r(DS1, nbins=100, error=DS1.loc_error)
        sigma1.append(popt[0])
        mu1.append(popt[1])
    except:
        sigma1.append(0)
        mu1.append(np.nan)
        t1.append(0)
    
    
    #%% DS2
    try:
        start=time.time()
        DS2.Train_Affine(lr=learning_rates2[i], epochs=epochs[1], opt_fn=tf.optimizers.Adam)
        DS2.Transform_Affine()
        t2.append(round((time.time()-start)/60,1))
        popt = ErrorDistribution_r(DS2, nbins=100, error=DS2.loc_error)
        sigma2.append(popt[0])
        mu2.append(popt[1])
    except:
        sigma2.append(0)
        mu2.append(np.nan)
        t2.append(0)
        
        
    #%% DS3        
    try:
        start=time.time()
        DS3.Train_Splines(lr=learning_rates3[i], epochs=epochs[2], gridsize=gridsize, edge_grids=1, opt_fn=tf.optimizers.SGD)
        DS3.Transform_Splines()
        t3.append(round((time.time()-start)/60,1))
        popt = ErrorDistribution_r(DS3, nbins=100, error=DS3.loc_error)
        sigma3.append(popt[0])
        mu3.append(popt[1])
    except:
        sigma3.append(0)
        mu3.append(np.nan)
        t3.append(0)


#%% all in one figure
fig, ax = plt.subplots()
ax.set_xlabel('learning-rate')
ax.set_ylabel('Average Pair Distance [nm]')

p1=ax.errorbar(learning_rates1, mu1, yerr=sigma1, xerr=None, ls=':', fmt='', ecolor='blue', capsize=5, label='Shift')
p2=ax.errorbar(learning_rates2, mu2, yerr=sigma2, xerr=None, ls=':', fmt='', ecolor='red', capsize=5, label='Shift+Affine')
p3=ax.errorbar(learning_rates3, mu3, yerr=sigma3, xerr=None, ls=':', fmt='', ecolor='green', capsize=5, label='Shift+Catmull-Rom Splines')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(handles=[p1, p2, p3], loc='upper right')
fig.tight_layout()
