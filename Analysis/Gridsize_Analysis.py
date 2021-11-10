# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:18:47 2021

@author: Mels
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
import copy
import time
import scipy.special as scpspc

import sys
sys.path.insert(0, 'C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment')
from dataset import dataset
from dataset_simulation import dataset_simulation, dataset_copy, Deform, Affine_Deform

plt.close('all')
#%%
def calculate_curve(pos1, pos2, xlim, mu=0):
    def func(r, sigma, mu=mu):
        # from Churchman et al 2006
        sigma2=sigma**2
        if mu==0:
            return r/sigma2*np.exp(-r**2/2/sigma2)
        else:
            return (r/sigma2)*np.exp(-(mu**2+r**2)/2/sigma2)*scpspc.jv(0, r*mu/sigma2)
    
    dist = np.sqrt( np.sum( ( pos1 - pos2 )**2, axis = 1) )
    dist=dist[np.argwhere(dist<xlim)]
    n=np.histogram(dist)
    
    N = pos1.shape[0] * ( n[1][1]-n[1][0] )
    xn=(n[1][:-1]+n[1][1:])/2
    popt, pcov = curve_fit(func, xn, n[0]/N, p0=np.std(xn))
    return popt[0], pcov[0][0]
        
        
if True: #% Load Excel Niekamp
    DS1 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv',
                  linked=False, pix_size=1, loc_error=1.4, mu=.3, FrameLinking=True, BatchOptimization=False)
    DS2 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set2/set2_beads_locs.csv',
                  linked=False, pix_size=1, loc_error=1.4, mu=.3, FrameLinking=True)
    DS1.load_dataset_excel()
    DS2.load_dataset_excel()
    DS1.pix_size=159
    DS2.pix_size=DS1.pix_size
    DS1.link_dataset()
    DS2.link_dataset()
    
    ## optimization params
    learning_rates = [1e3, .1, 1e-3]
    epochs = [100, None, 300]
    pair_filter = [250, 30, 30]
    

#%%
begin=50
end=25000
N=50

gridsizes=np.logspace(np.log10(begin), np.log10(end), N, endpoint=False)
sigma1,sigma_error1=(np.zeros([len(gridsizes)],dtype=float),np.zeros([len(gridsizes)],dtype=float))
sigma2,sigma_error2=(np.zeros([len(gridsizes)],dtype=float),np.zeros([len(gridsizes)],dtype=float))
filtered1,filtered2=(np.zeros([len(gridsizes)],dtype=float),np.zeros([len(gridsizes)],dtype=float))

for i in range(len(gridsizes)):
    gridsize=gridsizes[i]

    try:
        DS10=copy.deepcopy(DS1)
        DS20=copy.deepcopy(DS2)
        DS10.TrainRegistration(execute_linked=True, learning_rates=learning_rates, 
                              epochs=epochs, pair_filter=pair_filter, gridsize=gridsize)
        
        if DS20 is not None:
            DS20.copy_models(DS10) ## Copy all mapping parameters
            DS20.ApplyRegistration()
            DS20.Filter(pair_filter[1])
            
        sgm1, sigma_error1[i] = calculate_curve(DS10.ch1.pos, DS10.ch2.pos, pair_filter[2], mu=DS1.loc_error)
        sgm2, sigma_error2[i] = calculate_curve(DS20.ch1.pos, DS20.ch2.pos, pair_filter[2], mu=DS2.loc_error)
        if sgm1>0: sigma1[i]=sgm1
        else: sigma1[i]=100
        if sgm2>0: sigma2[i]=sgm2
        else: sigma2[i]=100
        filtered1[i]=(1-(DS10.ch1.pos.shape[0]/DS10.ch10.pos.shape[0]))*100
        filtered2[i]=(1-(DS20.ch1.pos.shape[0]/DS20.ch10.pos.shape[0]))*100
    except:
        sigma1[i]=100
        sigma2[i]=100
        filtered1[i]=100
        filtered2[i]=100
    
    
#%% plotting
plt.figure()
plt.errorbar(gridsizes, sigma1, xerr=None, yerr=sigma_error1,
             fmt='o',ms=5, color='blue', linestyle=':', label='Training')
plt.errorbar(gridsizes, sigma2, xerr=None, yerr=sigma_error2,
             fmt='d',ms=5, color='red', linestyle=':', label='Testing')
plt.hlines(np.sqrt(2)*DS1.loc_error, gridsizes[0],gridsizes[-1], color='black',label='CRLB')
plt.legend()
plt.xlabel('gridsize [nm]')
plt.ylabel(r'$\sigma$ [nm]')
plt.xscale('log')
plt.ylim(0,20)
plt.xlim(gridsizes[0],gridsizes[-1])


#%%
fig, ax1 = plt.subplots(figsize=(12,6))
ax2 = ax1.twinx()
lns1=ax1.errorbar(gridsizes, sigma1, xerr=None, yerr=sigma_error1,
             fmt='',ms=5, color='blue', linestyle=':', label=r'$\sigma_{training}$')
lns2=ax1.errorbar(gridsizes*1.02, sigma2, xerr=None, yerr=sigma_error2,
             fmt='',ms=5, color='red', linestyle=':', label=r'$\sigma_{testing}$')
lns3=ax1.hlines(np.sqrt(2)*DS1.loc_error, gridsizes[0],gridsizes[-1],linestyle='-.', color='black',label=r'$\sigma_{CRLB}$')

gs=np.logspace(np.log10(begin), np.log10(end), N*2+1, endpoint=True)
lns4=ax2.bar(gs[:-1:2], filtered1, width=np.diff(gs)[::2], align="edge",
             label='% filtered training', alpha=.3)
lns5=ax2.bar(gs[1::2], filtered2, width=np.diff(gs)[1::2], align="edge", 
             label='% filtered testing', alpha=.3)

opt_weight=np.min(sigma2)
opt_weight_idx=np.argmin(sigma2)
ax1.vlines(gridsizes[opt_weight_idx], 0, 30, color='green', linestyle='--', alpha=0.5,
           label=('Optimal Gridsize='+str(round(gridsizes[opt_weight_idx]))+r'nm, $\sigma_{training,opt}$='+str(round(sigma2[opt_weight_idx],2))))
ax1.hlines(sigma2[opt_weight_idx],gridsizes[0],gridsizes[-1], linestyle='--', alpha=0.5, color='green')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)

ax1.set_xlabel('gridsize [nm]')
ax1.set_ylabel(r'$\sigma$ [nm]')
ax2.set_ylabel('% of points filtered')
ax1.set_xscale('log')
#ax2.set_xscale('log')
ax1.set_ylim(0,30)
ax2.set_ylim(0,200)
ax2.set_yticks([0,25,50,75,100])
ax1.set_xlim(gridsizes[0],gridsizes[-1])
plt.tight_layout()