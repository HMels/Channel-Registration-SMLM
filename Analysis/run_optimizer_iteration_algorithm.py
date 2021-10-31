# main.py
"""
Created on Thu Sep  9 14:55:12 2021

@author: Mels
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from dataset import dataset
import numpy as np
import copy

import sys
sys.path.insert(0, 'C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment')

def ErrorDistribution_r(dataset):
    if not dataset.linked: raise Exception('Dataset should first be linked before registration errors can be derived!')
    pos1=dataset.ch1.pos_all()
    pos2=dataset.ch2.pos_all()
        
    # Calculating the error
    dist, avg, r = dataset.ErrorDist(pos1, pos2)
    p0 = np.std(dist)
    return p0, np.average(dist)


#%%
opts=[
      tf.optimizers.Adam,
      tf.optimizers.Adagrad,
      tf.optimizers.Adadelta,
      tf.optimizers.Adamax,
      tf.optimizers.Ftrl,
      tf.optimizers.Nadam,
      tf.optimizers.RMSprop,
      tf.optimizers.SGD
      ]

opts_name=[
      'Adam',
      'Adagrad',
      'Adadelta',
      'Adamax',
      'Ftrl',
      'Nadam',
      'RMSprop',
      'SGD'
      ]

epochs = 8*np.logspace(0,3,15).astype('int')
gridsize=3000
DS0 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv',
              linked=False, pix_size=1, loc_error=1.4, FrameLinking=True, FrameOptimization=False)
DS0.load_dataset_excel()
DS0.link_dataset()

DS1=copy.deepcopy(DS0)
#DS1.Train_Shift(lr=1000, epochs=100)
#DS1.Transform_Shift()

popt = ErrorDistribution_r(DS1)
sigma1, sigma2,sigma3=(popt[0]*np.ones([len(opts), len(epochs)]),popt[0]*np.ones([len(opts), len(epochs)]),
                       popt[0]*np.ones([len(opts), len(epochs)]))
mu1,mu2,mu3=(popt[1]*np.ones([len(opts), len(epochs)]),popt[1]*np.ones([len(opts), len(epochs)]),
             popt[1]*np.ones([len(opts), len(epochs)]))


#%% Shift
for i in range(len(opts)):
    for j in range(len(epochs)):
        DS2=copy.deepcopy(DS0)        
        try:
            DS2.Train_Shift(lr=1000, epochs=epochs[j], opt_fn=opts[i])
            DS2.Transform_Shift()
            popt = ErrorDistribution_r(DS2)
            sigma1[i,j]=(popt[0])
            mu1[i,j]=(popt[1])
        except:
            sigma1[i,j]=(0)
            mu1[i,j]=(np.nan)
        
      
 
#%% Affine
for i in range(len(opts)):
    for j in range(len(epochs)):
        DS2=copy.deepcopy(DS1)
        try:
            DS2.Train_Affine(lr=10, epochs=epochs[j], opt_fn=opts[i])
            DS2.Transform_Affine()
            popt = ErrorDistribution_r(DS2)
            sigma2[i,j]=(popt[0])
            mu2[i,j]=(popt[1])
        except:
            sigma2[i,j]=(0)
            mu2[i,j]=(np.nan)
            
            
#%% Spline
for i in range(len(opts)):
    for j in range(len(epochs)):
        DS2=copy.deepcopy(DS1)
        try:
            DS2.Train_Splines(lr=1e-2, epochs=epochs[j], gridsize=gridsize, edge_grids=1, opt_fn=opts[i])
            DS2.Transform_Splines()
            popt = ErrorDistribution_r(DS2)
            sigma3[i,j]=(popt[0])
            mu3[i,j]=(popt[1])
        except:
            sigma3[i,j]=(0)
            mu3[i,j]=(np.nan)


#%% Plotting
fig, ax = plt.subplots(nrows = 4, ncols = 2)
popt = ErrorDistribution_r(DS1)
for i in range(len(opts)):
    i_mod = i//2
    i_dev = i%2
    ax[i_mod,i_dev].title.set_text(str(opts_name[i]))
    ax[i_mod,i_dev].errorbar(np.concatenate([[1],epochs]), np.concatenate([[popt[1]],mu1[i,:]]), 
                             yerr=np.concatenate([[popt[0]],sigma1[i,:]]), label='Shift',
                             ls=':',fmt='', color='blue', ecolor='blue', capsize=3)
    ax[i_mod,i_dev].errorbar(np.concatenate([[1],epochs]), np.concatenate([[popt[1]],mu2[i,:]]), 
                             yerr=np.concatenate([[popt[0]],sigma2[i,:]]), label='Affine',
                             ls=':',fmt='', color='red', ecolor='red', capsize=3)
    ax[i_mod,i_dev].errorbar(np.concatenate([[1],epochs]), np.concatenate([[popt[1]],mu3[i,:]]), 
                             yerr=np.concatenate([[popt[0]],sigma3[i,:]]), label='Catmull-Rom Splines',
                             ls=':',fmt='', color='green', ecolor='green', capsize=3)
    
    ax[i_mod,i_dev].set_ylim(5,5e5)
    ax[i_mod,i_dev].set_xscale('log')
    ax[i_mod,i_dev].set_yscale('log')
    ax[i_mod,i_dev].set_xlim(1, epochs[-1])
    
    
    if i_mod==0 and i_dev==1: ax[i_mod,i_dev].legend()
    #else: ax[i_mod,i_dev].legend()
    if i_mod==3: ax[i_mod,i_dev].set_xlabel('Iterations')
    else: ax[i_mod,i_dev].set_xticklabels([])
    if i_dev==0: ax[i_mod,i_dev].set_ylabel('Average Error [nm]')
    else: ax[i_mod,i_dev].set_yticklabels([])

