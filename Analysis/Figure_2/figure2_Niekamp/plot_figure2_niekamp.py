# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:32:36 2021

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
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle

import sys
sys.path.insert(0, 'C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment')

from dataset import dataset
from dataset_simulation import dataset_simulation, dataset_copy, Deform, Affine_Deform
import time
from Channel import Channel
from CatmullRomSpline2D import CatmullRomSpline2D
plt.rc('font', size=10)
dpi=800
ylim=[3,6]

plt.close('all')
#%% loading the dataset
if True: ## Niekamp
    maxDistance=1000
    DS01=dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv',
      pix_size=1, loc_error=1.4, mu=0.3,
      linked=False, FrameLinking=True, BatchOptimization=False)
    DS02=dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set2/set2_beads_locs.csv',
      pix_size=1, loc_error=1.4, mu=0.3,
      linked=False, FrameLinking=True)
    DS01.load_dataset_excel()
    DS02.load_dataset_excel()
    DS01.pix_size=159
    DS02.pix_size=DS01.pix_size
    
    Num=DS01.ch1.pos.shape[0]
    output_path0='C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment/Analysis/Figure_2/figure2_Niekamp/'
        
    pair_filter=[250, 30, 8]
    DS01.link_dataset(maxDistance=maxDistance)
        
    DS01.AffineLLS()
    DS01.Filter(pair_filter[0]) 
    
    
#%%
figsize1=(3.08,1.65)
fig1,ax1=DS01.ErrorFOV(figsize=figsize1, ps=3, colorbar=False)
ax1[0].add_patch(Rectangle((-37, -37), 10, 1, ec='black', fc='black'))
ax1[0].text(-37, -36, r'10$\mu$m', ha='left', va='bottom')
#ax1[1].text(x=np.min(DS01.ch2.pos[:,0])/1000+2, y=np.max(DS01.ch2.pos[:,1])/1000-2, 
#           s='N='+str(Num), ha='left', va='top', bbox=dict(boxstyle="square",
#                                                           ec=(1., 0.5, 0.5),
#                                                           fc=(1., 0.8, 0.8),
#                                                           ))
fig1.savefig(output_path0+'Figures/fig1', transparent=True, dpi=dpi)



#%% fig2
sigma1_fig2=np.loadtxt(output_path0+'DataOutput/sigma1_fig2.txt')
mu1_fig2=np.loadtxt(output_path0+'DataOutput/mu1_fig2.txt')
sigma2_fig2=np.loadtxt(output_path0+'DataOutput/sigma2_fig2.txt')
mu2_fig2=np.loadtxt(output_path0+'DataOutput/mu2_fig2.txt')
epochs_fig2=np.loadtxt(output_path0+'DataOutput/epochs_fig2.txt')
epochs_fig2[0]=epochs_fig2[1]*0.9

opts=[ tf.optimizers.Adam, tf.optimizers.Adagrad, tf.optimizers.Adadelta, tf.optimizers.Adamax,
      tf.optimizers.Ftrl, tf.optimizers.Nadam, tf.optimizers.RMSprop, tf.optimizers.SGD ]
opts_name=[ 'Adam', 'Adagrad', 'Adadelta', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD']
svfig=['Figure_2a','Figure_2b','Figure_2c','Figure_2d','Figure_2e','Figure_2f','Figure_2g','Figure_2h']

fig2=plt.figure(figsize=(4.15,3.5))
ax2=fig2.add_subplot(111)

for i in range(len(opts_name)):
   #plt.errorbar(epochs_fig2, mu2_fig2[i,:], yerr=sigma2_fig2[i,:], label=opts_name[i],
   #            xerr=None, ls=':', fmt='', capsize=10,) 
   plt.errorbar(epochs_fig2, mu2_fig2[i,:], label=opts_name[i], ls=':') 

ax2.set_xscale('log')
#ax.set_yscale('log')
ax2.set_xlabel('iterations')
#ax2.set_ylabel(r'$\sigma$ [nm]')
ax2.set_ylim(ylim)
ax2.set_yticks([])
ax2.set_xlim([epochs_fig2[0],epochs_fig2[-1]])
#ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(ncol=2, frameon=False)
fig2.tight_layout()
fig2.savefig(output_path0+'Figures/fig2', transparent=True, dpi=dpi)


#%% fig3
sigma1_fig3=np.loadtxt(output_path0+'DataOutput/sigma1_fig3.txt')
sigma2_fig3=np.loadtxt(output_path0+'DataOutput/sigma2_fig3.txt')
std1_fig3=np.loadtxt(output_path0+'DataOutput/std1_fig3.txt')
std2_fig3=np.loadtxt(output_path0+'DataOutput/std2_fig3.txt')
learning_rates=np.loadtxt(output_path0+'DataOutput/learning_rates.txt')


fig3, ax3 = plt.subplots(figsize=(4.25,2.25))
p1=ax3.errorbar(learning_rates, sigma1_fig3, yerr=std1_fig3,
               xerr=None, ls=':', fmt='', ecolor='blue', capsize=2, label='Training')
p3=ax3.errorbar(learning_rates, sigma2_fig3, yerr=std2_fig3,
               xerr=None, ls=':', fmt='', ecolor='green', capsize=2, label='Cross-Validation')

ax3.set_xscale('log')
ax3.set_xlabel('learning-rate')
#ax3.set_ylabel(r'$\sigma$ [nm]')
ax3.set_ylim([ylim[0],ylim[1]])
ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
ax3.legend(handles=[p1, p3], loc='upper right', frameon=False)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
fig3.tight_layout()
fig3.savefig(output_path0+'Figures/fig3', transparent=True, dpi=dpi)


#%% fig4
sigma1_fig4=np.loadtxt(output_path0+'DataOutput/sigma1_fig4.txt')
sigma2_fig4=np.loadtxt(output_path0+'DataOutput/sigma2_fig4.txt')
std1_fig4=np.loadtxt(output_path0+'DataOutput/std1_fig4.txt')
std2_fig4=np.loadtxt(output_path0+'DataOutput/std2_fig4.txt')
gridsizes=np.loadtxt(output_path0+'DataOutput/gridsizes.txt')

    
fig4, ax4 = plt.subplots(figsize=(3.90, 3.5))
lns1=ax4.errorbar(gridsizes, sigma1_fig4, yerr=std1_fig4, 
                  xerr=None, fmt='',ms=5, color='blue', linestyle=':', label='Training')
lns2=ax4.errorbar(gridsizes*1.02, sigma2_fig4, yerr=std1_fig4,
                  xerr=None, fmt='',ms=5, color='red', linestyle=':', label='Cross-Validation')
#lns3=ax1.hlines(DS01.coloc_error, gridsizes[0],gridsizes[-1],linestyle='-.', color='black',label='CRLB')

opt_weight=np.min(sigma2_fig4)
opt_weight_idx=np.argmin(sigma2_fig4)
ax4.vlines(gridsizes[opt_weight_idx], ylim[0]/2, opt_weight, color='green', linestyle='--', alpha=0.5)
ax4.hlines(opt_weight,gridsizes[0],gridsizes[opt_weight_idx], linestyle='--', alpha=0.5, color='green')
ax4.legend(loc='upper right', frameon=False, ncol=2)

ax4.set_xlabel('gridsize [nm]')
ax4.set_ylabel(r'$\sigma$ [nm]')
ax4.set_xscale('log')
ax4.set_ylim([ylim[0], ylim[1]])
ax4.yaxis.set_major_locator(MaxNLocator(integer=True))
ax4.set_xlim(gridsizes[0],gridsizes[-1])
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
plt.tight_layout()
fig4.savefig(output_path0+'Figures/fig4', transparent=True, dpi=dpi)
