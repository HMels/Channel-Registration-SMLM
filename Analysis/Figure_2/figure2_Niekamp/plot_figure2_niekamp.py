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
figsize1=(4.15,2.25)
fig1,ax1=DS01.ErrorFOV(figsize=figsize1, ps=1, colorbar=True, center=[6,6], clusters=False, alpha=0)
ax1[0].add_patch(Rectangle((-35, -35), 10, 1, ec='black', fc='black'))
ax1[0].text(-34, -35, r'10$\mu$m', ha='left', va='bottom')
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

fig2=plt.figure(figsize=(3.9,2.25))
ax2=fig2.add_subplot(111)

for i in range(len(opts_name)):
   plt.errorbar(epochs_fig2, mu2_fig2[i,:], label=opts_name[i], ls=':') 

ax2.set_xscale('log')
ax2.set_xlabel('iterations')
ax2.set_ylim([4,10])
ax2.set_yticks([4,6,8, 10])
ax2.set_xlim([epochs_fig2[0],epochs_fig2[-1]])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(ncol=2, frameon=False, loc='upper right')
fig2.tight_layout()
fig2.savefig(output_path0+'Figures/fig2', transparent=True, dpi=dpi)


#%% fig3
ylim=[3,5.5]
sigma1_fig3=np.loadtxt(output_path0+'DataOutput/sigma1_fig3.txt')
sigma2_fig3=np.loadtxt(output_path0+'DataOutput/sigma2_fig3.txt')
learning_rates=np.loadtxt(output_path0+'DataOutput/learning_rates.txt')


fig3=plt.figure(figsize=(4.4,2.1))
ax3 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax4 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
p1=ax3.errorbar(learning_rates, sigma1_fig3, yerr=None,
               xerr=None, ms=5, ls=':', fmt='', color='blue', label='estimation')
p3=ax3.errorbar(learning_rates, sigma2_fig3, yerr=None,
               xerr=None, ms=5, ls=':', fmt='', color='red', label='testing')

ax3.set_xscale('log')
ax3.set_ylabel(r'precision [nm]')
ax3.set_ylim([ylim[0],ylim[1]])
ax3.set_xlim([learning_rates[0],learning_rates[np.argwhere(sigma2_fig3==100)[0]]])
ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.legend(handles=[p1], loc='lower right', ncol=1,frameon=False)
ax3.get_xaxis().set_visible(False)
ax3.set_xticks([])

ylim1=[0,1]
mu1_fig3=np.loadtxt(output_path0+'DataOutput/mu1_fig3.txt')
mu2_fig3=np.loadtxt(output_path0+'DataOutput/mu2_fig3.txt')
learning_rates=np.loadtxt(output_path0+'DataOutput/learning_rates.txt')


p1=ax4.errorbar(learning_rates, np.abs(mu1_fig3), yerr=None,
               xerr=None, ms=5, ls=':', fmt='', color='blue', label='estimation')
p3=ax4.errorbar(learning_rates, np.abs(mu2_fig3), yerr=None,
               xerr=None, ms=5, ls=':', fmt='', color='red', label='testing')

ax4.set_xscale('log')
ax4.set_xlabel('learning-rate')
ax4.set_ylabel(r'bias')
ax4.set_ylim(ylim1)
ax4.set_xlim([learning_rates[0],learning_rates[np.argwhere(sigma2_fig3==100)[0]]])
ax4.yaxis.set_major_locator(MaxNLocator(integer=True))
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
fig3.tight_layout()
fig3.savefig(output_path0+'Figures/fig3', transparent=True, dpi=dpi)


#%% fig4
sigma1_fig4=np.loadtxt(output_path0+'DataOutput/sigma1_fig4.txt')
sigma2_fig4=np.loadtxt(output_path0+'DataOutput/sigma2_fig4.txt')
gridsizes=np.loadtxt(output_path0+'DataOutput/gridsizes.txt')

    

fig4=plt.figure(figsize=(4.03,2.1))
ax5 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax6 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
lns1=ax5.errorbar(gridsizes, sigma1_fig4, yerr=None, 
                  xerr=None, fmt='',ms=5, color='blue', linestyle=':', label='estimation')
lns2=ax5.errorbar(gridsizes*1.02, sigma2_fig4, yerr=None, 
                  xerr=None, fmt='',ms=5, color='red', linestyle=':', label='testing')

ax5.set_yticks([])
ax5.set_xscale('log')
ax5.set_ylim([ylim[0], ylim[1]])
ax5.yaxis.set_major_locator(MaxNLocator(integer=True))
ax5.set_xlim(gridsizes[0],gridsizes[np.argwhere(np.isnan(sigma2_fig4))[0][0]])
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['left'].set_visible(False)
ax5.spines['bottom'].set_visible(False)
ax5.get_xaxis().set_visible(False)
ax5.set_yticks([])
ax5.set_xticks([])
ax5.legend(handles=[p3], loc='lower right', ncol=1,frameon=False)


mu1_fig4=np.loadtxt(output_path0+'DataOutput/mu1_fig4.txt')
mu2_fig4=np.loadtxt(output_path0+'DataOutput/mu2_fig4.txt')
gridsizes=np.loadtxt(output_path0+'DataOutput/gridsizes.txt')

    
lns1=ax6.errorbar(gridsizes, np.abs(mu1_fig4), yerr=None, 
                  xerr=None, fmt='',ms=5, color='blue', linestyle=':', label='estimation')
lns2=ax6.errorbar(gridsizes*1.02, np.abs(mu2_fig4), yerr=None, 
                  xerr=None, fmt='',ms=5, color='red', linestyle=':', label='testing')

ax6.set_xlabel('gridsize [nm]')
ax6.set_yticks([])
ax6.set_xscale('log')
ax6.set_ylim(ylim1)
ax6.yaxis.set_major_locator(MaxNLocator(integer=True))
ax6.set_xlim(gridsizes[0],gridsizes[np.argwhere(np.isnan(sigma2_fig4))[0][0]])
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['left'].set_visible(False)
ax6.set_yticks([])
plt.tight_layout()
fig4.savefig(output_path0+'Figures/fig4', transparent=True, dpi=dpi)