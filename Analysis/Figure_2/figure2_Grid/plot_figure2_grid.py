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
#%% fig1 - Gridview
def SplinesDeform(Dataset, gridsize=3000, edge_grids=1, random_error=None): # creating a splines offset error
    Dataset.edge_grids=edge_grids
    Dataset.gridsize=gridsize
    ControlPoints=Dataset.generate_CPgrid(gridsize, edge_grids)
    
    ControlPoints=ControlPoints.numpy()
    if random_error is not None:
        ControlPoints+=np.random.rand(*(ControlPoints.shape))*random_error/gridsize
    ControlPoints=tf.Variable(ControlPoints, trainable=False, dtype=tf.float32)
    
    Dataset.SplinesModel=CatmullRomSpline2D(ControlPoints)
    Dataset.Apply_Splines()
    Dataset.SplinesModel=None
    Dataset.gridsize=None
    Dataset.edge_grids=None
    return Dataset, ControlPoints



#%% loading the dataset
if True: ## Grid
    ## dataset params
    random_error=60
    Num=14400     # number of points
    gridsize=3000
    loc_error=1.4
    DS01 = dataset_simulation(imgshape=[512, 512], loc_error=loc_error, linked=True,
                             pix_size=159, FrameLinking=False, BatchOptimization=False)
    
    deform=Affine_Deform(A=np.array([[ 1.0031357 ,  0.00181658, -1.3986971], 
                                  [-0.00123012,  0.9972918, 50.3556707 ]]))
    DS01.generate_dataset_grid(N=Num, deform=deform)
    DS01, ControlPoints=SplinesDeform(DS01, gridsize=1.5*gridsize, random_error=random_error)
    
    ## optimization params
    execute_linked=True
    learning_rate=1e-3
    epochs=300
    pair_filter = [None, None, 100]
    
    output_path0='C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment/Analysis/Figure_2/figure2_Grid/'


figsize1=(1.872,1.872)
fig,ax=DS01.show_channel(DS01.ch1.pos, ps=5, figsize=figsize1, alpha=.7)
DS01.show_channel(DS01.ch2.pos, ps=5, color='blue',fig=fig, ax=ax, alpha=.7)

window=np.array([[0, 10000],[-5000, 5000]])
#window=np.array([[-18500,-3500],[-7500, 7500]])
ax.add_patch(plt.Rectangle((window[0,0]/1000,window[1,0]/1000), (window[0,1]-window[0,0])/1000,
                            (window[1,1]-window[1,0])/1000, ec='red', fc='none', linewidth=1, zorder=10))
fig.savefig(output_path0+'Figures/fig0', transparent=True, dpi=dpi)

DS01copy=DS01.SubsetWindow(window=window)
figsize1=(1.872,1.872)
fig,ax=DS01copy.show_channel(DS01copy.ch1.pos, ps=20, figsize=figsize1, alpha=.7, addpatch=False)
DS01copy.show_channel(DS01copy.ch2.pos, ps=20, color='blue',fig=fig, ax=ax, alpha=.7, addpatch=False)
x1=np.min(DS01copy.ch1.pos[:,0])/1000 +.5
x2=np.min(DS01copy.ch1.pos[:,1])/1000 +.5
ax.add_patch(Rectangle((x1,x2), 2, .1, ec='black', fc='black'))
ax.text(x1, x2+.1, r'2$\mu$m', ha='left', va='bottom')
fig.savefig(output_path0+'Figures/fig0zoom', transparent=True, dpi=dpi)


#%
figsize1=(3.775,1.872)
fig1,ax1=DS01.ErrorFOV(figsize=figsize1, ps=1, colorbar=True, center=[6,6], 
                       clusters=False, alpha=0, precision=2000, norm=1000)
ax1[0].add_patch(Rectangle((-35, -35), 10, 1, ec='black', fc='black'))
ax1[0].text(-35, -34, r'10$\mu$m', ha='left', va='bottom')
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

fig2=plt.figure(figsize=(4.15,3.2))
ax2=fig2.add_subplot(111)

for i in range(len(opts_name)):
   plt.errorbar(epochs_fig2, mu1_fig2[i,:], label=opts_name[i], ls=':') 

ax2.set_xscale('log')
ax2.set_xlabel('iterations')
ax2.set_ylabel('bias [nm]')
ax2.set_ylim(0,600)
ax2.set_yticks([0, 200, 400, 600])
ax2.set_xlim([epochs_fig2[0],epochs_fig2[-1]])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(ncol=1, frameon=False)
fig2.tight_layout()
fig2.savefig(output_path0+'Figures/fig2', transparent=True, dpi=dpi)

'''
#%% fig3
sigma1_fig3=np.loadtxt(output_path0+'DataOutput/sigma1_fig3.txt')
sigma2_fig3=np.loadtxt(output_path0+'DataOutput/sigma2_fig3.txt')
mu1_fig3=np.loadtxt(output_path0+'DataOutput/mu1_fig3.txt')
mu2_fig3=np.loadtxt(output_path0+'DataOutput/mu2_fig3.txt')
learning_rates=np.loadtxt(output_path0+'DataOutput/learning_rates.txt')


fig3, ax3 = plt.subplots(figsize=(4.9,2.25))
p1=ax3.errorbar(learning_rates, mu1_fig3, yerr=sigma1_fig3,
               xerr=None, ls=':', fmt='', ecolor='blue', label='estimation')
p3=ax3.errorbar(learning_rates*1.13, mu2_fig3, yerr=sigma2_fig3,
               xerr=None, ls=':', fmt='', ecolor='red', label='testing')

ax3.set_xscale('log')
ax3.set_xlabel('learning-rate')
ax3.set_ylabel('bias [nm]')
ax3.set_ylim([0,600])
ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
ax3.legend(handles=[p1,p3], loc='lower left', frameon=False, ncol=1)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_yticks([0,200,400,600])
fig3.tight_layout()
fig3.savefig(output_path0+'Figures/fig3', transparent=True, dpi=dpi)

'''
#%% fig3
sigma1_fig3=np.loadtxt(output_path0+'DataOutput/sigma1_fig3.txt')
sigma2_fig3=np.loadtxt(output_path0+'DataOutput/sigma2_fig3.txt')
sigma1_fig3[np.argwhere(sigma1_fig3==0)]=1e6
sigma2_fig3[np.argwhere(sigma2_fig3==0)]=1e6
learning_rates=np.loadtxt(output_path0+'DataOutput/learning_rates.txt')

fig3=plt.figure(figsize=(3.9,2.75))
ax3 = plt.subplot2grid((5, 1), (0, 0), rowspan=2)
ax4 = plt.subplot2grid((5, 1), (2, 0), rowspan=3)
p1=ax3.errorbar(learning_rates, sigma1_fig3, yerr=None,
               xerr=None, ms=5, ls=':', fmt='', color='blue', label='estimation')
p3=ax3.errorbar(learning_rates, sigma2_fig3, yerr=None,
               xerr=None, ms=5, ls=':', fmt='', color='red', label='testing')

ax3.set_xscale('log')
ax3.set_ylabel(r'precision [nm]')
ax3.set_ylim([0,200])
ax3.set_xlim([learning_rates[0], learning_rates[np.argwhere(sigma2_fig3==1e6)[0]]])
ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.set_yticks([0,200])
ax3.legend(handles=[p1,p3], loc='upper right', frameon=False, ncol=1)
ax3.get_xaxis().set_visible(False)
ax3.set_xticks([])

mu1_fig3=np.loadtxt(output_path0+'DataOutput/mu1_fig3.txt')
mu2_fig3=np.loadtxt(output_path0+'DataOutput/mu2_fig3.txt')
mu2_fig3[np.argwhere(np.isnan(mu1_fig3))]=1e6
mu2_fig3[np.argwhere(np.isnan(mu2_fig3))]=1e6
learning_rates=np.loadtxt(output_path0+'DataOutput/learning_rates.txt')


p1=ax4.errorbar(learning_rates, np.abs(mu1_fig3), yerr=None,
               xerr=None, ms=5, ls=':', fmt='', color='blue', label='estimation')
p3=ax4.errorbar(learning_rates, np.abs(mu2_fig3), yerr=None,
               xerr=None, ms=5, ls=':', fmt='', color='red', label='testing')

ax4.set_xscale('log')
ax4.set_xlabel('learning-rate')
ax4.set_ylabel(r'bias [nm]')
ax4.set_ylim([0,600])
ax4.set_xlim([learning_rates[0], learning_rates[np.argwhere(sigma2_fig3==1e6)[0]]])
ax4.yaxis.set_major_locator(MaxNLocator(integer=True))
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.set_yticks([0,200,400,600])
fig3.tight_layout()
fig3.savefig(output_path0+'Figures/fig3', transparent=True, dpi=dpi)


#%% fig4
DS01.Train_Splines(learning_rate=1e-3, epochs=300, gridsize=3800, edge_grids=1)
DS01.Apply_Splines() 


#%%
figsize1=(4.054,2.010)
fig1,ax1=DS01.ErrorFOV(figsize=figsize1, ps=1, colorbar=True, center=[6,6], 
                       clusters=False, alpha=0, precision=2000, norm=1000)
ax1[0].add_patch(Rectangle((-35, -35), 10, 1, ec='black', fc='black'))
ax1[0].text(-35, -34, r'10$\mu$m', ha='left', va='bottom')
fig1.savefig(output_path0+'Figures/fig1aligned', transparent=True, dpi=dpi)