# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:51:11 2021

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
import time
from Channel import Channel
from CatmullRomSpline2D import CatmullRomSpline2D
plt.rc('font', size=10)
output_path0='C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment/Analysis/Figure_2/figure2_Grid/'
dpi=800 

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
    return Dataset


#%% fig5 - Error vs Density
execute_linked=True
learning_rate=2e-3
epochs=300
pair_filter = [None, None, 20]
gridsize=3800

## dataset params
CRS_error=10 # implemented splines error
random_error=3
Num=14000     # number of points
loc_error=1.4

Ntimes = [4, 16, 64, 256]

#%%
sigma_fig5,mu_fig5, N_fig5=(np.zeros([len(Ntimes), 1], dtype=np.float32),
                            np.zeros([len(Ntimes), 1], dtype=np.float32), 
                            np.zeros([len(Ntimes), 1], dtype=np.float32))
for i in range(len(Ntimes)):
    DS1 = dataset_simulation(imgshape=[512, 512], loc_error=loc_error, linked=True,
                             pix_size=159, FrameLinking=False, BatchOptimization=False)
    deform=Affine_Deform(A=np.array([[ 1.0031357 ,  0.00181658, -1.3986971], 
                                  [-0.00123012,  0.9972918, 300.3556707 ]]))
    DS1.generate_dataset_grid(N=1000*Ntimes[i], deform=deform)
    DS1=SplinesDeform(DS1, gridsize=1.5*gridsize, random_error=random_error)
    DS1, DS2 = DS1.SplitDataset(linked=True)
    
    DS1.Train_Splines(learning_rate, gridsize=gridsize, edge_grids=1, epochs=epochs)
    DS1.Apply_Splines() 
    if DS2 is not None:
        DS2.copy_models(DS1)
        DS2.Apply_Splines() 
        temp,_ = DS2.ErrorDistribution_r(nbins=100, xlim=pair_filter[2],
                                         error=DS1.coloc_error, plot_on=False)
        sigma_fig5[i,0]=temp[0]
        mu_fig5[i,0]=temp[1]
        N_fig5[i,0]=(DS2.ch1.pos.shape[0])
    else: 
        temp,_ = DS1.ErrorDistribution_r(nbins=100, xlim=pair_filter[2], 
                                         error=DS1.coloc_error, plot_on=False)
        sigma_fig5[i,0]=temp[0]
        mu_fig5[i,0]=temp[1]
        N_fig5[i,0]=(DS1.ch1.pos.shape[0])
    del DS1, DS2
    
    
#%% Plotting fig5
def ErrorDist(popt0 ,popt1, N, xlim=31, error=None, fig=None):
    ## fit bar plot data using curve_fit
    def func(r, sigma, mu):
        # from Churchman et al 2006
        sigma2=sigma**2
        return (r/sigma2)*np.exp(-(mu**2+r**2)/2/sigma2)*scpspc.jv(0, r*mu/sigma2)
    
    if fig is None: fig=plt.figure(figsize=(3.90, 2.75))
    ax=fig.add_subplot(111)
    x = np.linspace(0, xlim, 1000)
    ## plot how function should look like
    if error is not None:
        sgm=error
        y = func(x, sgm, 0)
        ax.plot(x, y, '--', c='black', label=(r'optimum'))
        
    for n in range(len(N)):
        y = func(x, popt0[n],popt1[n])
        ax.plot(x, y, label=(r'N='+str(100*round(N[n]/100))))
    

    # Some extra plotting parameters
    ax.set_ylim(0)
    ax.set_xlim([0,xlim])
    ax.set_yticks([])
    ax.set_xticks([0, 5, 10, 15,20])
    ax.set_xlabel('absolute error [nm]')
    ax.legend(loc='upper right', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig,ax
    
fig5,ax5=ErrorDist(np.average(sigma_fig5,axis=1),np.average(mu_fig5,axis=1), np.average(N_fig5,axis=1), xlim=pair_filter[2], error=np.sqrt(2)*loc_error)

plt.tight_layout()
fig5.savefig(output_path0+'Figures/fig5', transparent=True, dpi=dpi)