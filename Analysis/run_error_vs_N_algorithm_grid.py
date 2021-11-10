# run_error_vs_N_algorithm
"""
Created on Tue Oct 26 16:19:26 2021

@author: Mels
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import copy
import numpy.random as rnd
from scipy.optimize import curve_fit

import sys
sys.path.insert(0, 'C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment')

from dataset import dataset
from dataset_simulation import dataset_simulation, dataset_copy, Deform, Affine_Deform
import time
from Channel import Channel
from Align_Modules.Splines import CatmullRomSpline2D

plt.close('all')
#%% Plotting
def ErrorDist(popt, N, xlim=31, error=None):
    ## fit bar plot data using curve_fit
    def func(r, sigma):
        # from Churchman et al 2006
        sigma2=sigma**2
        return r/sigma2*np.exp(-r**2/2/sigma2)
        #return A*(r/sigma2)/(2*np.pi)*np.exp(-(mu**2+r**2)/2/sigma2)*scpspc.jv(0, r*mu/sigma2)
    
    plt.figure()
    x = np.linspace(0, xlim, 1000)
    ## plot how function should look like
    if error is not None:
        sgm=np.sqrt(2)*error
        y = func(x, sgm)
        plt.plot(x, y, '--', c='black', label=(r'optimum: $\sigma$='+str(round(sgm,2))+'[nm]'))
        
    for n in range(len(N)):
        y = func(x, popt[n])
        plt.plot(x, y, label=(r'fit: $\sigma$='+str(np.round(popt[n],2))+'[nm], for N='+str(100*round(N[n]/100))))
    

    # Some extra plotting parameters
    plt.ylim(0)
    plt.xlim([0,xlim])
    plt.xlabel('Absolute error [nm]')
    plt.ylabel('# of localizations')
    plt.legend()
    plt.tight_layout()
    
    
def SplinesDeform(Dataset, gridsize=3000, edge_grids=1, error=10):
    Dataset.edge_grids=edge_grids
    Dataset.gridsize=gridsize
    ControlPoints=Dataset.generate_CPgrid(gridsize, edge_grids)
    
    #ControlPoints+=rnd.randn(ControlPoints.shape[0],ControlPoints.shape[1],ControlPoints.shape[2])*error/gridsize
    ControlPoints=ControlPoints.numpy()
    ControlPoints[::2, ::2,:]+=error/gridsize
    ControlPoints[1::2, 1::2,:]-=error/gridsize
    ControlPoints=tf.Variable(ControlPoints, trainable=False, dtype=tf.float32)
    
    Dataset.SplinesModel=CatmullRomSpline2D(ControlPoints)
    Dataset.Transform_Splines()
    Dataset.SplinesModel=None
    Dataset.gridsize=None
    Dataset.edge_grids=None
    return Dataset


#%%
## optimization params
learning_rates = [1000, .1, 1e-3]
epochs = [100, 100, 100]
pair_filter = [None, None,  20]
gridsize=3000


DS2=None
Nsim=1
Ntimes = [4, 16, 64, 256, 512] 
sigma, N=(np.zeros([len(Ntimes), Nsim], dtype=np.float32), np.zeros([len(Ntimes), Nsim], dtype=np.float32))
for i in range(len(Ntimes)):
    Ncopy=Ntimes[i]
    for j in range(Nsim):
        DS1 = dataset_simulation(imgshape=[512, 512], loc_error=1.4, linked=True,
                                 pix_size=159, FrameLinking=False, FrameOptimization=False)
        deform=Affine_Deform()
        DS1.generate_dataset_grid(N=1000*Ncopy, deform=deform)
        DS1=SplinesDeform(DS1, gridsize=1.5*gridsize, error=3)
        DS1, DS2 = DS1.SplitDataset(linked=True)
        #DS1.ErrorPlotImage(DS2)
        
        DS1.ShiftModel=None
        DS1.Train_Shift(lr=learning_rates[0], epochs=epochs[0], opt_fn=tf.optimizers.Adagrad)
        DS1.Transform_Shift()
        DS1.Filter(pair_filter[0])
        
        DS1.AffineModel=None
        DS1.Train_Affine(lr=learning_rates[1], epochs=epochs[1], opt_fn=tf.optimizers.Adam)
        DS1.Transform_Affine()
        
        DS1.SplinesModel=None
        DS1.Train_Splines(lr=learning_rates[2], epochs=epochs[2], gridsize=gridsize, edge_grids=1, opt_fn=tf.optimizers.SGD)
        DS1.Transform_Splines()
        DS1.Filter(pair_filter[1])
        
        if DS2 is not None: 
            DS2.copy_models(DS1)
            DS2.Transform_Shift()
            DS2.Transform_Affine()
            DS2.Transform_Splines()
            DS2.Filter(pair_filter[1])
            sg = DS2.ErrorDistribution_r(nbins=100, xlim=pair_filter[2], error=DS1.loc_error)
            N[i,j]=(DS2.ch1.pos.shape[0])
        else: 
            sg = DS1.ErrorDistribution_r(nbins=100, xlim=pair_filter[2], error=DS1.loc_error)
            N[i,j]=(DS1.ch1.pos.shape[0])
        sigma[i,j]=(sg)
    
#%% plotting
ErrorDist(np.average(sigma,axis=1), np.average(N,axis=1), xlim=pair_filter[2], error=DS1.loc_error)
