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


plt.close('all')

def create_larger_dataset(Dataset, Ncopy, pair_filter=[250, 30]):                          
    pos1=Dataset.ch1.pos_all()
    pos2=Dataset.ch2.pos_all()
    frame=Dataset.ch1.frame.numpy()
    
    ## Image starts at 0
    Min=np.min(np.stack((np.min(pos1,axis=0), np.min(pos2,axis=0)),axis=1), axis=1)
    pos1-=Min
    pos2-=Min
    Max=np.max(np.stack((np.max(pos1,axis=0), np.max(pos2,axis=0)),axis=1), axis=1)
    
    frame1, posa, posb=(frame, pos1, pos2)
    for i in range(Ncopy):
        for j in range(Ncopy):
            shift=tf.stack((i*Max[0]*tf.ones((pos1.shape[0])),j*Max[0]*tf.ones((pos1.shape[0]))), axis=1)
            posa=tf.concat((posa, pos1+shift), axis=0)
            posb=tf.concat((posb, pos2+shift), axis=0)
            frame1=tf.concat((frame1, frame), axis=0)
    pos1=posa
    pos2=posb
            
    '''
    MaxMat=Max[None,None,:]*np.stack(((np.arange(0,Ncopy)[None,:]*np.ones(Ncopy)[:,None]), 
                                      (np.arange(0,Ncopy)[:,None]*np.ones(Ncopy)[None,:])),axis=2)
    pos1=np.reshape(pos1[None,None,:,:]+MaxMat[:,:, None,:], [-1,2])
    pos2=np.reshape(pos2[None,None,:,:]+MaxMat[:,:, None,:], [-1,2])
    frame1=np.reshape(Dataset.ch1.frame[None,:]*np.ones(((Ncopy)**2,1)), [-1])
    
    # Generate localization error
    pos1=generate_locerror(pos1, Dataset.loc_error) 
    pos2=generate_locerror(pos2, Dataset.loc_error) 
    Dataset.loc_error*=np.sqrt(2)
    
    
    # put back the calculated deformation
    pos2-=Max*Ncopy/2
    deform= Affine_Deform(
        tf.concat([Dataset.AffineModel.variables[0],Dataset.AffineModel.variables[1][:,None]],axis=1)
        )
    pos2=deform.deform(pos2)
    pos2+=Dataset.ShiftModel.variables[0]
    pos2+=Max*Ncopy/2
    '''
    del Dataset.ch1, Dataset.ch2
    Dataset.ch1 = Channel(pos1,frame1)
    Dataset.ch2 = Channel(pos2,frame1)
    Dataset.ch20 = copy.deepcopy(Dataset.ch2)
    
    ## reset models and image
    Dataset.center_image()
    Dataset.SplinesModel.ControlPoints.assign(Dataset.SplinesModel.ControlPoints*Ncopy)
    Dataset.gridsize*=Ncopy
    Dataset.ShiftModel.d.assign(-1*Dataset.ShiftModel.d)
    #Dataset.Transform_Shift()
    unit=tf.Variable([[1,0],[0,1]], dtype=tf.float32)
    Dataset.AffineModel.A.assign(
        (tf.linalg.inv(Dataset.AffineModel.A)-unit)/10 + unit
        )
    Dataset.AffineModel.d.assign(-1*Dataset.AffineModel.d)
    Dataset.Transform_Affine()
    #Dataset.Transform_Splines()
    Dataset.ShiftModel=None
    Dataset.AffineModel=None
    Dataset.SplinesModel=None
    return Dataset
    

def generate_locerror(pos, loc_error):
# Generates a Gaussian localization error over the localizations
    if loc_error != 0:
        pos[:,0] += rnd.normal(0, loc_error, pos.shape[0])
        pos[:,1] += rnd.normal(0, loc_error, pos.shape[0])
    return pos


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
        y = func(x, *popt[n])
        plt.plot(x, y, label=(r'fit: $\sigma$='+str(np.round(popt[n][0],2))+' for N='+str(N[n])))
    

    # Some extra plotting parameters
    plt.ylim(0)
    plt.xlim([0,xlim])
    plt.xlabel('Absolute error [nm]')
    plt.ylabel('# of localizations')
    plt.legend()
    plt.tight_layout()
    

#%% Load Excel Niekamp
DS0=dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv',
              linked=False, pix_size=1, loc_error=1.4, FrameLinking=True, FrameOptimization=False)
DS0.load_dataset_excel()
DS0.link_dataset()
DS0.Train_Shift(lr=1000, epochs=100, opt_fn=tf.optimizers.Adagrad)
DS0.Transform_Shift()
DS0.Filter(250)
DS0.Train_Affine(lr=.1, epochs=100, opt_fn=tf.optimizers.Adam)
DS0.Transform_Affine()
DS0.Train_Splines(lr=1e-3, epochs=100, gridsize=3000, edge_grids=1, opt_fn=tf.optimizers.SGD)
DS0.Transform_Splines()
DS0.Filter(30)


#%%
## optimization params
learning_rates = [1000, .1, 1e-3]
epochs = [100, 100, 100]
pair_filter = [1000, 1000,  1000]
gridsize=3000

sigma=[]
N = []
for Ncopy in [1,2]:
    DS1=copy.deepcopy(DS0)
    DS1=create_larger_dataset(DS1, Ncopy=Ncopy)
    DS1.link_dataset()
    
    #DS1.Train_Shift(lr=learning_rates[0], epochs=epochs[0], opt_fn=tf.optimizers.Adagrad)
    #DS1.Transform_Shift()
    DS1.Filter(pair_filter[0])
    DS1.Train_Affine(lr=learning_rates[1], epochs=epochs[1], opt_fn=tf.optimizers.Adam)
    DS1.Transform_Affine()
    #DS1.Train_Splines(lr=learning_rates[2], epochs=epochs[2], gridsize=gridsize, edge_grids=1, opt_fn=tf.optimizers.SGD)
    #DS1.Transform_Splines()
    DS1.Filter(pair_filter[1])
    
    sg = DS1.ErrorDistribution_r(nbins=100, xlim=pair_filter[2], error=DS1.loc_error)
    sigma.append(sg)
    N.append(DS1.ch1.pos.shape[0])
    
#%% plotting
ErrorDist(sigma, N, xlim=pair_filter[2], error=DS0.loc_error)

#%% test 
'''
plt.close('all')
DS0.ErrorDistribution_r(nbins=100, xlim=pair_filter[2], error=DS0.loc_error)
#DS0.PlotSplineGrid(gridsize=DS0.gridsize, edge_grids=DS0.edge_grids, plotarrows=False, plotmap=True)

DS1=copy.deepcopy(DS0)
DS1=create_larger_dataset(DS1, Ncopy=1)

DS1.Train_Shift(lr=1000, epochs=100, opt_fn=tf.optimizers.Adagrad)
DS1.Transform_Shift()
DS1.Filter(pair_filter[0])

DS1.Train_Affine(lr=.1, epochs=100, opt_fn=tf.optimizers.Adam)
DS1.Transform_Affine()

DS1.Train_Splines(lr=1e-3, epochs=100, gridsize=3000, edge_grids=1, opt_fn=tf.optimizers.SGD)
DS1.Transform_Splines()

DS1.Filter(pair_filter[1])
#DS1.link_dataset()

DS1.ErrorDistribution_r(nbins=100, xlim=pair_filter[2], error=DS1.loc_error)
#DS1.PlotSplineGrid(gridsize=DS1.gridsize, edge_grids=DS1.edge_grids, plotarrows=False, plotmap=True)
'''