# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:02:19 2021

@author: Mels
"""
# main.py
"""
Created on Thu Sep  9 14:55:12 2021

@author: Mels
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy.random as rnd
import numpy as np
import copy

from dataset import dataset
from dataset_simulation import dataset_simulation, dataset_copy, Deform, Affine_Deform
import time
from Channel import Channel
from Align_Modules.Splines import CatmullRomSpline2D


plt.close('all')
#%% Cluster_fn
def register_cluster(Dataset, learning_rates=[1000,.1,1e-3], epochs=[100,100,100],
                     pair_filter=[1000,1000], gridsize=5):
    #Dataset, CoM=center_mass(Dataset)
    Dataset.center_image()
    
    #% Shift Transform
    Dataset.Train_Shift(lr=learning_rates[0], epochs=epochs[0], opt_fn=tf.optimizers.Adagrad)
    Dataset.Transform_Shift()
    
    #% Affine Transform
    #Dataset.Filter(pair_filter[0])
    Dataset.Train_Affine(lr=learning_rates[1], epochs=epochs[1], opt_fn=tf.optimizers.Adam)
    Dataset.Transform_Affine()
    
    #% CatmullRomSplines
    Dataset.Train_Splines(lr=learning_rates[2], epochs=epochs[2], gridsize=gridsize, edge_grids=1, opt_fn=tf.optimizers.SGD)
    Dataset.Transform_Splines()
    #Dataset.Filter(pair_filter[1])
    #Dataset.uncenter_mass(Dataset, CoM)
    return Dataset


def center_mass(Dataset):
    CoM = ( (tf.reduce_sum(Dataset.ch1.pos, axis=0)+tf.reduce_sum(Dataset.ch2.pos, axis=0))/
           (Dataset.ch1.pos.shape[0]+Dataset.ch2.pos.shape[0]) )
    Dataset.ch1.pos.assign(Dataset.ch1.pos-CoM)
    Dataset.ch2.pos.assign(Dataset.ch2.pos-CoM)
    #Dataset.ch20.pos.assign(Dataset.ch20.pos-CoM)
    Dataset.img, Dataset.imgsize, Dataset.mid = Dataset.imgparams() 
    return Dataset, CoM


def uncenter_mass(Dataset, CoM):
    Dataset.ch1.pos.assign(Dataset.ch1.pos+CoM)
    Dataset.ch2.pos.assign(Dataset.ch2.pos+CoM)
    #Dataset.ch20.pos.assign(Dataset.ch20.pos+CoM)
    Dataset.img, Dataset.imgsize, Dataset.mid = Dataset.imgparams() 
    return Dataset


#%% generate cluster
def generate_cluster(Nclust=500, mu=[16000, 16000], sigma=[20,20], loc_error=1.4):
    Dataset=dataset_cluster_simulation(pix_size=159, linked=False, loc_error=loc_error, imgshape=[512, 512])

    pos1=gauss_2d(mu,sigma, Nclust)
    pos2=copy.deepcopy(pos1)
    pos1=generate_locerror(pos1, loc_error)
    pos2=generate_locerror(pos2, loc_error)
    frame=np.ones(pos1.shape[0],dtype=np.float32)
    
    pos2=Affine_Deform().deform(pos2)
    #pos2=Deform(shift=[100,100], rotation=0.5, scaling=[1.05, 1.02]).deform(pos2)
    
    Dataset.ch1=Channel(pos1, frame)
    Dataset.ch2=Channel(pos2, frame)
    Dataset.ch20=copy.deepcopy(Dataset.ch2)
    Dataset=SplinesDeform(Dataset, gridsize=3, error=5*loc_error)
    
    Dataset.img, Dataset.imgsize, Dataset.mid = Dataset.imgparams() 
    return Dataset

    
class dataset_cluster_simulation(dataset):
    def __init__(self, pix_size=1, linked=False, loc_error=10, imgshape=[512, 512], 
                 FrameLinking=False, FrameOptimization=False):
        self.pix_size=pix_size    # the multiplicationfactor to change the dataset into units of nm
        self.imgshape=imgshape    # number of pixels of the dataset
        self.loc_error=loc_error  # localization error
        self.linked=linked        # is the data linked/paired?
        self.FrameLinking=FrameLinking              # will the dataset be linked or NN per frame?
        self.FrameOptimization=FrameOptimization    # will the dataset be optimized per frame
        dataset.__init__(self, path=None,pix_size=pix_size,linked=linked,imgshape=imgshape,
                         FrameLinking=FrameLinking, loc_error=loc_error, FrameOptimization=FrameOptimization)
    
    def relink_dataset(self):
        self.linked=True     
    
    
#%% miscalleneous functions
def SplinesDeform(Dataset, gridsize=3000, edge_grids=1, error=10):
    Dataset.edge_grids=edge_grids
    Dataset.gridsize=gridsize
    ControlPoints=Dataset.generate_CPgrid(gridsize, edge_grids)
    
    ControlPoints=ControlPoints.numpy()
    ControlPoints+=rnd.randn(ControlPoints.shape[0],ControlPoints.shape[1],ControlPoints.shape[2])*error/gridsize
    ControlPoints[::2, ::2,:]+=error/gridsize
    ControlPoints[1::2, 1::2,:]-=error/gridsize
    ControlPoints=tf.Variable(ControlPoints, trainable=False, dtype=tf.float32)
    
    Dataset.SplinesModel=CatmullRomSpline2D(ControlPoints)
    Dataset.Transform_Splines()
    Dataset.SplinesModel=None
    Dataset.gridsize=None
    Dataset.edge_grids=None
    return Dataset

def generate_locerror(pos, loc_error):
 # Generates a Gaussian localization error over the localizations
     if loc_error != 0:
         pos[:,0] += rnd.normal(0, loc_error, pos.shape[0])
         pos[:,1] += rnd.normal(0, loc_error, pos.shape[0])
     return pos 
    
def gauss_2d(mu, sigma, N):
    '''
    Generates a 2D gaussian cluster
    Parameters
    ----------
    mu : 2 float array
        The mean location of the cluster.
    sigma : 2 float array
        The standard deviation of the cluster.
    N : int
        The number of localizations.
    Returns
    -------
    Nx2 float Array
        The [x1,x2] localizations .
    '''
    x1 = np.float32( rnd.normal(mu[0], sigma[0], N) )
    x2 = np.float32( rnd.normal(mu[1], sigma[1], N) )
    return np.array([x1, x2]).transpose()
    

#%% parameters
learning_rates=[1000,.01,1e-3]
epochs=[100,None,100]
pair_filter=[None,None]
gridsize=3

DS1=generate_cluster()
DS2=None
#DS1,DS2=DS1.SplitDataset()
DS1.find_neighbours(maxDistance=1000)

#%% optimization
start=time.time()
DS1=register_cluster(DS1, learning_rates=learning_rates, epochs=epochs, pair_filter=pair_filter, gridsize=gridsize)
print('Optimized in',round((time.time()-start)/60,1),'minutes!')


#%% Mapping DS2 (either a second dataset or the cross validation)
if DS2 is not None:
    DS2.copy_models(DS1) ## Copy all mapping parameters
    DS2.Transform_Shift() ## Transforms
    DS2.Transform_Affine()
    DS2.Transform_Splines()


#%% output
nbins=100
xlim=100
    
if not DS1.linked:
    DS1.relink_dataset()

## DS1
#DS1.ErrorPlot(nbins=nbins)
DS1.ErrorDistribution_xy(nbins=nbins, xlim=xlim, error=DS1.loc_error)
DS1.ErrorDistribution_r(nbins=nbins, xlim=xlim, error=DS1.loc_error)

#%% DS2 output
if DS2 is not None: ## Coupling dataset
    DS2.relink_dataset()
    #DS2.Filter(pair_filter[1])
    #DS2.ErrorPlot(nbins=nbins)
    #DS2.ErrorDistribution_xy(nbins=nbins, xlim=xlim, error=DS1.loc_error)
    DS2.ErrorDistribution_r(nbins=nbins, xlim=xlim, error=DS2.loc_error)

## DS1 vs DS2
DS1.ErrorPlotImage(DS2)

#%% Image overview
if True:
    DS1.generate_channel(precision=5, heatmap=True)
    DS1.plot_channel()
