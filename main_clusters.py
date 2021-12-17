# main.py
"""
Created on Thu Sep  9 14:55:12 2021

@author: Mels
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import copy
import time
import numpy as np

from dataset import dataset
from dataset_simulation import dataset_simulation, dataset_copy, Deform, Affine_Deform


plt.close('all')
DS2=None
#%% Load datasets
if True: #% Load FRET clusters
    maxDistance=300
    k=8
    DS1locs = dataset(['C:/Users/Mels/Documents/example_MEP/ch0_locs_picked_clusters.hdf5', 
                   'C:/Users/Mels/Documents/example_MEP/ch1_locs_picked_clusters.hdf5'],
                  pix_size=159, loc_error=10, mu=0, imgshape=[256,512], 
                  linked=False, FrameLinking=False, BatchOptimization=False, execute_linked=False)
    DS1locs.load_dataset_hdf5(align_rcc=True, transpose=False)
    DS1locs, DS2locs=DS1locs.SplitDatasetClusters()
    DS1=DS1locs.ClusterDataset(loc_error=None)
    DS2=DS2locs.ClusterDataset(loc_error=None)
    DS1.link_dataset(maxDistance=maxDistance)
    
    ## optimization params
    learning_rate1=2e-3
    epochs1=1000
    pair_filter1=[None, None]
    gridsize1=5000
    
    learning_rate2=2e-3
    epochs2=1000
    pair_filter2=[None, None, maxDistance]
    gridsize2=5000
    
    
if False: #% Load DNA_PAINT
    maxDistance=300
    k=1
    DS1locs = dataset(['C:/Users/Mels/Documents/DNA_PAINT/DNA_PAINT-chan1_picked.hdf5',
                       'C:/Users/Mels/Documents/DNA_PAINT/DNA_PAINT-chan2_picked.hdf5'],
                  pix_size=159, loc_error=10, mu=0, imgshape=[256,512], 
                  linked=False, FrameLinking=False, BatchOptimization=False, execute_linked=True)
    DS1locs.load_dataset_hdf5(align_rcc=True, transpose=False)
    #DS1locs, DS2locs=DS1locs.SplitDatasetClusters()
    DS1=DS1locs.ClusterDataset(loc_error=None)
    #DS2=DS2locs.ClusterDataset()
    DS1.link_dataset(maxDistance=maxDistance)
    DS1,DS2=DS1.SplitDataset()
    
    ## optimization params
    learning_rate=5e-5
    epochs=None
    pair_filter=[None, None, maxDistance]
    gridsize=1000
    

#%% running the model of Clusters
fig=plt.figure(figsize=(12,6))
ax1=fig.add_subplot(141)
fig,ax1=DS1.show_channel(DS1locs.ch1.pos, ps=7, fig=fig, ax=ax1)
DS1locs.show_channel(DS1locs.ch2.pos, ps=7, color='blue', alpha=.7,fig=fig, ax=ax1)

start=time.time()
DS1.AffineLLS(maxDistance, k)
DS1.Filter(pair_filter1[0]) 

#% CatmullRomSplines
if epochs1 is not None:
    DS1.Train_Splines(learning_rate1, epochs1, gridsize1, edge_grids=1, opt_fn=tf.optimizers.SGD, 
                      maxDistance=maxDistance, k=k)
    DS1.Apply_Splines()
DS1.Filter(pair_filter1[1])


#%% applying data
DS1locs.copy_models(DS1) ## Copy all mapping parameters
DS1locs.Apply_Affine(DS1.AffineMat)
if DS1locs.SplinesModel is not None: DS1locs.Apply_Splines()

ax2=fig.add_subplot(142)
fig,ax2=DS1.show_channel(DS1locs.ch1.pos, ps=7,fig=fig, ax=ax2)
DS1locs.show_channel(DS1locs.ch2.pos, ps=7, color='blue', alpha=.7,fig=fig, ax=ax2)

if DS2 is not None:
    DS2locs.copy_models(DS1) ## Copy all mapping parameters
    DS2locs.Apply_Affine(DS1.AffineMat)
    if DS2locs.SplinesModel is not None: DS2locs.Apply_Splines()
   

#%% running the model on the data
DS1locs.link_dataset(maxDistance=maxDistance)
DS1locs.AffineLLS(maxDistance, k)
DS1locs.Filter(pair_filter2[0]) 

#% CatmullRomSplines
if epochs2 is not None:
    DS1locs.Train_Splines(learning_rate2, epochs2, gridsize2, edge_grids=1, opt_fn=tf.optimizers.SGD, 
                      maxDistance=maxDistance, k=k)
    DS1locs.Apply_Splines()
DS1locs.Filter(pair_filter2[1])


if DS2locs is not None:
    DS2locs.copy_models(DS1locs) ## Copy all mapping parameters
    DS2locs.Apply_Affine(DS1locs.AffineMat)
    if DS2locs.SplinesModel is not None: DS2locs.Apply_Splines()

print('Optimized in',round((time.time()-start)/60,1),'minutes!')

ax3=fig.add_subplot(143)
fig,ax3=DS1locs.show_channel(DS1locs.ch1.pos, ps=7,fig=fig, ax=ax3)
DS1locs.show_channel(DS1locs.ch2.pos, ps=7, color='blue', alpha=.7,fig=fig, ax=ax3)


#%% reload and print
DS1locs.reload_dataset()
DS1locs.copy_models(DS1) ## Copy all mapping parameters
DS1locs.Apply_Affine(DS1.AffineMat)
if DS1locs.SplinesModel is not None: DS1locs.Apply_Splines()

DS1locs.copy_models(DS1locs) ## Copy all mapping parameters
DS1locs.Apply_Affine(DS1locs.AffineMat)
if DS1locs.SplinesModel is not None: DS1locs.Apply_Splines()

ax4=fig.add_subplot(144)
fig,ax4=DS1locs.show_channel(DS1locs.ch1.pos, ps=7,fig=fig, ax=ax4)
DS1locs.show_channel(DS1locs.ch2.pos, ps=7, color='blue', alpha=.7,fig=fig, ax=ax4)


#%% output
nbins=100
xlim=pair_filter2[2]
    
if not DS1locs.linked:
    DS1locs.link_dataset(maxDistance=maxDistance, FrameLinking=True)

## DS1
#DS1locs.ErrorPlot(nbins=nbins)
DS1locs.ErrorDistribution_xy(nbins=nbins, xlim=xlim, error=DS1locs.coloc_error, fit_data=True)
#DS1locs.ErrorDistribution_r(nbins=nbins, xlim=xlim, error=DS1locs.coloc_error, mu=DS1.mu, fit_data=True)
#DS1locs.ErrorFOV()


#%% DS2 output
if DS2locs is not None:
    if not DS2locs.linked: 
        DS2locs.link_dataset(maxDistance=maxDistance,FrameLinking=True)
            
    DS2locs.Filter(pair_filter2[1])
    #DS2locs.ErrorPlot(nbins=nbins)
    DS2locs.ErrorDistribution_xy(nbins=nbins, xlim=xlim, error=DS2locs.coloc_error)
    #DS2locs.ErrorDistribution_r(nbins=nbins, xlim=xlim, error=DS2locs.coloc_error, mu=DS2.mu)
    #DS2locs.ErrorFOV()


#%% Image overview
FIG,AX=DS1locs.show_channel(DS1locs.ch1.pos, ps=7)
DS1locs.show_channel(DS1locs.ch2.pos, ps=7, color='blue', alpha=.7, fig=FIG, ax=AX)
if False:
    DS1locs.generate_channel(precision=DS1.pix_size)
    DS1locs.plot_1channel()
    
if False and DS2 is not None:
    DS2locs.generate_channel(precision=DS2.pix_size)
    DS2locs.plot_1channel()
    
    
#%% model summary
DS1locs.model_summary()
if DS2locs is not None: DS2locs.model_summary()

