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
if False: #% Load Beads
    maxDistance=None
    DS1 = dataset(['C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5'],
                  pix_size=159, loc_error=1.4, mu=0,
                  linked=True, FrameLinking=False, BatchOptimization=False, execute_linked=True)
    DS1.load_dataset_hdf5(align_rcc=False)
    
    ## optimization params
    learning_rate = 1e-2
    epochs = 300
    pair_filter = [None, 30, 30]
    gridsize=4000
    
    
if False: #% Load Excel Niekamp
    maxDistance=1000
    k=1
    DS1 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv',
                  pix_size=1, loc_error=1.4, mu=0.3, linked=False,
                  FrameLinking=True, BatchOptimization=False, execute_linked=True)
    DS2 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set2/set2_beads_locs.csv',
                  pix_size=1, loc_error=1.4, mu=0.3, linked=False, 
                  FrameLinking=True, execute_linked=True)
    DS1.load_dataset_excel()
    DS2.load_dataset_excel()
    DS1.pix_size=159
    DS2.pix_size=DS1.pix_size
    DS1.link_dataset(maxDistance=maxDistance)
    DS2.link_dataset(maxDistance=maxDistance)
    
    ## optimization params
    learning_rate = 1e-3
    epochs = 300
    pair_filter = [250, 30, 15]
    gridsize=6500


if True: #% Load FRET clusters
    maxDistance=300
    k=1
    DS1locs = dataset(['C:/Users/Mels/Documents/example_MEP/ch0_locs_picked_clusters.hdf5', 
                   'C:/Users/Mels/Documents/example_MEP/ch1_locs_picked_clusters.hdf5'],
                  pix_size=159, loc_error=10, mu=0, imgshape=[512,256], 
                  linked=False, FrameLinking=False, BatchOptimization=False, execute_linked=True)
    DS1locs.load_dataset_hdf5(align_rcc=True, transpose=True)
    #DS1locs, DS2locs=DS1locs.SplitDatasetClusters()
    DS1=DS1locs.ClusterDataset(loc_error=None)
    #DS2=DS2locs.ClusterDataset()
    DS1.link_dataset(maxDistance=maxDistance)
    DS1,DS2=DS1.SplitDataset()
    
    ## optimization params
    learning_rate=1e-3
    epochs=None
    pair_filter=[None, None, maxDistance]
    gridsize=5000
    
    
if False: #% Load DNA-paint
    maxDistance=300
    k=1
    DS1locs = dataset(['C:/Users/Mels/Documents/DNA_PAINT/DNA_PAINT-chan1_picked.hdf5',
                       'C:/Users/Mels/Documents/DNA_PAINT/DNA_PAINT-chan2_picked.hdf5'],
                  pix_size=159, loc_error=10, mu=0, imgshape=[512,256], 
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
    
fig,ax=DS1.show_channel(DS1.ch1.pos, ps=7)
DS1.show_channel(DS1.ch2.pos, ps=7, color='blue',fig=fig, ax=ax)
#%% running the model
start=time.time()
#DS1.AffineLLS(maxDistance, k)
DS1.Filter(pair_filter[0]) 

#% CatmullRomSplines
if epochs is not None:
    DS1.Train_Splines(learning_rate, epochs, gridsize, edge_grids=1, opt_fn=tf.optimizers.SGD, 
                      maxDistance=maxDistance, k=k)
    DS1.Apply_Splines()
    
DS1.Filter(pair_filter[1])
print('Optimized in',round((time.time()-start)/60,1),'minutes!')


if DS2 is not None:
    DS2.copy_models(DS1) ## Copy all mapping parameters
    #DS2.Apply_Affine(DS1.AffineMat)
    if DS2.SplinesModel is not None: DS2.Apply_Splines()
   

#%% output
nbins=100
xlim=pair_filter[2]
    
if not DS1.linked:
    DS1.link_dataset(maxDistance=maxDistance, FrameLinking=True)

## DS1
#DS1.ErrorPlot(nbins=nbins)
DS1.ErrorDistribution_xy(nbins=nbins, xlim=xlim, error=DS1.coloc_error, fit_data=True)
#DS1.ErrorDistribution_r(nbins=nbins, xlim=xlim, error=DS1.coloc_error, mu=DS1.mu, fit_data=True)
#DS1.ErrorFOV()


#%% DS2 output
if DS2 is not None:
    if not DS2.linked: 
        DS2.link_dataset(maxDistance=maxDistance,FrameLinking=True)
            
    DS2.Filter(pair_filter[1])
    #DS2.ErrorPlot(nbins=nbins)
    DS2.ErrorDistribution_xy(nbins=nbins, xlim=xlim, error=DS2.coloc_error)
    #DS2.ErrorDistribution_r(nbins=nbins, xlim=xlim, error=DS2.coloc_error, mu=DS2.mu)
    #DS2.ErrorFOV()


#%% Image overview
fig,ax=DS1.show_channel(DS1.ch1.pos, ps=7)
DS1.show_channel(DS1.ch2.pos, ps=7, color='blue', fig=fig, ax=ax)
if False:
    DS1.generate_channel(precision=DS1.pix_size)
    DS1.plot_1channel()
    
if False and DS2 is not None:
    DS2.generate_channel(precision=DS2.pix_size)
    DS2.plot_1channel()
    
    
#%% model summary
DS1.model_summary()
if DS2 is not None: DS2.model_summary()
