# main.py
"""
Created on Thu Sep  9 14:55:12 2021

@author: Mels
"""
import matplotlib.pyplot as plt
import tensorflow as tf

from dataset import dataset
from dataset_simulation import dataset_simulation, dataset_copy, Deform, Affine_Deform
import time
from Channel import Channel
import numpy as np 

plt.close('all')

start=time.time()
ch1_pos = np.array([[1, 2, 3, 4, 5],[ 0.2, 0.6, 1.1, .3, 1]]).transpose()

DS1 = dataset(['C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5'],
              linked=True, pix_size=159, loc_error=1.4, FrameLinking=False, FrameOptimization=False)
DS1.ch1 =  Channel(ch1_pos, np.ones(ch1_pos.shape[0]))
DS1.ch2 =  Channel(ch1_pos+1, np.ones(ch1_pos.shape[0]))
DS1.ch20 =  Channel(ch1_pos+1, np.ones(ch1_pos.shape[0]))
DS2=None

## optimization params
learning_rates = [1000, 1, 1e-2]
epochs = [None, None, 20]
pair_filter = [10, 10]
gridsize=1


#%% Shift Transform
DS1.Train_Shift(lr=learning_rates[0], epochs=epochs[0])
DS1.Transform_Shift()

#%% Affine Transform
DS1.Filter(pair_filter[0])
DS1.Train_Affine(lr=learning_rates[1], epochs=epochs[1], opt_fn=tf.optimizers.Adam)
DS1.Transform_Affine()

#%% CatmullRomSplines
DS1.SplinesModel=None
DS1.Train_Splines(lr=learning_rates[2], epochs=epochs[2], gridsize=gridsize, edge_grids=1, opt_fn=tf.optimizers.SGD)
DS1.Transform_Splines()
DS1.plot_SplineGrid()
DS1.Filter(pair_filter[1])
print('Optimized in ',round((time.time()-start)/60,1),'minutes!')


#%% Mapping DS2 (either a second dataset or the cross validation)
if DS2 is not None:
    DS2.copy_models(DS1) ## Copy all mapping parameters
    DS2.Transform_Shift() ## Transforms
    DS2.Transform_Affine()
    DS2.Transform_Splines()


#%% output
nbins=100
xlim=pair_filter[1]
    
if not DS1.linked:
    try:
        DS1.relink_dataset()
    except:
        DS1.link_dataset(FrameLinking=False)
    DS1.Filter(pair_filter[1])

## DS1
DS1.ErrorPlot(nbins=nbins)
DS1.ErrorDistribution_xy(nbins=nbins, xlim=xlim, error=DS1.loc_error)
DS1.ErrorDistribution_r(nbins=nbins, xlim=xlim, error=DS1.loc_error)

#%% DS2 output
if DS2 is not None: ## Coupling dataset
    if not DS2.linked: 
        try:
            DS2.relink_dataset()
        except:
            DS2.link_dataset(FrameLinking=False)
    DS2.Filter(pair_filter[1])
    DS2.ErrorPlot(nbins=nbins)
    DS2.ErrorDistribution_xy(nbins=nbins, xlim=xlim, error=DS1.loc_error)
    DS2.ErrorDistribution_r(nbins=nbins, xlim=xlim, error=DS2.loc_error)

## DS1 vs DS2
DS1.ErrorPlotImage(DS2)

#%% Image overview
if True:
    DS1.generate_channel(precision=DS1.pix_size*DS1.subset)
    DS1.plot_channel()
    #DS1.plot_1channel()
