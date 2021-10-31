# main.py
"""
Created on Thu Sep  9 14:55:12 2021

@author: Mels
"""
import matplotlib.pyplot as plt
import tensorflow as tf

import sys
sys.path.insert(0, 'C:/Users/Mels/OneDrive/MASTER_AP/MEP/24-channel-alignment')

from dataset import dataset
from dataset_simulation import dataset_simulation, dataset_copy, Deform, Affine_Deform
from Channel import Channel
import copy
import numpy as np 

plt.close('all')

ch1_pos = np.zeros([15,2])
ch1_pos[:,0] = np.linspace(-2.5,2.5,ch1_pos.shape[0])
ch1_pos[:,1] = np.random.random(ch1_pos.shape[0])*5-2.5+np.random.randn(ch1_pos.shape[0] )*0.1
ch2_pos = np.zeros(ch1_pos.shape)+np.random.randn(ch1_pos.shape[0],2 )*0.1
ch2_pos[:,0]+=ch1_pos[:,0]*1.01 + ch1_pos[:,1]*.01 + .2
ch2_pos[:,1]+=ch1_pos[:,1]*1.002 + ch1_pos[:,0]*.3 + .1
np.random.randn()

#%%
DS1 = dataset(['C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5'],
              linked=True, pix_size=159, loc_error=1.4, FrameLinking=False, FrameOptimization=False)
DS1.ch1 = Channel(ch1_pos, np.ones(ch1_pos.shape[0]))
DS1.ch2 = Channel(ch2_pos, np.ones(ch1_pos.shape[0]))
DS1.ch20 = Channel(ch2_pos, np.ones(ch1_pos.shape[0]))
DS1.center_image()
DS2=copy.deepcopy(DS1)

## optimization params
learning_rates = [None, 1, 1e-2]
epochs = [None, 100, 100]
gridsize=1
edge_grids=1


#%% Affine Transform
DS1.PlotGridMapping(DS1.AffineModel, gridsize, edge_grids)
DS1.Train_Affine(lr=learning_rates[1], epochs=epochs[1], opt_fn=tf.optimizers.Adam)
DS1.Transform_Affine()
DS1.PlotGridMapping(DS1.AffineModel, gridsize, edge_grids)

#%% CatmullRomSplines
DS2.PlotSplineGrid(gridsize=gridsize, edge_grids=edge_grids)
DS2.Train_Splines(lr=learning_rates[2], epochs=epochs[2], gridsize=gridsize, edge_grids=edge_grids, opt_fn=tf.optimizers.SGD)
DS2.Transform_Splines()
DS2.PlotSplineGrid(gridsize=gridsize, edge_grids=edge_grids)

