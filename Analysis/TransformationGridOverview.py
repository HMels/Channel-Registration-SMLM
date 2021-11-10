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


from Align_Modules.Affine import AffineModel
from Align_Modules.Polynomial3 import Polynomial3Model
from Align_Modules.RigidBody import RigidBodyModel
from Align_Modules.Splines import CatmullRomSpline2D
from Align_Modules.Shift import ShiftModel

plt.close('all')

ch1_pos = np.zeros([15,2])
ch1_pos[:,0] = np.linspace(-2.5,2.5,ch1_pos.shape[0])
ch1_pos[:,1] = np.random.random(ch1_pos.shape[0])*5-2.5+np.random.randn(ch1_pos.shape[0] )*0.1
ch2_pos = np.zeros(ch1_pos.shape)+np.random.randn(ch1_pos.shape[0],2 )*0.1
ch2_pos[:,0]+=ch1_pos[:,0]*1.01 + ch1_pos[:,1]*.01 + .2
ch2_pos[:,1]+=ch1_pos[:,1]*1.002 + ch1_pos[:,0]*.3 + .1
np.random.randn()

ch1_pos*=10
ch2_pos*=10


#%%
DS1 = dataset(['C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5'],
              linked=True, pix_size=159, loc_error=1.4, FrameLinking=False, BatchOptimization=False)
DS1.ch1 = Channel(ch1_pos, np.ones(ch1_pos.shape[0]))
DS1.ch10 = Channel(ch1_pos, np.ones(ch1_pos.shape[0]))
DS1.ch2 = Channel(ch2_pos, np.ones(ch1_pos.shape[0]))
DS1.ch20 = Channel(ch2_pos, np.ones(ch1_pos.shape[0]))
DS1.ch20linked = Channel(ch2_pos, np.ones(ch1_pos.shape[0]))
DS1.center_image()
DS2=copy.deepcopy(DS1)

## optimization params
learning_rates = [None, 1, 1e-2]
epochs = [None, 100, 100]
gridsize=10
edge_grids=0

'''
#%% Affine Transform
DS1.AffineModel=AffineModel()
DS1.PlotGridMapping(DS1.AffineModel, gridsize, edge_grids)
DS1.Train_Model(DS1.AffineModel, lr=learning_rates[1], epochs=epochs[1], opt_fn=tf.optimizers.Adam)
DS1.Transform_Model(DS1.AffineModel)
DS1.PlotGridMapping(DS1.AffineModel, gridsize, edge_grids)
'''

#%% CatmullRomSplines
ch1_input,ch2_input=DS2.InitializeSplines(gridsize=gridsize, edge_grids=edge_grids)
DS2.SplinesModel=CatmullRomSpline2D(DS2.ControlPoints)
DS2.PlotSplineGrid()
DS2.Train_Model(DS2.SplinesModel, lr=learning_rates[2], epochs=epochs[2], opt_fn=tf.optimizers.SGD, 
                 ch1=ch1_input, ch2=ch2_input)        
DS2.ControlPoints = DS2.SplinesModel.ControlPoints
DS2.ch2.pos.assign(DS2.InputSplines(
    DS2.Transform_Model(DS2.SplinesModel, ch2=DS2.InputSplines(DS2.ch2.pos)),
    inverse=True))
DS2.PlotSplineGrid()
