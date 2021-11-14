# main.py
"""
Created on Thu Sep  9 14:55:12 2021

@author: Mels
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import copy
import time

from dataset import dataset
from dataset_simulation import dataset_simulation, dataset_copy, Deform, Affine_Deform


plt.close('all')
DS2=None
#%% Load datasets
if False: #% Load Beads
    maxDistance=None
    DS1 = dataset(['C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5'],
                  pix_size=159, loc_error=1.4, mu=0,
                  linked=True, FrameLinking=False, BatchOptimization=False)
    DS1.load_dataset_hdf5(align_rcc=False)
    
    ## optimization params
    learning_rates = [1000, 1, 1e-2]
    epochs = [100, None, 300]
    pair_filter = [None, 30, 30]
    gridsize=500


if False: #% Load Clusters
    maxDistance=800
    DS1 = dataset(['C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5', 
                   'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5'],
                  pix_size=159, loc_error=10, mu=0,
                  linked=False, FrameLinking=True, BatchOptimization=False)
    DS1.load_dataset_hdf5(align_rcc=False)
    DS1, DS2=DS1.SplitDataset()   
    DS1.link_dataset(maxDistance=maxDistance)
    
    ## optimization params
    learning_rates = [1000, .1, 1e-3]
    epochs = [100, None, 200]
    pair_filter = [250, 250, 250]
    gridsize=500
    
    
if False: #% Load Excel Niekamp
    maxDistance=1000
    DS1 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv',
                  pix_size=1, loc_error=1.4, mu=0.3,
                  linked=False, FrameLinking=True, BatchOptimization=False)
    DS2 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set2/set2_beads_locs.csv',
                  pix_size=1, loc_error=1.4, mu=0.3,
                  linked=False, FrameLinking=True)
    DS1.load_dataset_excel()
    DS2.load_dataset_excel()
    DS1.pix_size=159
    DS2.pix_size=DS1.pix_size
    DS1.link_dataset(maxDistance=maxDistance)
    DS2.link_dataset(maxDistance=maxDistance)
    
    ## optimization params
    learning_rates = [1e3, .1, 1e-3]
    epochs = [100, None, 300]
    pair_filter = [250, 30, 30]
    gridsize=6500


if False: #% DNA-PAINT
    maxDistance=2000
    DS1 = dataset(['C:/Users/Mels/Documents/DNA_PAINT/DNA_PAINT-chan1.hdf5', 
                   'C:/Users/Mels/Documents/DNA_PAINT/DNA_PAINT-chan2.hdf5'],
                  pix_size=159, loc_error=None, mu=0,
                  linked=False, FrameLinking=True, BatchOptimization=True)
    DS1.load_dataset_hdf5(align_rcc=True)
    #DS1=DS1.SubsetRandom(subset=0.1)
    #DS1=DS1.SubsetWindow(subset=0.2)
    #DS1.link_dataset(maxDistance=maxDistance)
    DS1.find_neighbours(maxDistance=maxDistance, FrameLinking=True)
    DS1.SubsetAddFrames(50, 1)
    #DS2=DS1.SimpleShift(DS2, maxDistance=800)
    
    ## optimization params
    learning_rates = [1000, 1e-3, 1e-3]
    epochs = [10, 10, None]
    pair_filter = [None, None, maxDistance]
    gridsize=1000
    
 
if True: #% Load Excel Niekamp test clusters
    maxDistance=1000
    DS1 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv',
                  pix_size=1, loc_error=1.4, mu=0.3,
                  linked=False, FrameLinking=True, BatchOptimization=True)
    DS2 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set2/set2_beads_locs.csv',
                  pix_size=1, loc_error=1.4, mu=0.3,
                  linked=False, FrameLinking=True)
    DS1.load_dataset_excel()
    DS2.load_dataset_excel()
    DS1.pix_size=159
    DS2.pix_size=DS1.pix_size
    DS1.link_dataset(maxDistance=maxDistance)
    DS2.link_dataset(maxDistance=maxDistance)
    DS2=DS1.SimpleShift(DS2, maxDistance=maxDistance)
    DS1.find_neighbours(FrameLinking=True,maxDistance=maxDistance)
    DS1.SubsetAddFrames(5, 1)
    
    ## optimization params
    learning_rates = [1e2, 1e-3, 1e-3]
    epochs = [None, 10, None]
    pair_filter = [None, None, 100]
    gridsize=6500
    
    
#%% running the model
DS1.TrainRegistration(execute_linked=False, learning_rates=learning_rates, 
                      epochs=epochs, pair_filter=pair_filter, gridsize=gridsize)

if DS2 is not None:
    DS2.copy_models(DS1) ## Copy all mapping parameters
    DS2.ApplyRegistration()
    

#%% output
nbins=100
xlim=pair_filter[2]
    
if not DS1.linked:
    DS1.link_dataset(maxDistance=maxDistance, FrameLinking=True)

## DS1
DS1.ErrorPlot(nbins=nbins)
DS1.ErrorDistribution_xy(nbins=nbins, xlim=xlim, error=DS1.loc_error, fit_data=True)
DS1.ErrorDistribution_r(nbins=nbins, xlim=xlim, error=DS1.loc_error, mu=DS1.mu, fit_data=True)


#%% DS2 output
if DS2 is not None:
    if not DS2.linked: 
        DS2.link_dataset(maxDistance=maxDistance,FrameLinking=True)
            
    DS2.Filter(pair_filter[1])
    DS2.ErrorPlot(nbins=nbins)
    DS2.ErrorDistribution_xy(nbins=nbins, xlim=xlim, error=DS2.loc_error)
    DS2.ErrorDistribution_r(nbins=nbins, xlim=xlim, error=DS2.loc_error, mu=DS2.mu)


#%% DS1 vs DS2
DS1.ErrorFOV(DS2)


#%% Image overview
if False:
    DS0=copy.deepcopy(DS1)
    if DS2 is not None:
        DS0.AppendDataset(DS2)
    DS0.reload_dataset()
    DS0.ApplyRegistration()
    #DS0.SubsetWindow(subset=0.2)
    #DS0.Filter(pair_filter[1])
    DS0.generate_channel(precision=DS0.pix_size*DS0.subset)
    DS0.plot_channel()
    #DS0.plot_1channel()
    
#%% model summary
DS1.model_summary()
if DS2 is not None: DS2.model_summary()
