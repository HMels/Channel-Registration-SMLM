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
    DS1 = dataset(['C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5'],
                  linked=True, pix_size=159, loc_error=1.4, FrameLinking=False, FrameOptimization=False)
    DS1.load_dataset_hdf5()
    DS2=None
    
    ## optimization params
    learning_rates = [1000, 1, 1e-2]
    epochs = [100, 500, 300]
    pair_filter = [1000, 10, 10]
    gridsize=500


if False: #% Load Clusters
    DS1 = dataset([ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
                        'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ],
                  linked=False, pix_size=159, loc_error=10, FrameLinking=True, FrameOptimization=False)
    DS1.load_dataset_hdf5(align_rcc=False)
    #DS1 = DS1.SubsetRandom(subset=0.2)
    #DS1 = DS1.SubsetWindow(subset=0.2)
    DS1, DS2 = DS1.SplitDataset()   
    #DS1.SplitBatches(50) 
    DS1.link_dataset(maxDist=800)
    #DS2=DS1.SimpleShift(DS2, maxDist=800)
    #DS1.FrameOptimization=True
    #DS1.find_neighbours(maxDistance=500, FrameLinking=True)
    
    ## optimization params
    learning_rates = [1000, .1, 1e-3]
    epochs = [100, None, 200]
    pair_filter = [250, 250, 250]
    gridsize=500
    

if True: #% Load Excel Niekamp
    DS1 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv',
                  linked=False, pix_size=1, loc_error=1.4, FrameLinking=True, FrameOptimization=False)
    DS2 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set2/set2_beads_locs.csv',
                  linked=False, pix_size=1, loc_error=1.4, FrameLinking=True)
    DS1.load_dataset_excel()
    DS2.load_dataset_excel()
    DS1.pix_size=159
    DS2.pix_size=DS1.pix_size
    DS1.link_dataset()
    DS2.link_dataset()
    
    ## optimization params
    learning_rates = [1e3, .1, 1e-3]
    epochs = [100, None, 200]
    pair_filter = [250, 30, 30]
    gridsize=3000


#%% running the model
DS1.TrainRegistration(learning_rates, epochs, pair_filter, gridsize=gridsize)

if DS2 is not None:
    DS2.copy_models(DS1) ## Copy all mapping parameters
    DS2.ApplyRegistration()
    

#%% output
nbins=100
xlim=pair_filter[2]
    
if not DS1.linked:
    DS1.link_dataset(maxDist=pair_filter[1], FrameLinking=True)

## DS1
DS1.ErrorPlot(nbins=nbins)
DS1.ErrorDistribution_xy(nbins=nbins, xlim=xlim, error=DS1.loc_error)
DS1.ErrorDistribution_r(nbins=nbins, xlim=xlim, error=DS1.loc_error)


#%% DS2 output
if DS2 is not None:
    if not DS2.linked: 
        DS2.link_dataset(maxDist=pair_filter[0],FrameLinking=True)
            
    DS2.Filter(pair_filter[1])
    DS2.ErrorPlot(nbins=nbins)
    DS2.ErrorDistribution_xy(nbins=nbins, xlim=xlim, error=DS1.loc_error)
    DS2.ErrorDistribution_r(nbins=nbins, xlim=xlim, error=DS2.loc_error)


#%% DS1 vs DS2
DS1.ErrorFOV(DS2)



#%% Image overview
if True:
    DS0=copy.deepcopy(DS1)
    if DS2 is not None:
        DS0.AppendDataset(DS2)
    DS0.reload_dataset()
    DS0.ApplyRegistration()
    DS0.SubsetWindow(subset=0.2)
    #DS0.Filter(pair_filter[1])
    DS0.generate_channel(precision=DS2.pix_size*DS2.subset)
    DS0.plot_channel()
    #DS0.plot_1channel()