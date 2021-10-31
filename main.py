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

plt.close('all')

start=time.time()
#%% Load datasets
if False: #% Load Beads
    DS1 = dataset(['C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5'],
                  linked=True, pix_size=159, loc_error=1.4, FrameLinking=False, FrameOptimization=False)
    DS1.load_dataset_hdf5()
    DS2=None
    
    ## optimization params
    learning_rates = [1000, 1, 1e-2]
    epochs = [100, 500, 300]
    pair_filter = [1000, 10]
    gridsize=500


if False: #% Load Clusters
    DS1 = dataset([ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
                        'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ],
                  linked=False, pix_size=159, loc_error=10, FrameLinking=False, FrameOptimization=True)
    DS1.load_dataset_hdf5(align_rcc=False)
    #DS1 = DS1.SubsetRandom(subset=0.2)
    DS1 = DS1.SubsetWindow(subset=0.2)
    DS1, DS2 = DS1.SplitDataset()
    DS1.find_neighbours(maxDistance=1000)
    
    ## optimization params
    learning_rates = [1000, .1, 1e-3]
    epochs = [15, 6, 9]
    pair_filter = [1000, 1000]
    gridsize=1000
    

if True: #% Load Excel Niekamp
    DS1 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv',
                  linked=False, pix_size=1, loc_error=1.4, FrameLinking=True, FrameOptimization=False)
    DS2 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set2/set2_beads_locs.csv',
                  linked=False, pix_size=1, loc_error=1.4, FrameLinking=True)
    DS1.load_dataset_excel()
    DS2.load_dataset_excel()
    DS1.pix_size=159
    DS1.link_dataset()
    DS2.link_dataset()
    
    ## optimization params
    learning_rates = [1000, .1, 1e-3]
    epochs = [200, 200, 200]
    pair_filter = [250, 30]
    gridsize=3000


if False: #% copy clusters
    DS1 = dataset_copy('C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5',
                  linked=False, pix_size=159, loc_error=10, FrameLinking=False, FrameOptimization=True)
    deform=Affine_Deform()
    #deform=Deform(random_deform=False, shift=None ) #,shear=None, scaling=None)
    DS1.load_copydataset_hdf5(deform)
    #DS1 = DS1.SubsetRandom(subset=0.2, linked=True)
    DS1 = DS1.SubsetWindow(subset=0.2, linked=True)
    DS1, DS2 = DS1.SplitDataset(linked=True)
    DS1.find_neighbours(maxDistance=1000)
    
    ## optimization params
    learning_rates = [1000, .1, 1e-4]
    epochs = [5, None, 10]
    pair_filter = [2000, 2000]
    gridsize=3000
    

if False: #% generate dataset beads
    DS1 = dataset_simulation(imgshape=[256, 512], loc_error=1.4, linked=True,
                             pix_size=159, FrameLinking=False, FrameOptimization=False)
    deform=Deform(shear=None, scaling=None, random_deform=False)
    DS1.generate_dataset_beads(N=216, deform=deform)
    #DS1, DS2 = DS1.SplitDataset(linked=True)
    DS2=None
    
    ## optimization params
    learning_rates = [1000, 1, 1e-2]
    epochs = [100, 500, 300]
    pair_filter = [1000, 10]
    gridsize=500
    
    
if False: #% generate dataset clusters
    DS1 = dataset_simulation(imgshape=[256, 512], loc_error=10, linked=False, 
                             pix_size=159, FrameLinking=False, FrameOptimization=True)
    deform=Deform(random_deform=False, shift=None ) #,shear=None, scaling=None)
    DS1.generate_dataset_clusters(deform=deform)
    #DS1 = DS1.SubsetRandom(subset=0.2, linked=True)
    DS1 = DS1.SubsetWindow(subset=0.5, linked=True)
    DS1, DS2 = DS1.SplitDataset(linked=True)
    DS1.find_neighbours(maxDistance=1000)
    
    ## optimization params
    learning_rates = [1000, 1, 1e-5]
    epochs = [5, 20, 10]
    pair_filter = [800, 400]
    gridsize=1000
    
    
#%% Shift Transform
DS1.Train_Shift(lr=learning_rates[0], epochs=epochs[0], opt_fn=tf.optimizers.Adagrad)
DS1.Transform_Shift()

#%% Affine Transform
DS1.Filter(pair_filter[0])
DS1.Train_Affine(lr=learning_rates[1], epochs=epochs[1], opt_fn=tf.optimizers.Adam)
DS1.Transform_Affine()
#DS1.PlotGridMapping(DS1.AffineModel, gridsize, edge_grids)

#%% CatmullRomSplines
DS1.Train_Splines(lr=learning_rates[2], epochs=epochs[2], gridsize=gridsize, edge_grids=1, opt_fn=tf.optimizers.SGD)
DS1.Transform_Splines()
#DS1.PlotSplineGrid(gridsize=gridsize, edge_grids=edge_grids)
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
