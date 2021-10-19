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
DS2=None
#%% Load datasets
if False: #% Load Beads
    DS1 = dataset(['C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5'],
                  linked=True, pix_size=159, loc_error=1.4, FrameLinking=False, FrameOptimization=False)
    DS1.load_dataset_hdf5()
    gridsize=500


if False: #% Load Clusters
    DS1 = dataset([ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
                        'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ],
                  linked=False, pix_size=159, loc_error=10, FrameLinking=False, FrameOptimization=False)
    DS1.load_dataset_hdf5(align_rcc=False)
    #DS1 = DS1.SubsetRandom(subset=0.2)
    DS1 = DS1.SubsetWindow(subset=0.2)
    DS1, DS2 = DS1.SplitDataset()
    gridsize=1000
    

if False: #% Load Excel Niekamp
    DS1 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv',
                  linked=False, pix_size=1, loc_error=1.4, FrameLinking=True, FrameOptimization=False)
    DS2 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set2/set2_beads_locs.csv',
                  linked=False, pix_size=1, loc_error=1.4, FrameLinking=True)
    DS1.load_dataset_excel()
    DS2.load_dataset_excel()
    DS1.link_dataset()
    DS2.link_dataset()
    gridsize=3000


if True: #% copy clusters
    DS1 = dataset_copy('C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5',
                  linked=False, pix_size=159, loc_error=10, FrameLinking=False, FrameOptimization=False)
    #deform=Affine_Deform()
    deform=Deform(random_deform=False, shift=None ) #,shear=None, scaling=None)
    DS1.load_copydataset_hdf5(deform)
    #DS1 = DS1.SubsetRandom(subset=0.2, linked=True)
    DS1 = DS1.SubsetWindow(subset=0.2, linked=True)
    #DS1, DS2 = DS1.SplitDataset(linked=True)
    gridsize=1000
    

if False: #% generate dataset beads
    DS1 = dataset_simulation(imgshape=[256, 512], loc_error=1.4, linked=True,
                             pix_size=159, FrameLinking=False, FrameOptimization=False)
    deform=Deform(shear=None, scaling=None, random_deform=False)
    DS1.generate_dataset_beads(N=216, deform=deform)
    DS1, DS2 = DS1.SplitDataset(linked=True)
    gridsize=200
    
    
if False: #% generate dataset clusters
    DS1 = dataset_simulation(imgshape=[256, 512], loc_error=10, linked=False, 
                             pix_size=159, FrameLinking=False, FrameOptimization=False)
    deform=Deform(random_deform=False, shift=None ) #,shear=None, scaling=None)
    DS1.generate_dataset_clusters(deform=deform)
    #DS1 = DS1.SubsetRandom(subset=0.2, linked=True)
    DS1 = DS1.SubsetWindow(subset=0.1, linked=True)
    #DS1, DS2 = DS1.SplitDataset(linked=True)
    gridsize=1000
    
    
#%% Params
pair_filter = [250, 100]
learning_rate = 1
if DS1.FrameOptimization: epochs = 3
else: epochs = 300

if not DS1.linked: # generate Neighbours
    DS1.find_neighbours(maxDistance=250)

'''
#%% Shift Transform
DS1.Train_Shift(lr=1000*learning_rate, epochs=5*epochs)
DS1.Transform_Shift()
'''
#%% Affine Transform
DS1.Filter(pair_filter[0])
DS1.Train_Affine(lr=learning_rate/10, epochs=epochs*2, opt_fn=tf.optimizers.Adam)
DS1.Transform_Affine()

#%% CatmullRomSplines
DS1.Train_Splines(lr=learning_rate/1000, epochs=epochs*3, gridsize=gridsize, edge_grids=1, opt_fn=tf.optimizers.SGD)
DS1.Transform_Splines()
#DS1.plot_SplineGrid()
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
#DS1.ErrorPlot(nbins=nbins)
DS1.ErrorDistribution_xy(nbins=nbins, xlim=xlim)
DS1.ErrorDistribution_r(nbins=nbins, xlim=xlim, error=DS1.loc_error)

if DS2 is not None: ## Coupling dataset
    if not DS2.linked: 
        try:
            DS2.relink_dataset()
        except:
            DS2.link_dataset(FrameLinking=False)
    DS2.Filter(pair_filter[1])
    #DS2.ErrorPlot(nbins=nbins)
    DS2.ErrorDistribution_xy(nbins=nbins, xlim=xlim)
    DS2.ErrorDistribution_r(nbins=nbins, xlim=xlim, error=DS2.loc_error)

## DS1 vs DS2
DS1.ErrorPlotImage(DS2)

#%% Image overview
if True:
    DS1.generate_channel(precision=DS1.pix_size*DS1.subset)
    DS1.plot_channel()
    #DS1.plot_1channel()
