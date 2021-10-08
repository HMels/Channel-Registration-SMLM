# main.py
"""
Created on Thu Sep  9 14:55:12 2021

@author: Mels
"""
import matplotlib.pyplot as plt

from dataset import dataset
from dataset_simulation import dataset_simulation, Deform
import time

plt.close('all')

start=time.time()
DS2=None
#%% Load datasets
if False: #% Load Beads
    DS1 = dataset(['C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5'],
                  linked=True, pix_size=159, FrameLinking=False, FrameOptimization=False)
    DS1.load_dataset_hdf5()
    gridsize=500


if False: #% Load Clusters
    DS1 = dataset([ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
                        'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ],
                  linked=False, pix_size=159, FrameLinking=False, FrameOptimization=False)
    DS1.load_dataset_hdf5(align_rcc=False)
    #DS1 = DS1.SubsetRandom(subset=0.2)
    DS1 = DS1.SubsetWindow(subset=0.2)
    DS1, DS2 = DS1.SplitDataset()
    gridsize=1000
    

if False: #% Load Excel
    DS1 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv',
                  linked=False, pix_size=1, FrameLinking=True, FrameOptimization=False)
    DS2 = dataset('C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set2/set2_beads_locs.csv',
                  linked=False, pix_size=1, FrameLinking=True)
    DS1.load_dataset_excel()
    DS2.load_dataset_excel()
    DS1.link_dataset()
    gridsize=3000


if False: #% generate dataset beads
    DS1 = dataset_simulation(imgshape=[256, 512], linked=True, pix_size=159, FrameLinking=False, FrameOptimization=False)
    deform=Deform(shear=None, scaling=None, random_deform=False)
    DS1.generate_dataset_beads(N=216, error=10, noise=0, deform=deform)
    DS1, DS2 = DS1.SplitDataset(linked=True)
    gridsize=200
    
    
if True: #% generate dataset clusters
    DS1 = dataset_simulation(imgshape=[256, 512], linked=False, pix_size=159, FrameLinking=False, FrameOptimization=False)
    deform=Deform(random_deform=False ) #,shear=None, scaling=None)
    DS1.generate_dataset_clusters(deform=deform, noise=0)
    #DS1 = DS1.SubsetRandom(subset=0.2, linked=True)
    DS1 = DS1.SubsetWindow(subset=0.2, linked=True)
    #DS1, DS2 = DS1.SplitDataset(linked=True)
    gridsize=1000
    
    
#%% Params
pair_filter = [1000, 1000]
learning_rate = 100
if DS1.FrameOptimization: epochs = 1
else: epochs = 100

if not DS1.linked: # generate Neighbours
    DS1.find_neighbours(maxDistance=2000, k=30)


#%% Shift Transform
DS1.Train_Shift(lr=100*learning_rate, epochs=5*epochs)
DS1.Transform_Shift()
'''
#%% Affine Transform
DS1.Filter(pair_filter[0])
DS1.AffineModel=None
DS1.Train_Affine(lr=learning_rate/100, epochs=epochs*2)
DS1.Transform_Affine()

#%% CatmullRomSplines
DS1.Train_Splines(lr=1e-2*learning_rate, epochs=epochs*2, gridsize=gridsize, edge_grids=1)
DS1.Transform_Splines()
#DS1.plot_SplineGrid()
DS1.Filter(pair_filter[1])
print('Optimized in ',round(time.time()-start,1),'seconds!')
'''

#%% Mapping DS2 (either a second dataset or the cross validation)
if DS2 is not None:
    DS2.copy_models(DS1) ## Copy all mapping parameters
    DS2.Transform_Shift() ## Transforms
    DS2.Transform_Affine()
    DS2.Transform_Splines()


#%% output
nbins=100
xlim=pair_filter[1]

DS1.relink_dataset()
if DS2 is not None: DS2.relink_dataset()    
    
if not DS1.linked:
    DS1.link_dataset(FrameLinking=False)
    DS1.Filter(pair_filter[1])

## DS1
DS1.ErrorPlot(nbins=nbins)
DS1.ErrorDistribution_xy(nbins=nbins, xlim=xlim)

if DS2 is not None: ## Coupling dataset
    if not DS2.linked: DS2.link_dataset()
    DS2.Filter(pair_filter[1])
    DS2.ErrorPlot(nbins=nbins)
    DS2.ErrorDistribution_xy(nbins=nbins, xlim=xlim)

## DS1 vs DS2
DS1.ErrorPlotImage(DS2)

#%% Image overview
if True:
    DS1.generate_channel(precision=DS1.pix_size/10)
    DS1.plot_channel()
    #DS1.plot_1channel()